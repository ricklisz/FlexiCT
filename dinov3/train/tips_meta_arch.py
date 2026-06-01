# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# TIPS: Text-Image Pretraining with Spatial awareness
# This module implements TIPS by combining DINOv3 SSL losses with CLIP contrastive loss.
#
# Key differences from standard DINOv2/DINOv3:
# 1. Uses SINGLE global crop instead of 2 (as per TIPS paper Section 3.2)
# 2. Adds CLIP contrastive loss for vision-language alignment

import logging
import math
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dinov3.data.augmentations import RandomCropTIPS3D
import dinov3.distributed as distributed
from dinov3.eval.text.clip_loss import memory_efficient_clip_loss
from dinov3.eval.text.text_tower import TextTower, build_text_model
from dinov3.eval.text.hf_text_tower import build_hf_text_model, HFTextTower
from dinov3.eval.text.tokenizer import get_tokenizer
from dinov3.fsdp.text_fsdp import wrap_text_model_fsdp
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.param_groups import get_params_groups_with_decay_fsdp, fuse_params_groups

logger = logging.getLogger("dinov3")

# TIPS uses exactly 1 global crop (hardcoded as per TIPS paper)
TIPS_N_GLOBAL_CROPS = 1


class TIPSMetaArch(SSLMetaArch):
    """
    TIPS: Text-Image Pretraining with Spatial awareness
    
    This class extends SSLMetaArch (DINOv3) by adding vision-language alignment
    via CLIP contrastive loss. It combines:
    - DINO loss (self-distillation on CLS tokens)
    - iBOT loss (masked image modeling on patch tokens)
    - KoLeo loss (uniformity regularization)
    - Gram loss (optional feature alignment)
    - CLIP loss (NEW: vision-language contrastive alignment)
    
    Key architectural difference from DINOv2:
    Uses SINGLE global crop (not 2) for self-distillation - increases throughput by 25%
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # TIPS: hardcoded 1 global crop (as per TIPS paper Section 3.2)
        self.n_global_crops = TIPS_N_GLOBAL_CROPS
        logger.info(f"OPTIONS -- TIPS -- n_global_crops: {self.n_global_crops} (hardcoded)")
        
        # VLM configuration
        self.vlm_enabled = cfg.vlm.enabled if hasattr(cfg, 'vlm') else False
        
        if self.vlm_enabled:
            logger.info("OPTIONS -- VLM (TIPS)")
            logger.info(f"OPTIONS -- VLM -- enabled: {cfg.vlm.enabled}")
            logger.info(f"OPTIONS -- VLM -- loss_weight: {cfg.vlm.loss_weight}")
            logger.info(f"OPTIONS -- VLM -- embed_dim: {cfg.vlm.embed_dim}")
            logger.info(f"OPTIONS -- VLM -- use_projection_head: {cfg.vlm.use_projection_head}")
            
            self.vlm_loss_weight = cfg.vlm.loss_weight
            self.vlm_embed_dim = cfg.vlm.embed_dim
            self.vlm_use_projection_head = cfg.vlm.use_projection_head
            self.vlm_init_logit_scale = cfg.vlm.init_logit_scale
            self.vlm_freeze_logit_scale = cfg.vlm.freeze_logit_scale
            
            # Determine text encoder type: HuggingFace or custom
            self.use_hf_text_encoder = cfg.vlm.get("use_hf_text_encoder", False)
            self.vlm_hf_sharded_load = cfg.vlm.get(
                "hf_sharded_load", self.use_hf_text_encoder
            )
            self._text_model_mesh = None
            self._text_model_process_group = None
            
            if self.use_hf_text_encoder:
                # HuggingFace text encoder (e.g., Qwen3-Embedding)
                hf_model_name = cfg.vlm.get("hf_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B")
                logger.info(f"OPTIONS -- VLM -- Using HuggingFace text encoder: {hf_model_name}")
                
                self.text_model = build_hf_text_model(
                    model_name_or_path=hf_model_name,
                    embed_dim=cfg.vlm.embed_dim,
                    pooling_type=cfg.vlm.get("hf_pooling_type", "last_token"),
                    use_flash_attention=cfg.vlm.get("hf_use_flash_attention", True),
                    torch_dtype=cfg.vlm.get("hf_torch_dtype", "bfloat16"),
                    freeze_backbone=cfg.vlm.get("text_model_freeze_backbone", False),
                    use_projection=cfg.vlm.get("hf_use_projection", True),
                    max_length=cfg.vlm.get("hf_max_length", 512),
                    padding_side=cfg.vlm.get("hf_padding_side", "left"),
                )
                # HF text tower has its own tokenizer
                self.tokenizer = None  # Will be obtained from text_model
                self.vlm_text_vocab_path = None
                self.vlm_text_backbone_pretrained_weights = None
            else:
                # Custom text transformer
                logger.info(f"OPTIONS -- VLM -- Using custom text backbone: {cfg.vlm.text_backbone_config}")
                
                self.text_model = build_text_model(
                    embed_dim=cfg.vlm.embed_dim,
                    backbone_model_config=cfg.vlm.text_backbone_config,
                    freeze_backbone=cfg.vlm.text_model_freeze_backbone,
                    num_head_blocks=cfg.vlm.text_model_num_head_blocks,
                    head_blocks_is_causal=cfg.vlm.text_model_head_blocks_is_causal,
                    head_blocks_drop_prob=cfg.vlm.text_model_head_blocks_drop_prob,
                    tokens_pooler_type=cfg.vlm.text_model_tokens_pooler_type,
                    use_linear_projection=cfg.vlm.text_model_use_linear_projection,
                )
                # Store tokenizer path for later initialization
                self.vlm_text_vocab_path = cfg.vlm.text_vocab_path_or_url
                self.tokenizer = None
                # Text backbone pretrained weights path
                self.vlm_text_backbone_pretrained_weights = cfg.vlm.text_backbone_pretrained_weights
            
            # Learnable temperature parameter for CLIP loss
            self.logit_scale = nn.Parameter(torch.empty(1))
            if self.vlm_freeze_logit_scale:
                self.logit_scale.requires_grad = False
            
            # Vision projection head to align vision features to VLM embedding space
            # Like DINOTxt: concatenate [CLS, mean(patch)] -> 2*embed_dim, then project to vlm_embed_dim
            vision_input_dim = 2 * self.embed_dim  # CLS + mean(patch) concatenated
            if self.vlm_use_projection_head:
                self.vlm_vision_projection = nn.Linear(vision_input_dim, self.vlm_embed_dim, bias=False)
                logger.info(f"OPTIONS -- VLM -- vision projection: {vision_input_dim} (CLS+patch) -> {self.vlm_embed_dim}")
            else:
                self.vlm_vision_projection = nn.Identity()
                assert vision_input_dim == self.vlm_embed_dim, (
                    f"Vision input dim ({vision_input_dim}) must match VLM embed_dim ({self.vlm_embed_dim}) "
                    "when use_projection_head is False"
                )
            
    def init_weights(self, *, resume: bool = False) -> None:
        """Initialize weights for both SSL and VLM components."""
        super().init_weights()
        
        if self.vlm_enabled:
            logger.info("Initializing VLM (TIPS) weights...")
            
            # Initialize logit scale
            nn.init.constant_(self.logit_scale, self.vlm_init_logit_scale)
            logger.info(f"Initialized logit_scale to {self.vlm_init_logit_scale}")
            
            # Initialize text model (loads pretrained HF backbone via from_pretrained)
            if not resume:
                use_sharded = (
                    self.use_hf_text_encoder
                    and self.vlm_hf_sharded_load
                    and dist.is_initialized()
                )
                if use_sharded:
                    self.text_model.init_weights(
                        sharded=True,
                        world_mesh=self._text_model_mesh,
                        process_group=self._text_model_process_group,
                        src_rank=0,
                    )
                else:
                    self.text_model.init_weights()
            
            # Initialize vision projection
            if isinstance(self.vlm_vision_projection, nn.Linear):
                nn.init.normal_(
                    self.vlm_vision_projection.weight,
                    std=self.vlm_vision_projection.in_features ** -0.5,
                )
            
            # Load pretrained text backbone weights if provided (only for custom text model)
            if not self.use_hf_text_encoder and self.vlm_text_backbone_pretrained_weights:
                logger.info(f"Loading pretrained text backbone from: {self.vlm_text_backbone_pretrained_weights}")
                state_dict = torch.load(self.vlm_text_backbone_pretrained_weights, map_location="cpu")
                if "text_model" in state_dict:
                    state_dict = state_dict["text_model"]
                self.text_model.load_state_dict(state_dict, strict=False)

    def init_weights_no_fsdp(self) -> None:
        """Initialize weights without FSDP for both SSL and VLM components."""
        super().init_weights_no_fsdp()
        
        if self.vlm_enabled:
            logger.info("Initializing VLM (TIPS) weights (no FSDP)...")
            
            nn.init.constant_(self.logit_scale, self.vlm_init_logit_scale)
            self.text_model.init_weights()
            
            if isinstance(self.vlm_vision_projection, nn.Linear):
                nn.init.normal_(
                    self.vlm_vision_projection.weight,
                    std=self.vlm_vision_projection.in_features ** -0.5,
                )
            # Load pretrained text backbone weights if provided (only for custom text model)
            if not self.use_hf_text_encoder and self.vlm_text_backbone_pretrained_weights:
                logger.info(f"Loading pretrained text backbone from: {self.vlm_text_backbone_pretrained_weights}")
                state_dict = torch.load(self.vlm_text_backbone_pretrained_weights, map_location="cpu")
                if "text_model" in state_dict:
                    state_dict = state_dict["text_model"]
                self.text_model.load_state_dict(state_dict, strict=False)

    def get_tokenizer(self):
        """Get or initialize the tokenizer."""
        if self.tokenizer is None and self.vlm_enabled:
            if self.use_hf_text_encoder:
                # HuggingFace text encoder has its own tokenizer
                self.tokenizer = self.text_model.get_tokenizer()
            else:
                # Custom text encoder uses BPE tokenizer
                self.tokenizer = get_tokenizer(self.vlm_text_vocab_path)
        return self.tokenizer

    def build_3D_data_augmentation(self, cfg):
        return RandomCropTIPS3D(
            local_views_scale = cfg.crops.local_views_scale,
            local_crops_number = cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            use_intensity_transforms=False
        )
        
    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        """
        Forward pass and backward pass for TIPS training.
        
        Key difference from DINOv2: Uses 1 global crop (hardcoded as per TIPS paper).
        
        Args:
            data: Dictionary containing:
                - collated_global_crops: Global crop images [1*B, C, H, W]
                - collated_local_crops: Local crop images [n_local*B, C, H, W]
                - collated_masks: Masks for iBOT
                - mask_indices_list: Indices of masked patches
                - masks_weight: Weights for masked patches
                - n_masked_patches: Number of masked patches
                - text_tokens: Tokenized captions [B, seq_len] (NEW for TIPS)
            teacher_temp: Teacher temperature for DINO/iBOT losses
            iteration: Current training iteration
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        del ignored_kwargs
        metrics_dict = {}

        # TIPS: hardcoded 1 global crop
        n_global_crops = TIPS_N_GLOBAL_CROPS
        n_local_crops = self.n_local_crops
        B = data["collated_local_crops"].shape[0] // n_local_crops
        
        # Validate batch dimensions
        expected_global_shape = n_global_crops * B
        actual_global_shape = data["collated_global_crops"].shape[0]
        assert actual_global_shape == expected_global_shape, (
            f"TIPS expects {expected_global_shape} global crops "
            f"(n_global_crops={n_global_crops} * B={B}), got {actual_global_shape}. "
            f"Make sure to use DataAugmentationTIPS and collate_data_and_cast_tips."
        )
        
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]
        metrics_dict["n_global_crops"] = n_global_crops

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        # Determine crop size
        if global_crops.dim() == 4:
            student_global_crops_size = global_crops.shape[-1]
        elif global_crops.dim() == 5:
            student_global_crops_size = global_crops.shape[-3:]
        else:
            raise ValueError(f"Unexpected global_crops dim={global_crops.dim()} shape={tuple(global_crops.shape)}")

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data, (
                "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            )
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
        else:
            gram_teacher_crops = None

        # Teacher output
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output
        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        # Gram output
        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=student_global_crops_size,
            )
        else:
            gram_global = {}

        # Get text tokens if VLM is enabled

        text_tokens = data["text_tokens"]
        # Move BatchEncoding to GPU (must assign result - .to() returns new object)
        text_tokens = text_tokens.to(device=global_crops.device)
        
        # Compute losses
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
            text_tokens=text_tokens,
            n_global_crops=n_global_crops,
        )

        self.backprop_loss(loss_accumulator)

        return loss_accumulator, metrics_dict | loss_dict

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
        text_tokens=None,
        n_global_crops=None,
    ):
        """
        Compute all losses including SSL losses and CLIP loss.
        
        TIPS-specific: No DINO global loss (only 1 global crop, so no cross-view comparison).
        """
        loss_dict = {}
        loss_accumulator = 0.0

        # DINO local loss: student(local crops) vs. teacher(global crop)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * local_weight * dino_local_crops_loss

        # TIPS: No DINO global loss (only 1 global crop, no cross-view self-distillation)
        loss_dict["dino_global_crops_loss"] = torch.tensor(0.0, device=student_local["cls_after_head"].device)

        # KoLeo loss: Not used in TIPS paper, but can be optionally enabled
        # KoLeo operates on batch dimension (uniformity across samples), so it works with 1 global crop
        # Set dino.koleo_loss_weight: 0.0 in config to disable (recommended for faithful TIPS)
        if self.dino_koleo_loss_weight > 0:
            koleo_loss = self.koleo_loss(student_global["cls_pre_head"][0])
            loss_dict["koleo_loss"] = koleo_loss
            loss_accumulator += self.dino_koleo_loss_weight * koleo_loss
        else:
            loss_dict["koleo_loss"] = torch.tensor(0.0, device=student_local["cls_after_head"].device)

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # Gram loss
        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            if self.gram_loss_schedule is not None:
                gram_loss_weight = self.gram_loss_schedule[iteration]
            else:
                gram_loss_weight = self.gram_loss_weight

            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        # ============ CLIP LOSS (TIPS) ============
        if self.vlm_enabled and text_tokens is not None:
            clip_loss = self._compute_clip_loss(
                student_global=student_global,
                text_tokens=text_tokens,
            )
            loss_dict["clip_loss"] = clip_loss
            loss_dict["logit_scale"] = self.logit_scale.exp().item()
            loss_accumulator += self.vlm_loss_weight * clip_loss

        return loss_accumulator, loss_dict

    def _compute_clip_loss(
        self,
        student_global,
        text_tokens,
    ) -> Tensor:
        """
        Compute CLIP contrastive loss between vision and text features.
        
        Vision features are obtained like DINOTxt:
        - Concatenate [CLS token, mean(patch tokens)] along channel dimension
        - Results in 2*embed_dim features, then projected to vlm_embed_dim
        
        Args:
            student_global: Dictionary containing student outputs
            text_tokens: Tokenized text captions. For custom tokenizer: [B, seq_len].
                         For HF tokenizer: dict with input_ids [B, seq_len] and attention_mask.
            
        Returns:
            CLIP contrastive loss
        """
        # Get student CLS token (pre-head, i.e., raw backbone output)
        # student_global["cls_pre_head"] shape: [n_global_crops, B, D]
        # student_global["patch_pre_head"] shape: [n_global_crops, B, P, D]
        # For TIPS with 1 global crop, use the first (only) one
        cls_token = student_global["cls_pre_head"][0]  # [B, D]
        patch_tokens = student_global["patch_pre_head"][0]  # [B, P, D]
        
        # Mean pool patch tokens (like DINOTxt)
        mean_patch_token = torch.mean(patch_tokens, dim=1)  # [B, D]
        
        # Concatenate CLS + mean(patch) along channel dimension (like DINOTxt)
        image_features = torch.cat([cls_token, mean_patch_token], dim=-1)  # [B, 2*D]
        
        # Project vision features to VLM embedding space
        image_features = self.vlm_vision_projection(image_features)  # [B, vlm_embed_dim]
        
        # Normalize image features
        image_features = F.normalize(image_features, dim=-1)
        
        # Encode text through text model
        # Handle both custom tokenizer (tensor) and HF tokenizer (dict)
        if self.use_hf_text_encoder:
            # HF text encoder expects dict with input_ids and attention_mask
            text_features = self.text_model(text_tokens=text_tokens)
          
        else:
            # Custom text encoder expects token indices tensor
            text_features = self.text_model(text_tokens)
        
        # Normalize text features
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute CLIP loss using memory-efficient implementation
        clip_loss = memory_efficient_clip_loss(
            image_features,
            text_features,
            self.logit_scale.exp(),
            group=torch.distributed.group.WORLD,
        )
        
        return clip_loss

    def get_params_groups(self):
        """
        Get parameter groups for optimizer.
        Includes both SSL and VLM components.
        """
        all_params_groups = []
        
        # SSL parameter groups (student model)
        for name, m in self.student.items():
            logger.info(f"Getting parameter groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        
        # VLM parameter groups
        if self.vlm_enabled:
            logger.info("Getting parameter groups for VLM components")
            
            # Vision projection parameters
            if isinstance(self.vlm_vision_projection, nn.Linear):
                vision_proj_params = [{
                    "params": list(self.vlm_vision_projection.parameters()),
                    "lr_multiplier": 1.0,
                    "wd_multiplier": 1.0,
                    "is_last_layer": False,
                    "name": "vlm_vision_projection",
                }]
                all_params_groups += vision_proj_params
            
            # Text model parameters
            text_params_groups = get_params_groups_with_decay_fsdp(
                model=self.text_model,
                lr_decay_rate=1.0,
                patch_embed_lr_mult=1.0,
                dino_head_wd_multiplier=1.0,
            )
            if self.cfg.optim.multi_tensor_optim:
                text_params_groups = fuse_params_groups(text_params_groups)
                for g in text_params_groups:
                    g["foreach"] = True
                    g["fused"] = True
            # Ensure text model param groups have required keys
            for g in text_params_groups:
                assert "lr_multiplier" in g, f"Missing 'lr_multiplier' in param group: {g.get('name', 'unnamed')}"
                assert "wd_multiplier" in g, f"Missing 'wd_multiplier' in param group: {g.get('name', 'unnamed')}"
                assert "is_last_layer" in g, f"Missing 'is_last_layer' in param group: {g.get('name', 'unnamed')}"
            all_params_groups += text_params_groups
            
            # Logit scale parameter
            if not self.vlm_freeze_logit_scale:
                logit_scale_params = [{
                    "params": [self.logit_scale],
                    "lr_multiplier": 1.0,
                    "wd_multiplier": 0.0,  # No weight decay for logit scale
                    "is_last_layer": False,
                    "name": "logit_scale",
                }]
                all_params_groups += logit_scale_params
        
        return all_params_groups

    def _maybe_compile_text_model(self) -> None:
        if not self.vlm_enabled:
            return
        compile_text = self.cfg.vlm.get("compile_text", self.cfg.train.compile)
        if not compile_text:
            return
        use_cuda_graphs = self.cfg.train.cudagraphs
        dynamic = self.cfg.vlm.get("text_compile_dynamic", True)
        if use_cuda_graphs and dynamic:
            logger.info("VLM text compile: cudagraphs enabled, forcing dynamic=False")
            dynamic = False

        def _compile_module(module: nn.Module) -> nn.Module:
            if use_cuda_graphs:
                module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
            else:
                module.compile(dynamic=dynamic)
            return module

        if isinstance(self.text_model, HFTextTower):
            if self.text_model.backbone is None:
                self.text_model.build_backbone_from_config()
            if isinstance(self.text_model.backbone, nn.Module):
                self.text_model.backbone = _compile_module(self.text_model.backbone)
            if isinstance(self.text_model.projection, nn.Linear):
                self.text_model.projection = _compile_module(self.text_model.projection)
        elif isinstance(self.text_model, TextTower):
            if hasattr(self.text_model.backbone, "blocks"):
                for block_id, block in enumerate(self.text_model.backbone.blocks):
                    if isinstance(block, nn.Identity):
                        continue
                    self.text_model.backbone.blocks[block_id] = _compile_module(block)
            if hasattr(self.text_model.head, "blocks"):
                for block_id, block in enumerate(self.text_model.head.blocks):
                    if isinstance(block, nn.Identity):
                        continue
                    self.text_model.head.blocks[block_id] = _compile_module(block)
            if isinstance(self.text_model.head.linear_projection, nn.Linear):
                self.text_model.head.linear_projection = _compile_module(
                    self.text_model.head.linear_projection
                )
        else:
            self.text_model = _compile_module(self.text_model)

    def prepare_for_distributed_training(self) -> None:
        """
        Prepare model for distributed training.
        Extends parent to also handle VLM components.
        """
        # Call parent's prepare_for_distributed_training for SSL components
        super().prepare_for_distributed_training()
        
        # Wrap and move VLM components to GPU
        if self.vlm_enabled:
            device = torch.device("cuda")
            logger.info("Initializing and moving VLM components to GPU...")
            
            # Determine param dtype from config
            param_dtype = torch.bfloat16 if self.cfg.compute_precision.param_dtype == "bf16" else torch.float32

            process_group = distributed.get_process_subgroup()
            self._maybe_compile_text_model()
            self._text_model_mesh = wrap_text_model_fsdp(
                self.text_model, self.cfg, process_group=process_group
            )
            self._text_model_process_group = process_group
            logger.info(f"  text_model wrapped with FSDP and moved to {device}")
            
            # Move vision projection to GPU
            # Check if on meta device - need to use to_empty() + reinitialize
            
            if isinstance(self.vlm_vision_projection, nn.Linear):
                try:
                    first_param = next(self.vlm_vision_projection.parameters())
                    if first_param.device.type == "meta":
                        logger.info("  Materializing vlm_vision_projection from meta device...")
                        # Move to empty tensors on GPU, then reinitialize
                        self.vlm_vision_projection = self.vlm_vision_projection.to_empty(device=device)
                        # Reinitialize weights with correct dtype
                        nn.init.normal_(
                            self.vlm_vision_projection.weight,
                            std=self.vlm_vision_projection.in_features ** -0.5,
                        )
                        # Convert to correct dtype
                        self.vlm_vision_projection = self.vlm_vision_projection.to(param_dtype)
                    else:
                        self.vlm_vision_projection = self.vlm_vision_projection.to(device=device, dtype=param_dtype)
                except StopIteration:
                    pass  # No parameters
            else:
                # Identity or other - just move
                self.vlm_vision_projection = self.vlm_vision_projection.to(device)
            logger.info(f"  vlm_vision_projection moved to {device}, dtype={param_dtype}")

            compile_text = self.cfg.vlm.get("compile_text", self.cfg.train.compile)
            if compile_text and isinstance(self.vlm_vision_projection, nn.Linear):
                use_cuda_graphs = self.cfg.train.cudagraphs
                dynamic = self.cfg.vlm.get("text_compile_dynamic", True)
                if use_cuda_graphs and dynamic:
                    dynamic = False
                if use_cuda_graphs:
                    self.vlm_vision_projection.compile(
                        fullgraph=True, dynamic=False, options={"triton.cudagraphs": True}
                    )
                else:
                    self.vlm_vision_projection.compile(dynamic=dynamic)
            
            # Materialize and move logit_scale to GPU
            # If on meta device, need to recreate with actual data
            if self.logit_scale.device.type == "meta":
                logger.info("  Materializing logit_scale from meta device...")
                self.logit_scale = nn.Parameter(
                    torch.tensor([self.vlm_init_logit_scale], device=device)
                )
                if self.vlm_freeze_logit_scale:
                    self.logit_scale.requires_grad = False
            else:
                self.logit_scale.data = self.logit_scale.data.to(device)
            logger.info(f"  logit_scale on {self.logit_scale.device}")

    def train(self):
        """Set model to training mode."""
        super().train()
        if self.vlm_enabled:
            self.text_model.train()
            if not self.text_model.freeze_backbone:
                self.text_model.backbone.train()

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        if self.vlm_enabled:
            self.text_model.eval()
