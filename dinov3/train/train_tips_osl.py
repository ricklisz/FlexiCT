# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# TIPS + OSL Training Script: DINOv3 + CLIP loss + Opposite Sentence Loss
# This script trains the TIPSMetaArch model with image-caption pairs and OSL.

import argparse
import gc
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path
from collections.abc import Sequence
from packaging import version
import torch
import torch.nn.functional as F
import torch.distributed
from omegaconf import OmegaConf
from torch.distributed._tensor import DTensor
from torch import Tensor

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from dinov3.configs import setup_config, setup_job
from dinov3.data import (
    SamplerType,
    make_data_loader,
    CombinedDataLoader,
)
from dinov3.data.masking import MaskingGenerator3D
from dinov3.logging import MetricLogger, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.tips_meta_arch import TIPSMetaArch, TIPS_N_GLOBAL_CROPS

assert version.parse(torch.__version__.split("+", 1)[0]) >= version.parse("2.1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger("dinov3")


# ============================================================================
# TIPSMetaArchOSL: Extends TIPSMetaArch with Opposite Sentence Loss (OSL)
# Optimized: Single vision forward, batched text encoding
# ============================================================================

class TIPSMetaArchOSL(TIPSMetaArch):
    """
    TIPS + OSL: Extends TIPSMetaArch with Opposite Sentence Loss.
    
    Optimizations over naive implementation:
    1. Vision features computed once, reused for CLIP and OSL
    2. All text (CLIP + OSL pos + OSL neg) encoded in single batched call
    3. All tensors moved to GPU once at start
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # OSL configuration
        self.osl_enabled = cfg.vlm.get("osl_enabled", True) if hasattr(cfg, 'vlm') else False
        self.osl_loss_weight = cfg.vlm.get("osl_loss_weight", 0.5) if hasattr(cfg, 'vlm') else 0.5
        self.osl_K = cfg.vlm.get("osl_K", 8) if hasattr(cfg, 'vlm') else 8
        
        if self.osl_enabled:
            logger.info("OPTIONS -- OSL (Opposite Sentence Loss)")
            logger.info(f"OPTIONS -- OSL -- enabled: {self.osl_enabled}")
            logger.info(f"OPTIONS -- OSL -- loss_weight: {self.osl_loss_weight}")
            logger.info(f"OPTIONS -- OSL -- K (pairs per sample): {self.osl_K}")
    
    def _get_vision_features(self, student_global) -> Tensor:
        """
        Compute normalized vision features from student global output.
        Concatenates CLS token with mean of patch tokens, then projects.
        
        Returns:
            image_features: (B, vlm_embed_dim) normalized
        """
        cls_token = student_global["cls_pre_head"][0]  # [B, D]
        patch_tokens = student_global["patch_pre_head"][0]  # [B, P, D]
        mean_patch_token = patch_tokens.mean(dim=1)  # [B, D]
        image_features = torch.cat([cls_token, mean_patch_token], dim=-1)  # [B, 2*D]
        image_features = self.vlm_vision_projection(image_features)  # [B, vlm_embed_dim]
        return F.normalize(image_features, dim=-1)
    
    def _encode_text(self, tokens) -> Tensor:
        """Encode tokens and return normalized features."""
        if self.use_hf_text_encoder:
            features = self.text_model(text_tokens=tokens)
        else:
            features = self.text_model(tokens)
        return F.normalize(features, dim=-1)
    
    def _encode_osl_text_batched(self, pos_tokens, neg_tokens) -> tuple[Tensor, Tensor]:
        """
        Encode OSL pos and neg tokens in a single forward pass.
        Both must have the same max_length so they can be concatenated.
        
        Returns:
            (pos_features, neg_features): Both normalized, shape (N, D)
        """
        pos_size = pos_tokens.input_ids.shape[0]
        
        if self.use_hf_text_encoder:
            from transformers import BatchEncoding
            combined_tokens = BatchEncoding({
                'input_ids': torch.cat([pos_tokens.input_ids, neg_tokens.input_ids], dim=0),
                'attention_mask': torch.cat([pos_tokens.attention_mask, neg_tokens.attention_mask], dim=0),
            })
            combined_features = self.text_model(text_tokens=combined_tokens)
        else:
            combined_tokens = torch.cat([pos_tokens, neg_tokens], dim=0)
            combined_features = self.text_model(combined_tokens)
        
        combined_features = F.normalize(combined_features, dim=-1)
        
        # Split back
        pos_features = combined_features[:pos_size]
        neg_features = combined_features[pos_size:]
        return pos_features, neg_features
    
    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        """
        Optimized forward and backward pass for TIPS + OSL training.
        
        Key optimizations:
        - Vision features computed once, reused for CLIP and OSL
        - All text encoded in single batched call
        - All data moved to GPU at start
        """
        del ignored_kwargs
        metrics_dict = {}
        device = torch.device("cuda")

        n_global_crops = TIPS_N_GLOBAL_CROPS
        n_local_crops = self.n_local_crops
        B = data["collated_local_crops"].shape[0] // n_local_crops
        
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]
        metrics_dict["n_global_crops"] = n_global_crops

        # Move all image data to GPU at once
        global_crops = data["collated_global_crops"].to(device, non_blocking=True)
        local_crops = data["collated_local_crops"].to(device, non_blocking=True)
        masks = data["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = data["mask_indices_list"].to(device, non_blocking=True)
        masks_weight = data["masks_weight"].to(device, non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].to(device, non_blocking=True)

        if global_crops.dim() == 4:
            student_global_crops_size = global_crops.shape[-1]
        elif global_crops.dim() == 5:
            student_global_crops_size = global_crops.shape[-3:]
        else:
            raise ValueError(f"Unexpected global_crops dim={global_crops.dim()}")

        gram_teacher_crops = None
        if self.has_gram_teacher and "collated_gram_teacher_crops" in data:
            gram_teacher_crops = data["collated_gram_teacher_crops"].to(device, non_blocking=True)

        # ============ VISION FORWARD PASSES ============
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
        gram_global = {}
        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=student_global_crops_size,
            )

        # ============ VLM: COMPUTE VISION FEATURES ONCE ============
        image_features = None
        if self.vlm_enabled:
            image_features = self._get_vision_features(student_global)

        # ============ VLM: BATCHED TEXT ENCODING ============
        clip_text_features = None
        osl_pos_features = None
        osl_neg_features = None
        has_osl = self.osl_enabled and self.vlm_enabled and "osl_pos_tokens" in data
        
        if self.vlm_enabled:
            # Move text tokens to GPU and encode CLIP text
            text_tokens = data["text_tokens"].to(device)
            clip_text_features = self._encode_text(text_tokens)
            
            if has_osl:
                # Move OSL tokens to GPU and encode in single batched call
                # (OSL pos and neg have same max_length, so they can be batched)
                osl_pos_tokens = data["osl_pos_tokens"].to(device)
                osl_neg_tokens = data["osl_neg_tokens"].to(device)
                osl_pos_features, osl_neg_features = self._encode_osl_text_batched(
                    osl_pos_tokens, osl_neg_tokens
                )

        # ============ COMPUTE LOSSES ============
        loss_dict = {}
        loss_accumulator = 0.0

        # DINO local loss
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0
        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * local_weight * dino_local_crops_loss

        # No DINO global loss for TIPS (single global crop)
        loss_dict["dino_global_crops_loss"] = torch.tensor(0.0, device=device)

        # KoLeo loss
        if self.dino_koleo_loss_weight > 0:
            koleo_loss = self.koleo_loss(student_global["cls_pre_head"][0])
            loss_dict["koleo_loss"] = koleo_loss
            loss_accumulator += self.dino_koleo_loss_weight * koleo_loss
        else:
            loss_dict["koleo_loss"] = torch.tensor(0.0, device=device)

        # iBOT loss
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
            gram_loss_weight = self.gram_loss_schedule[iteration] if self.gram_loss_schedule is not None else self.gram_loss_weight
            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_dict["gram_loss"] = gram_loss
            loss_accumulator += gram_loss * gram_loss_weight

        # ============ CLIP LOSS ============
        if self.vlm_enabled and image_features is not None and clip_text_features is not None:
            from dinov3.eval.text.clip_loss import memory_efficient_clip_loss
            clip_loss = memory_efficient_clip_loss(
                image_features,
                clip_text_features,
                self.logit_scale.exp(),
                group=torch.distributed.group.WORLD,
            )
            loss_dict["clip_loss"] = clip_loss
            loss_dict["logit_scale"] = self.logit_scale.exp().item()
            loss_accumulator += self.vlm_loss_weight * clip_loss

        # ============ OSL LOSS ============
        if has_osl and image_features is not None:
            osl_labels = data["osl_labels"].to(device)
            osl_mask = data["osl_mask"].to(device)
            
            osl_loss, osl_metrics = self._compute_osl_loss_from_features(
                image_features=image_features,
                pos_text_features=osl_pos_features,
                neg_text_features=osl_neg_features,
                osl_labels=osl_labels,
                osl_mask=osl_mask,
            )
            loss_dict["osl_loss"] = osl_loss
            loss_dict.update(osl_metrics)
            loss_accumulator += self.osl_loss_weight * osl_loss

        self.backprop_loss(loss_accumulator)

        return loss_accumulator, metrics_dict | loss_dict
    
    def _compute_osl_loss_from_features(
        self,
        image_features: Tensor,
        pos_text_features: Tensor,
        neg_text_features: Tensor,
        osl_labels: Tensor,
        osl_mask: Tensor,
    ) -> tuple[Tensor, dict]:
        """
        Compute OSL loss from pre-computed, normalized features.
        
        This is the fast path - features are already computed and on GPU.
        """
        device = image_features.device
        B = image_features.shape[0]
        K = self.osl_K
        D = image_features.shape[1]
        
        # Reshape text features: (B*K, D) -> (B, K, D)
        pos_text_features = pos_text_features.view(B, K, D)
        neg_text_features = neg_text_features.view(B, K, D)
        
        # Compute cosine similarities via batched matmul
        # image_features: (B, D) -> (B, 1, D)
        v = image_features.unsqueeze(1)
        
        # Efficient: use einsum for batched dot product
        sim_pos = torch.einsum('bkd,bkd->bk', v.expand(-1, K, -1), pos_text_features)
        sim_neg = torch.einsum('bkd,bkd->bk', v.expand(-1, K, -1), neg_text_features)
        
        # Scale by temperature
        temperature = self.logit_scale.exp()
        
        # Stack into logits: (B*K, 2)
        logits = torch.stack([sim_pos, sim_neg], dim=-1).view(B * K, 2) * temperature
        
        # Flatten labels and mask
        osl_labels_flat = osl_labels.view(B * K)
        osl_mask_flat = osl_mask.view(B * K)
        
        # Targets: y=1 -> class 0 (pos), y=0 -> class 1 (neg)
        targets = 1 - osl_labels_flat  # Equivalent to: where(labels==1, 0, 1)
        
        # Masked loss
        valid_logits = logits[osl_mask_flat]
        valid_targets = targets[osl_mask_flat]
        
        if valid_logits.numel() == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze(), {}
        
        osl_loss = F.cross_entropy(valid_logits, valid_targets)
        
        # Metrics (in no_grad for speed)
        metrics = {}
        with torch.no_grad():
            preds = valid_logits.argmax(dim=-1)
            metrics["osl_accuracy"] = (preds == valid_targets).float().mean()
            metrics["osl_n_valid_pairs"] = float(osl_mask_flat.sum())
            
            # Per-class accuracy
            valid_labels = osl_labels_flat[osl_mask_flat]
            true_mask = valid_labels == 1
            false_mask = valid_labels == 0
            if true_mask.any():
                metrics["osl_true_accuracy"] = (preds[true_mask] == valid_targets[true_mask]).float().mean()
            if false_mask.any():
                metrics["osl_false_accuracy"] = (preds[false_mask] == valid_targets[false_mask]).float().mean()
        
        return osl_loss, metrics


def unwrap_model(model):
    return getattr(model, "module", model)

def set_model_patch_size(model, ps: int):
    # student dict of subnets (common in DINOv3)
    for _, subm in getattr(model, "student", {}).items():
        for m in subm.modules():
            if hasattr(m, "set_patch_size"):
                m.set_patch_size(ps)
                model.student.backbone.patch_size = ps
    # teacher (if present)
    if hasattr(model, "teacher"):
        for _, subm in getattr(model, "teacher", {}).items():
            for m in subm.modules():
                if hasattr(m, "set_patch_size"):
                    m.set_patch_size(ps)
                    model.teacher.backbone.patch_size = ps
    # EMA teacher
    if hasattr(model, "model_ema"):
        for m in model.model_ema.modules():
            if hasattr(m, "set_patch_size"):
                m.set_patch_size(ps)
                model.model_ema.backbone.patch_size = ps
    # gram teacher
    if hasattr(model, "gram_teacher") and model.gram_teacher is not None:
        for m in model.gram_teacher.modules():
            if hasattr(m, "set_patch_size"):
                m.set_patch_size(ps)
        model.gram_teacher.backbone.patch_size = ps
        
def collate_data_and_cast_tips_osl(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
    tokenizer=None,
    max_length=8192,
    osl_max_length=256,
    osl_K=8,
):
    """
    Collate function for TIPS + OSL training.

    Key differences from standard collate:
    - Expects 1 global crop per sample
    - Handles OSL pairs (pos_texts, neg_texts, labels, mask)
    """
    # TIPS: hardcoded 1 global crop
    n_global_crops = 1
    n_local_crops = len(samples_list[0][0]["local_crops"])

    # Verify we have exactly 1 global crop
    assert len(samples_list[0][0]["global_crops"]) == 1, (
        f"TIPS expects 1 global crop, got {len(samples_list[0][0]['global_crops'])}"
    )

    # Collate global crops [1 * B, ...]
    collated_global_crops = torch.stack(
        [s[0]["global_crops"][0] for s in samples_list]
    )

    # Collate local crops [n_local_crops * B, ...]
    collated_local_crops = torch.stack(
        [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )

    # Collate gram teacher crops if present
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][0] for s in samples_list]
        )
    else:
        collated_gram_teacher_crops = None

    # Determine batch size for masking
    if local_batch_size is not None:
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)

    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []

    import random

    for i in range(n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:
            if mask.ndim == 2:
                shift_x, shift_y = (
                    random.randint(0, mask.shape[0] - 1),
                    random.randint(0, mask.shape[1] - 1),
                )
                mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
            elif mask.ndim == 3:
                shift_x, shift_y, shift_z = (
                    random.randint(0, mask.shape[0] - 1),
                    random.randint(0, mask.shape[1] - 1),
                    random.randint(0, mask.shape[2] - 1),
                )
                mask = torch.roll(mask, (shift_x, shift_y, shift_z), (0, 1, 2))
        masks_list.append(mask)
        upperbound += int(N * prob_max)

    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    # Tokenize captions for CLIP loss
    captions = [s[0]["caption"] for s in samples_list]
    text_tokens = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "text_tokens": text_tokens,
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)

    # ============ OSL DATA ============
    # Check if OSL data is present in samples
    if "osl_pairs" in samples_list[0][0] and "osl_labels" in samples_list[0][0]:
        pos_texts = []
        neg_texts = []
        osl_labels = []
        
        for s in samples_list:
            sample = s[0]
            for (s_pos, s_neg) in sample["osl_pairs"]:
                pos_texts.append(s_pos.lower() if s_pos else "")
                neg_texts.append(s_neg.lower() if s_neg else "")
            osl_labels.extend(sample["osl_labels"])
        
        # Tokenize OSL texts (shorter max_length since these are single sentences)
        osl_pos_tokens = tokenizer(
            pos_texts,
            padding="max_length",
            truncation=True,
            max_length=osl_max_length,
            return_tensors="pt",
        )
        osl_neg_tokens = tokenizer(
            neg_texts,
            padding="max_length",
            truncation=True,
            max_length=osl_max_length,
            return_tensors="pt",
        )
        
        osl_labels_tensor = torch.tensor(osl_labels, dtype=torch.long)
        osl_mask_tensor = osl_labels_tensor != -1
        
        out["osl_pos_tokens"] = osl_pos_tokens
        out["osl_neg_tokens"] = osl_neg_tokens
        out["osl_labels"] = osl_labels_tensor
        out["osl_mask"] = osl_mask_tensor

    return out

def check_params_finite(module, name="model"):
    """Check if all parameters in a module are finite."""
    for pname, p in module.named_parameters():
        if p.requires_grad and not getattr(p, "is_meta", False):
            if not torch.isfinite(p).all():
                raise RuntimeError(f"[NaN DEBUG] {name}.{pname} has non-finite values!")
    for bname, b in module.named_buffers():
        if not getattr(b, "is_meta", False):
            if b.numel() > 0 and not torch.isfinite(b).all():
                raise RuntimeError(f"[NaN DEBUG] {name}.{bname} buffer has non-finite values!")


def get_args_parser(add_help: bool = True):
    """Create argument parser for TIPS + OSL training."""
    parser = argparse.ArgumentParser("TIPS + OSL training (DINOv3 + CLIP + Opposite Sentence Loss)", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="./local_tips",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, fused=True, 
                             betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    """Build learning rate and other schedulers."""
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = 0
    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_schedulers_v2(cfg):
    """Build learning rate and other schedulers (v2 format)."""
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        lr_end *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, "
            f"LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        lr_end *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, "
            f"LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs
            if "cosine_epochs" in cfg.schedules.momentum
            else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    logger.info("Schedulers v2 ready.")
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    """Apply learning rate and weight decay schedules to optimizer."""
    for param_group in optimizer.param_groups:
        is_last_layer = param_group.get("is_last_layer", False)
        lr_multiplier = param_group.get("lr_multiplier", 1.0)
        wd_multiplier = param_group.get("wd_multiplier", 1.0)
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def build_data_loader_with_captions_osl(
    cfg,
    model,
    tokenizer,
    start_iter: int = 0,
    seed: int = 0,
    override_patch_size=16,
):
    """
    Build data loader for TIPS + OSL training with image-caption pairs.
    
    Uses TIPS-specific augmentation (1 global crop) and OSL-enabled collate function.
    
    Args:
        cfg: Configuration object
        model: TIPSMetaArchOSL model
        tokenizer: Text tokenizer
        start_iter: Starting iteration for sampler
        seed: Random seed
        override_patch_size: Patch size override
        
    Returns:
        DataLoader for TIPS + OSL training
    """
    # Build masking generator for iBOT 3D
    img_size = cfg.crops.global_crops_size
    patch_size = override_patch_size if override_patch_size is not None else cfg.student.patch_size 
    n_tokens = img_size[0] * img_size[1] * img_size[2] // (patch_size**3)
    rcc_masking = OmegaConf.select(cfg, "rcc.masking")
    use_rcc = (rcc_masking == "rcc")
    if use_rcc:
        from dinov3.data.masking import RCCMaskingGenerator3D
        mask_generator = RCCMaskingGenerator3D(
            input_size=(img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size),
            num_masking_patches=0.75 * n_tokens,
            grid_num=4,
            box_area_scale=(0.02, 0.05),
            box_aspect=(0.5, 2.0),    
        )
        logger.info('RCC masking selected!')
    else:
        mask_generator = MaskingGenerator3D(
            input_size=(img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size),
            max_num_voxels=0.5 * n_tokens,
        )
        logger.info('Default block-wise masking selected!')
    
    # OSL configuration
    osl_K = cfg.vlm.get("osl_K", 8) if hasattr(cfg, 'vlm') else 8
    osl_max_length = cfg.vlm.get("osl_max_length", 256) if hasattr(cfg, 'vlm') else 256
    
    # Build TIPS + OSL collate function
    collate_fn = partial(
        collate_data_and_cast_tips_osl,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        tokenizer=tokenizer,
        random_circular_shift=cfg.ibot.get("mask_random_circular_shift", False),
        max_length=cfg.vlm.get("hf_max_length", 8192),
        osl_max_length=osl_max_length,
        osl_K=osl_K,
    )
    
    def collate_with_ps(batch):
        out = collate_fn(batch)
        out["patch_size"] = patch_size
        return out

    # Use CT3D_CLIP_OSL dataset with OSL support
    from dinov3.data.datasets.ct_dataset import CT3D_CLIP_OSL
    norm_setting = "ct"
    csv_paths_env = os.environ.get("FLEXICT_REPORT_CSVS", "")
    csv_paths = [path.strip() for path in csv_paths_env.split(",") if path.strip()]
    if not csv_paths:
        raise ValueError("Set FLEXICT_REPORT_CSVS to one or more comma-separated report CSV paths.")
    
    # OSL seed for reproducible sampling
    osl_seed = cfg.vlm.get("osl_seed", None) if hasattr(cfg, 'vlm') else None
    
    dataset = CT3D_CLIP_OSL(
        csv_paths=csv_paths,
        transforms=model.build_3D_data_augmentation(cfg),
        spacing=(2.0, 2.0, 2.0),
        norm=norm_setting,
        reshuffle_probability=0.5,
        K=osl_K,
        osl_seed=osl_seed,
    )
    
    logger.info(f"OSL Dataset created with K={osl_K}, {len(dataset)} samples")
    logger.info(f"OSL S_db_plus sections: {list(dataset.S_db_plus.keys())}")
    
    # Determine sampler type
    sampler_type = SamplerType.SHARDED_INFINITE
    
    # Build data loader
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        collate_fn=collate_with_ps,
        persistent_workers=False,
        prefetch_factor=2,
    )
    
    return data_loader

def build_single_res_loader_from_cfg(cfg, model, tokenizer, start_iter):
    loaders = [
        build_data_loader_with_captions_osl(cfg, model, tokenizer, start_iter, override_patch_size=16),
        build_data_loader_with_captions_osl(cfg, model, tokenizer, start_iter, override_patch_size=8),
    ]
    # ratios [1,1] ⇒ alternate
    return CombinedDataLoader(
        loaders_with_ratios=zip(loaders, [0.5, 0.5]),
        batch_size=cfg.train.batch_size_per_gpu,
        combining_mode=0,   # standard interleave
        seed=cfg.train.seed + 42,
        name="PatchSizeAltDL",
    )

def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    tokenizer,
    start_iter,
    seed=65537,
    patch_sizes=(16, 8),
):
    global_crops_sizes = (
        [cfg.crops.global_crops_size] if isinstance(cfg.crops.global_crops_size, int) else cfg.crops.global_crops_size
    )
    local_crops_sizes = (
        [cfg.crops.local_crops_size] if isinstance(cfg.crops.local_crops_size, int) else cfg.crops.local_crops_size
    )
    gram_teacher_crops_sizes = (
        [cfg.crops.gram_teacher_crops_size]
        if cfg.crops.gram_teacher_crops_size is None or isinstance(cfg.crops.gram_teacher_crops_size, int)
        else cfg.crops.gram_teacher_crops_size
    )
    loader_ratios = (
        [cfg.crops.global_local_crop_pairs_ratios]
        if type(cfg.crops.global_local_crop_pairs_ratios) in [int, float]
        else cfg.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(gram_teacher_crops_sizes) == len(loader_ratios)

    loaders = []
    final_loader_ratios = []
    for increment, (global_crops_size_i, local_crops_size_i, gram_teacher_crops_size_i) in enumerate(
        zip(global_crops_sizes, local_crops_sizes, gram_teacher_crops_sizes)
    ):
        for j, ps in enumerate(patch_sizes):
            cfg_i = OmegaConf.create(cfg)
            cfg_i.crops.global_crops_size = global_crops_size_i
            cfg_i.crops.local_crops_size = local_crops_size_i
            cfg_i.crops.gram_teacher_crops_size = gram_teacher_crops_size_i
            cfg_i.train.seed = cfg.train.seed + increment * len(patch_sizes) + j + 1
            loaders.append(
                build_data_loader_with_captions_osl(cfg=cfg_i, model=model, tokenizer=tokenizer, start_iter=start_iter, override_patch_size=ps)
            )
            final_loader_ratios.append(loader_ratios[increment] / len(patch_sizes))
    assert len(loaders) == len(final_loader_ratios)
    data_loader = CombinedDataLoader(
        loaders_with_ratios=zip(loaders, final_loader_ratios),
        batch_size=cfg.train.batch_size_per_gpu,
        combining_mode=0,
        seed=seed,
        name="PatchSizeAltDL",
    )
    return data_loader

def do_train(cfg, model, tokenizer, resume=False):
    """
    Main training loop for TIPS + OSL.
    
    Args:
        cfg: Configuration object
        model: TIPSMetaArchOSL model (with OSL support)
        tokenizer: Text tokenizer
        resume: Whether to resume from checkpoint
    """
    torch.autograd.set_detect_anomaly(True)
    process_subgroup = distributed.get_process_subgroup()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    
    # Build optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    # Resume from checkpoint if available
    start_iter = 0
    last_checkpoint_dir = None
    if resume:
        last_checkpoint_dir = find_latest_checkpoint(ckpt_dir)

    # Initialize weights (skip HF load if resuming from DCP)
    model.init_weights(resume=last_checkpoint_dir is not None)

    if last_checkpoint_dir is not None:
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        start_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                strict_loading=False,
                process_group=process_subgroup,
            )
            + 1
        )
        # Reset to 0 if using gram loss with gram teacher
        if cfg.gram.use_loss and model.has_gram_teacher:
            start_iter = 0

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    if cfg.multidistillation.enabled:
        global_batch_size = cfg.multidistillation.global_batch_size
    else:
        global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    gcs = OmegaConf.to_container(cfg.crops.global_crops_size, resolve=True)
    if isinstance(gcs, list) and isinstance(gcs[0], int):
        # Build data loader
        data_loader = build_single_res_loader_from_cfg(
            cfg=cfg,
            model=model,tokenizer = tokenizer,
            start_iter=start_iter,
        )
    elif isinstance(gcs, list) and isinstance(gcs[0], Sequence) and len(gcs[0]) == 3:
        data_loader = build_multi_resolution_data_loader_from_cfg(
            cfg=cfg,
            model=model,
            tokenizer = tokenizer,
            start_iter=start_iter,
            patch_sizes=(16, 8),
        )

    # Metric logging
    logger.info("Starting TIPS training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    
    gc.collect()

    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
    
    consecutive_nan_count = 0
    logger.info(f"Rank {os.environ.get('RANK')} affinity: {os.sched_getaffinity(0)}")
    
    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header="TIPS Training",
        n_iterations=max_iter,
        start_iteration=start_iter,
    ):
        it = iteration
        data["global_batch_size"] = global_batch_size
        
        if iteration > max_iter:
            return
        
        # Garbage collection
        if (iteration + 1) % 150 == 0:
            logger.info("Garbage collection")
            gc.collect()

        # Load EMA teacher into Gram teacher at specified iteration
        if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
            model.gram_load_ema_teacher()

        # Apply learning rate and weight decay schedules
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
        
        # Forward and backward pass
        optimizer.zero_grad(set_to_none=True)
        ps = int(data.get("patch_size", cfg.student.patch_size))  # 16 or 8 from the loader
        set_model_patch_size(model, ps) 
        total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)

        # Gradient clipping
        if cfg.optim.clip_grad:
            for k, v in student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    v.parameters(),
                    max_norm=cfg.optim.clip_grad,
                )
                metrics_dict[f"{k}_grad_norm"] = (
                    grad_norm.full_tensor().item()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm.item()
                )
            
            # Also clip VLM components if enabled
            if model.vlm_enabled:
                vlm_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.text_model.parameters(),
                    max_norm=cfg.optim.clip_grad,
                )
                metrics_dict["text_model_grad_norm"] = (
                    vlm_grad_norm.full_tensor().item()
                    if isinstance(vlm_grad_norm, DTensor)
                    else vlm_grad_norm.item()
                )

        # Check for NaNs across all ranks
        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()
        
        # Reduce metrics for logging
        metrics_values = torch.stack(
            [torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach() for v in metrics_dict.values()]
        )
        torch.distributed.all_reduce(
            metrics_values,
            op=torch.distributed.ReduceOp.AVG,
            group=distributed.get_process_subgroup(),
        )
        metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
        
        # Handle NaN losses
        if total_loss_all_ranks.isnan().any():
            consecutive_nan_count += 1
            which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
            logger.warning("NaN loss detected on ranks: %s", which_ranks)
            logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
            metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
            if consecutive_nan_count > 2:
                msg = "Too many consecutive NaNs detected in loss, aborting..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0
        
        # Step optimizer
        optimizer.step()
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        # Update EMA teacher
        model.update_ema(mom)

        # Update gram teacher if applicable
        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(total_loss=total_loss, **metrics_dict)

        # Save checkpoint
        if (iteration + 1) % cfg.checkpointing.period == 0:
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                process_group=process_subgroup,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))
        
        iteration += 1

    logger.info("TIPS + OSL Training completed!")


def main(argv=None):
    """Main entry point for TIPS + OSL training."""
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]

    setup_job(output_dir=args.output_dir, seed=args.seed)
    cfg = setup_config(args, strict_cfg=False)
    logger.info(cfg)
    setup_logging(
        output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
        name="nan_logger",
    )

    # Build model with OSL support
    logger.info("Building TIPSMetaArchOSL model (TIPS + Opposite Sentence Loss)...")

    with torch.device("meta"):
        model = TIPSMetaArchOSL(cfg)
    # Setup distributed training
    model.prepare_for_distributed_training()
    # Initialize tokenizer
    tokenizer = None
    if model.vlm_enabled:
        logger.info("Initializing tokenizer...")
        tokenizer = model.get_tokenizer()
    
    # Log OSL configuration
    if model.osl_enabled:
        logger.info(f"OSL enabled with weight={model.osl_loss_weight}, K={model.osl_K}")
    
    # Run training
    do_train(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
