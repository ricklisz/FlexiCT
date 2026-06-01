# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import gc
import logging
from functools import partial
from typing import Tuple, Dict
import torch
from omegaconf import OmegaConf
import dinov3.distributed as distributed
from dinov3.checkpointer import init_fsdp_model_from_checkpoint, init_fsdp_model_from_dinov3
from dinov3.configs import get_default_config
from dinov3.data import SimpleDataAugmentationDINO3D, SimpleDataAugmentationDINO, NoDataAugmentationDINO, RandomCropDINO3D
from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, GramLoss, KoLeoLoss, KoLeoLossDistributed, iBOTPatchLoss
from dinov3.models import build_model_from_cfg
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.train.param_groups import fuse_params_groups, get_params_groups_with_decay_fsdp
from dinov3.utils import count_parameters
from pathlib import Path
import math
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.transforms as T
import numpy as np

from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

logger = logging.getLogger("dinov3")

def _summ(t):
    return (f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4g} max={t.max().item():.4g} "
            f"mean={t.mean().item():.4g} std={t.std(unbiased=False).item():.4g}")

def _check_finite(name, t):
    bad = not torch.isfinite(t).all()
    if bad:
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        print(f"[NaN DEBUG] {name}: non-finite (nan={n_nan}, inf={n_inf}) {_summ(t)}", flush=True)
    return not bad


class SSLMetaArch(nn.Module):
    """
    Modified version of SSLMetaArchCompilable including gram loss:
    - Gram loss is used only if gram.use_loss is set to true
    """

    def __init__(self, cfg):
        super().__init__()

        # assert cfg.multidistillation.enabled is False
        assert cfg.crops.local_crops_number > 0
        assert cfg.ibot.separate_head is True
        # assert cfg.train.centering == "sinkhorn_knopp"

        # For some reason FULL_SHARD doesn't work
        assert cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP"

        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()
        gram_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        torch.cuda.empty_cache()
        gc.collect()
        gram_backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(student_backbone)}")
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        gram_model_dict["backbone"] = gram_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim  # D
        self.dino_out_dim = cfg.dino.head_n_prototypes  # K

        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {cfg.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {cfg.dino.head_norm_last_layer}")
        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        student_model_dict["dino_head"] = dino_head_class()
        teacher_model_dict["dino_head"] = dino_head_class()
        self.dino_loss = DINOLoss(self.dino_out_dim)

        logger.info("OPTIONS -- KOLEO")
        logger.info(f"OPTIONS -- KOLEO -- loss_weight: {cfg.dino.koleo_loss_weight}")
        logger.info(f"OPTIONS -- KOLEO -- distributed: {cfg.dino.koleo_loss_distributed}")
        if cfg.dino.koleo_loss_distributed:
            logger.info(f"OPTIONS -- KOLEO -- topk: {cfg.dino.koleo_topk}")
            logger.info(
                f"OPTIONS -- KOLEO -- distributed_loss_group_size: {cfg.dino.koleo_distributed_loss_group_size}"
            )
            assert cfg.dino.koleo_distributed_replicas == 0, (
                "Option `dino.koleo_distributed_replicas` is no longer supported"
            )
            self.koleo_loss = KoLeoLossDistributed(
                topk=cfg.dino.koleo_topk,
                loss_group_size=cfg.dino.koleo_distributed_loss_group_size,
            )
        else:
            assert cfg.dino.koleo_topk == 1, "Non-distributed KoLeo loss only supports `dino.koleo_topk=1`"
            self.koleo_loss = KoLeoLoss()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        assert 0 <= cfg.ibot.mask_ratio_min_max[0] < cfg.ibot.mask_ratio_min_max[1] <= 1, (
            "provide a valid cfg.ibot.mask_ratio_min_max"
        )
        assert 0 <= cfg.ibot.mask_sample_probability <= 1, "provide a positive mask probability for ibot"
        logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
        logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
        logger.info(f"OPTIONS -- IBOT -- head_norm_last_layer: {cfg.ibot.head_norm_last_layer}")
        ibot_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_model_dict["ibot_head"] = ibot_head_class()
        teacher_model_dict["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)

        # Build student and teacher models
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.model_ema = self.teacher  # this may be overwritten for distillation
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        if cfg.distillation.enabled:
            self._setup_distillation()
        # No grad is needed for these two
        self.teacher.requires_grad_(False)
        self.model_ema.requires_grad_(False)
        self.ema_params_lists = None

        # getting config params fixed:
        self.n_local_crops = self.cfg.crops.local_crops_number
        self.is_distillation_enabled = self.cfg.distillation.enabled
        self.dino_global_ignore_diagonal = self.cfg.dino.global_ignore_diagonal
        self.dino_loss_weight = self.cfg.dino.loss_weight
        self.dino_koleo_loss_weight = self.cfg.dino.koleo_loss_weight
        self.ibot_loss_weight = self.cfg.ibot.loss_weight

        # Local loss reweighting
        if self.cfg.dino.reweight_dino_local_loss:
            iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * cfg.optim.epochs
            schedule_cfg = cfg.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_cfg.start,
                peak=schedule_cfg.peak,
                end=schedule_cfg.end,
                warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                ),
            )

        # Gram
        self.gram_use_loss = self.cfg.gram.use_loss
        self.gram_ema_teacher = False
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            # Gram regularization
            self.gram_loss = GramLoss(
                apply_norm=self.cfg.gram.normalized,
                remove_only_teacher_neg=self.cfg.gram.remove_only_teacher_neg,
                remove_neg=self.cfg.gram.remove_neg,
            )
            # Construct gram teacher
            self.has_gram_teacher = True if not cfg.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = nn.ModuleDict(gram_model_dict)
                self.gram_teacher.requires_grad_(False)
                logger.info(f"Gram teacher parameter at init: {next(self.gram_teacher.named_parameters())}")
            else:
                self.gram_teacher = None

            self.gram_loss_weight = self.cfg.gram.loss_weight
            if self.cfg.gram.get("loss_weight_schedule"):
                iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
                total_iterations = iter_per_epoch * cfg.optim.epochs
                schedule_cfg = self.cfg.gram.loss_weight_schedule
                self.gram_loss_schedule = linear_warmup_cosine_decay(
                    start=schedule_cfg.start,
                    peak=schedule_cfg.peak,
                    end=schedule_cfg.end,
                    warmup_iterations=iter_per_epoch * schedule_cfg.warmup_epochs,
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        iter_per_epoch * schedule_cfg.cosine_epochs if "cosine_epochs" in schedule_cfg else None
                    ),
                )
                logger.info(f"Applying gram loss weight schedule instead of `cfg.gram.loss_weight`: {schedule_cfg}")
            else:
                self.gram_loss_schedule = None
            self.gram_ema_teacher = self.cfg.gram.ema_teacher  # If true use the EMA_teacher as gram_teacher
            self.gram_ckpt = self.cfg.gram.ckpt  # Checkpoint to the first gram teacher model
            self.gram_img_level = self.cfg.gram.img_level  # Apply the loss on the image, if false on the batch
            self.gram_tokens_used = self.cfg.gram.tokens_used  # Any value in ["all", "masked", "unmasked"]
            # Update the teacher frequently
            self.gram_rep_update = self.cfg.gram.rep_update  # bool, if yes the gram teacher will be updated at the freq
            self.gram_update_frequency = self.cfg.gram.update_frequency  # defined by this var update_frequency
            self.gram_it_first_update = self.cfg.gram.it_first_update  # after iteration it_first_update is passed.
            self.gram_it_load_ema_teacher = (
                self.cfg.gram.it_load_ema_teacher
            )  # after iteration it_load_ema the ema teacher is loaded into the gram teacher
            self.gram_compute_stats = self.cfg.gram.compute_stats  # whether to compute auxiliary stats
            self.gram_params_lists = None

            if self.gram_ema_teacher and self.gram_ckpt is not None:
                raise ValueError(
                    "Cannot use both `gram.ema_teacher` and `gram.ckpt` at the same time. Please set one of them to False."
                )
            if self.gram_ckpt is None and self.gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "If no gram checkpoint is provided, `gram.it_load_ema_teacher` must be set to a non-negative value."
                )

            assert not (self.gram_ema_teacher and self.gram_rep_update)
            assert self.gram_tokens_used in ["all", "masked", "unmasked"]
            # Currently using masked/unmasked not handle at the image-level
            if self.gram_tokens_used in ["masked", "unmasked"]:
                assert self.gram_img_level is False

            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- loss_weight: {cfg.gram.loss_weight}")
            logger.info(f"OPTIONS -- GRAM -- ema teacher: {cfg.gram.ema_teacher}")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {cfg.gram.ckpt}")
            if self.cfg.gram.rep_update:
                logger.info(f"OPTIONS -- GRAM -- repeated update: {cfg.gram.rep_update}")
                logger.info(f"OPTIONS -- GRAM -- update freq: {cfg.gram.update_frequency}")
                logger.info(f"OPTIONS -- GRAM -- iteration first update: {cfg.gram.it_first_update}")

            logger.info(f"OPTIONS -- GRAM -- tokens_used: {cfg.gram.tokens_used}")
            logger.info(f"OPTIONS -- GRAM -- apply normalization: {cfg.gram.normalized}")
            logger.info(f"OPTIONS -- GRAM -- img_level: {cfg.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {cfg.gram.remove_neg}")
            logger.info(f"OPTIONS -- GRAM -- remove_only_teacher_neg: {cfg.gram.remove_only_teacher_neg}")

            if cfg.crops.gram_teacher_crops_size is None and self.has_gram_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size must be set to use gram loss")
            if cfg.crops.gram_teacher_crops_size is not None and self.gram_ema_teacher:
                raise ValueError("cfg.crops.gram_teacher_crops_size shoud be None when gram.ema_teacher=True")

            self.student_crop_size = cfg.crops.global_crops_size
            self.gram_global_teacher_resize_method = cfg.gram.global_teacher_resize_method
            self.gram_global_teacher_resize_antialias = cfg.gram.global_teacher_resize_antialias
            logger.info(f"OPTIONS -- global crops student/teacher size: {self.student_crop_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher size: {cfg.crops.gram_teacher_crops_size}")
            logger.info(f"OPTIONS -- global crops GRAM teacher resize method: {cfg.gram.global_teacher_resize_method}")
            logger.info(
                f"OPTIONS -- global crops GRAM teacher resize antialias: {cfg.gram.global_teacher_resize_antialias}"
            )

    def _setup_distillation(self):
        logger.info(f"Performing distillation from {self.cfg.distillation.full_cfg_path}")

        default_cfg = get_default_config()
        distillation_cfg = OmegaConf.load(self.cfg.distillation.full_cfg_path)
        distillation_cfg = OmegaConf.merge(default_cfg, distillation_cfg)

        assert distillation_cfg.ibot.separate_head is True
        # assert distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes
        # assert distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes
        
        assert distillation_cfg.ibot.head_n_prototypes == self.cfg.ibot.head_n_prototypes, (
            f"{distillation_cfg.ibot.head_n_prototypes} != {self.cfg.ibot.head_n_prototypes}"
        )
        assert distillation_cfg.dino.head_n_prototypes == self.cfg.dino.head_n_prototypes, (
            f"{distillation_cfg.dino.head_n_prototypes} != {self.cfg.dino.head_n_prototypes}"
        )
        
        # assert distillation_cfg.student.patch_size == self.cfg.student.patch_size

        teacher_model_dict = dict()

        backbone, embed_dim = build_model_from_cfg(distillation_cfg, only_teacher=True)
        teacher_model_dict["backbone"] = backbone

        teacher_model_dict["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.dino.head_n_prototypes,
            hidden_dim=distillation_cfg.dino.head_hidden_dim,
            bottleneck_dim=distillation_cfg.dino.head_bottleneck_dim,
            nlayers=distillation_cfg.dino.head_nlayers,
        )
        teacher_model_dict["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_cfg.ibot.head_n_prototypes,
            hidden_dim=distillation_cfg.ibot.head_hidden_dim,
            bottleneck_dim=distillation_cfg.ibot.head_bottleneck_dim,
            nlayers=distillation_cfg.ibot.head_nlayers,
        )
        self.teacher = nn.ModuleDict(teacher_model_dict)

    def init_weights(self) -> None:
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        if Path(self.cfg.student.dinov3_weights).expanduser().is_file():
            init_fsdp_model_from_dinov3(
                self.student.backbone,
                self.cfg.student.dinov3_weights,
                # skip_load_prefixes=['attn.qkv.bias_mask'],
                prefixes_not_sharded=["rope_embed.periods", "rope_embed_2D.periods", "rope_embed_3D.periods", 
                                      'attn.qkv.bias_mask'],
                process_group=distributed.get_default_process_group(),
                skip_patch_embed = False,
            )
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            if self.gram_ckpt is not None:
                logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
                init_fsdp_model_from_checkpoint(
                    self.gram_teacher,
                    self.gram_ckpt,
                    skip_load_prefixes=[
                        "dino_head",
                        "ibot_head",
                        "dino_loss.center",
                        "ibot_patch_loss.center",
                    ],
                    prefixes_not_sharded=["backbone.rope_embed.periods", "backbone.rope_embed_2D.periods", "backbone.rope_embed_3D.periods", 
                                          "attn.qkv.bias_mask"],
                    process_group=distributed.get_default_process_group(),
                )
                self.gram_teacher_initialized = True
            else:
                raise ValueError(f"Provide a correct path to {self.gram_ckpt}")
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        if self.cfg.student.resume_from_teacher_chkpt:
            logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
            init_fsdp_model_from_checkpoint(
                self.student,
                self.cfg.student.resume_from_teacher_chkpt,
                skip_load_prefixes=[
                    "dino_loss.center",
                    "ibot_patch_loss.center",
                ],
                prefixes_not_sharded=["backbone.rope_embed.periods", "backbone.rope_embed_2D.periods", "backbone.rope_embed_3D.periods", 
                                      "attn.qkv.bias_mask"],
                process_group=distributed.get_process_subgroup(),
            )
            self.model_ema.load_state_dict(self.student.state_dict())
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_prefixes=[],
                    prefixes_not_sharded=["backbone.rope_embed.periods", "backbone.rope_embed_2D.periods", "backbone.rope_embed_3D.periods", 
                                        "attn.qkv.bias_mask"],
                    process_group=distributed.get_process_subgroup(),
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")

    def init_weights_no_fsdp(self) -> None:
        # All weights are set to `nan` to ensure we initialize everything explicitly
        self.student.backbone.init_weights()
        if Path(self.cfg.student.dinov3_weights).expanduser().is_file():
            state_dict = torch.load(self.cfg.student.dinov3_weights, map_location="cpu")["teacher"]
            state_dict = {
                k.replace('backbone.', ''): v
                for k, v in state_dict.items()
                if 'ibot' not in k and 'dino_head' not in k and 'patch_embed_2D' not in k and 'patch_embed_3D' not in k
            }
            self.student.backbone.load_state_dict(state_dict, strict=False)
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()
        self.model_ema.load_state_dict(self.student.state_dict())
        if self.has_gram_teacher:
            if self.gram_ckpt is not None:
                logger.info(f"Loading pretrained weights from {self.gram_ckpt}")
                state_dict = torch.load(self.gram_ckpt, map_location="cpu")["teacher"]
                skip_substr = ["dino_head", "ibot_head", "dino_loss.center", "ibot_patch_loss.center"]
                state_dict = {k: v for k, v in state_dict.items() if not any(s in k for s in skip_substr)}
                self.gram_teacher.load_state_dict(state_dict, strict=True)
                self.gram_teacher_initialized = True
            else:
                raise ValueError(f"Provide a correct path to {self.gram_ckpt}")
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
        if self.cfg.student.resume_from_teacher_chkpt:
            logger.info(f"Loading pretrained weights from {self.cfg.student.resume_from_teacher_chkpt}")
            state_dict = torch.load(self.cfg.student.resume_from_teacher_chkpt, map_location="cpu")["teacher"]
            skip_substr = ["dino_loss.center", "ibot_patch_loss.center"]
            state_dict = {k: v for k, v in state_dict.items() if not any(s in k for s in skip_substr)}
            self.student.load_state_dict(state_dict, strict=True)
            self.model_ema.load_state_dict(self.student.state_dict())
        if self.cfg.distillation.enabled:
            if self.cfg.distillation.checkpoint_path != "ignore":
                logger.info(f"Loading teacher to distil from : {self.cfg.distillation.checkpoint_path}")
                init_fsdp_model_from_checkpoint(
                    self.teacher,
                    self.cfg.distillation.checkpoint_path,
                    skip_load_prefixes=[],
                )
            else:
                logger.info("Init teacher to distil from, used for testing purpose only")
                self.teacher.backbone.init_weights()
                self.teacher.dino_head.init_weights()
                self.teacher.ibot_head.init_weights()
            logger.info(f"Performing distillation from: {self.teacher}")
            
    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = self.n_local_crops  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        # global_crops is either [2B, C, H, W] or [2B, C, H, W, D]
        if global_crops.dim() == 4:
            student_global_crops_size = global_crops.shape[-1]          # keep existing behavior (W)
        elif global_crops.dim() == 5:
            student_global_crops_size = global_crops.shape[-3:]         # (H, W, D)
        else:
            raise ValueError(f"Unexpected global_crops dim={global_crops.dim()} shape={tuple(global_crops.shape)}")

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data, (
                "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            )
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
        else:
            gram_teacher_crops = None

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output (will trigger an all-gather to unshard)
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

        # Compute losses and backprop
        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )

        self.backprop_loss(loss_accumulator)

        # Return total weighted loss and a dict of metrics to log
        return loss_accumulator, metrics_dict | loss_dict

    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        *,
        upperbound,
        mask_indices_list,
        teacher_temp,
        n_masked_patches_tensor,
    ):
        n_crops, B = images.shape[:2]
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.teacher.ibot_head(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.teacher.dino_head(cls)  # [n_crops * B, K]
        if self.cfg.train.centering == "sinkhorn_knopp":
            # Center with sinkhorn-knopp
            cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
                cls_after_head, teacher_temp=teacher_temp
            )  # [n_crops * B, K]
            cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
            masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                masked_patch_after_head,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
            )  # [n_masked_patches, K]

            return {
                "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
                "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
                "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
                "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
                "cls_centered": cls_centered,  # [n_crops, B, K]
                "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
            }
        elif self.cfg.train.centering == "centering":
            # CLS tokens (K-way logits, shape [n_crops * B, K])
            cls_centered = self.dino_loss.softmax_center_teacher(
                cls_after_head, teacher_temp=teacher_temp
            )  # -> probs in [0,1], sum=1 over K
            cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]

            # Queue current CLS logits to update the center for next step
            self.dino_loss.update_center(cls_after_head)
            masked_patch_after_head = masked_patch_after_head.unsqueeze(0)
            masked_patch_centered = self.ibot_patch_loss.softmax_center_teacher(
                masked_patch_after_head, teacher_temp=teacher_temp
            ).squeeze(0)  # [n_masked_patches, K]
            # Update patch center with current masked patch logits
            self.ibot_patch_loss.update_center(masked_patch_after_head)
            return {
                "cls_pre_head": cls.unflatten(0, [n_crops, B]),
                "reg_pre_head": reg.unflatten(0, [n_crops, B]),
                "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),
                "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),
                "cls_centered": cls_centered,                      # [n_crops, B, K]
                "masked_patch_centered": masked_patch_centered,    # [N, K] or None
            }

    def get_gram_teacher_output(self, images, *, masks, teacher_global, student_global, student_global_crops_size):
        # Get student patch features
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]

        # Get gram targets
        if self.gram_ema_teacher:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [n_crops * B, P, D]
        else:
            if not self.gram_teacher_initialized:
                raise ValueError("Gram teacher has not been initialized. Load a checkpoint or from the EMA teacher.")
            
            if images.dim() == 5:
                n_crops, B, rgb, H, W = images.shape
                images = images.flatten(0, 1)  # [n_crops * B, rgb, H, W]

                with torch.no_grad():
                    backbone_out = self.gram_teacher.backbone(images, is_training=True)
                teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]

                # Downsample Gram teacher features if needed
                if teacher_patches.shape[1] != student_patches.shape[1]:
                    N = H // self.gram_teacher.backbone.patch_size
                    # print('current backbone patch size, ', self.gram_teacher.backbone.patch_size, teacher_patches.shape[1], N, 
                    #       student_patches.shape[1], student_global_crops_size )
                    assert teacher_patches.shape[1] == N**2
                    N_student = student_global_crops_size // self.student.backbone.patch_size
                    assert student_patches.shape[1] == N_student**2
                    patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))  # [n_crops * B, D, N, N]
                    patches_hw = torch.nn.functional.interpolate(
                        patches_hw,
                        size=(N_student, N_student),
                        mode=self.gram_global_teacher_resize_method,
                        align_corners=False,
                        antialias=self.gram_global_teacher_resize_antialias,
                    )
                    teacher_patches = patches_hw.flatten(-2, -1).transpose(
                        -2, -1
                    )  # [n_crops * B, N_student * N_student, D]
                    assert teacher_patches.shape == student_patches.shape
            elif images.dim() == 6:
                n_crops, B, C, H, W, D = images.shape
                images = images.flatten(0, 1)  # [n_crops * B, c, H, W, D]
                with torch.no_grad():
                    backbone_out = self.gram_teacher.backbone(images, is_training=True)
                teacher_patches = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P_T, D]
                # Downsample Gram teacher features if needed
                if teacher_patches.shape[1] != student_patches.shape[1]:
                    ps = self.gram_teacher.backbone.patch_size
                    # Teacher grid sizes
                    T_h = H // ps
                    T_w = W // ps
                    T_d = D // ps
                    assert len(student_global_crops_size) == 3
                    S_h = student_global_crops_size[0] // self.student.backbone.patch_size
                    S_w = student_global_crops_size[1] // self.student.backbone.patch_size
                    S_d = student_global_crops_size[2] // self.student.backbone.patch_size
                    assert teacher_patches.shape[1] == T_h  * T_w * T_d
                    assert student_patches.shape[1] == S_h * S_w * S_d
                    patches_hwd = (
                        teacher_patches.transpose(-2, -1)
                        .unflatten(-1, (T_h, T_w, T_d)))   # [NB, D, Nt_h, Nt_w, Nt_d]
                    patches_hwd = torch.nn.functional.interpolate(
                        patches_hwd,
                        size=(S_h,  S_w, S_d),
                        mode="trilinear",
                        align_corners=False,
                    )
                    teacher_patches = patches_hwd.flatten(-3, -1).transpose(
                        -2, -1
                    ) 
                    assert teacher_patches.shape == student_patches.shape
                
        # Select the patches to be considered in the loss
        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            "teacher_patches": teacher_patches,  # [n_crops * B, P, D] or [n_selected_patches, D]
            # Unmasked patches, for computing statistics
            "orig_student_patches": orig_student_patches,  # [n_crops * B, P, D]
            "orig_teacher_patches": orig_teacher_patches,  # [n_crops * B, P, D]
        }

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B = global_crops.shape[:2]
        n_local_crops, B = local_crops.shape[:2]

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks if not self.is_distillation_enabled else None, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.student.ibot_head(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

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
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        # Reweighting of DINO loss
        if self.cfg.dino.reweight_dino_local_loss:
            local_weight = self.dino_local_loss_schedule[iteration]
        else:
            local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        if self.dino_koleo_loss_weight > 0:
            koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
            loss_dict["koleo_loss"] = koleo_loss
            loss_accumulator += self.dino_koleo_loss_weight * koleo_scale * koleo_loss

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
                    # Save stats over masked / unmasked tokens
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

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def gram_load_ema_teacher(self):
        if self.has_gram_teacher:
            skip_load_prefixes = ["dino_head.", "ibot_head."]
            self.gram_teacher.load_state_dict(
                {
                    k: v
                    for k, v in self.model_ema.state_dict().items()
                    if not any(k.startswith(prefix) for prefix in skip_load_prefixes)
                }
            )
            self.gram_teacher.requires_grad_(False)
            self.gram_teacher.eval()
            self.gram_teacher_initialized = True

    def train(self):
        super().train()
        self.teacher.eval()
        if self.has_gram_teacher:
            self.gram_teacher.eval()

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        loss.backward()

    def update_ema(self, m):
        if self.ema_params_lists is None:
            student_param_list = []
            teacher_param_list = []
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.model_ema[k].parameters()):
                    student_param_list += [ms]
                    teacher_param_list += [mt]
            self.ema_params_lists = (student_param_list, teacher_param_list)
        else:
            student_param_list, teacher_param_list = self.ema_params_lists
        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def update_gram(self, m=0):
        if not self.has_gram_teacher:
            return
        logger.info("Updating gram teacher with teacher weights.")
        if self.gram_params_lists is None:
            teacher_param_list = []
            gramteacher_param_list = []
            for k in self.gram_teacher.keys():
                for mgt, mt in zip(self.gram_teacher[k].parameters(), self.teacher[k].parameters()):
                    gramteacher_param_list += [mgt]
                    teacher_param_list += [mt]
            self.gram_params_lists = (gramteacher_param_list, teacher_param_list)
        else:
            gramteacher_param_list, teacher_param_list = self.gram_params_lists

        with torch.no_grad():
            torch._foreach_mul_(gramteacher_param_list, m)
            torch._foreach_add_(gramteacher_param_list, teacher_param_list, alpha=1 - m)

    def build_data_augmentation_dino(self, cfg):
         ## We use no data augmentation instead of the original.
        # return NoDataAugmentationDINO(
        #     cfg.crops.global_crops_scale,
        #     cfg.crops.local_crops_scale,
        #     cfg.crops.local_crops_number,
        #     global_crops_size=cfg.crops.global_crops_size,
        #     local_crops_size=cfg.crops.local_crops_size,
        #     gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        #     gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        #     local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        #     share_color_jitter=cfg.crops.share_color_jitter,
        #     horizontal_flips=cfg.crops.horizontal_flips,
        #     mean=cfg.crops.rgb_mean,
        #     std=cfg.crops.rgb_std,
        # )
        return SimpleDataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
            use_intensity_transforms = True,
        )
    def build_3D_data_augmentation(self, cfg):
         ## We use simple data augmentation instead of the original.
        # return SimpleDataAugmentationDINO3D(
        #     global_crop_overlap = cfg.crops.global_crop_overlap,
        #     local_crops_number = cfg.crops.local_crops_number,
        #     global_crops_size=cfg.crops.global_crops_size,
        #     local_crops_size=cfg.crops.local_crops_size,
        #     gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        #     use_intensity_transforms=True
        # )
        return RandomCropDINO3D(
            local_views_scale = cfg.crops.local_views_scale,
            local_crops_number = cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            use_intensity_transforms=True
        )
    def get_maybe_fused_params_for_submodel(self, m: nn.Module):
        params_groups = get_params_groups_with_decay_fsdp(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
        )
        if self.cfg.optim.multi_tensor_optim:
            fused_params_groups = fuse_params_groups(params_groups)
            logger.info("fusing param groups")

            for g in fused_params_groups:
                g["foreach"] = True
                g["fused"] = True
            return fused_params_groups
        else:
            return params_groups

    def get_params_groups(self):
        all_params_groups = []
        for name, m in self.student.items():
            logger.info(f"Getting paramer groups for {name}")
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self) -> None:
        process_subgroup = distributed.get_process_subgroup()
        default_process_group = distributed.get_default_process_group()
        inference_only_models = [self.model_ema]
        inference_only_models_process_groups = [process_subgroup]
        if self.has_gram_teacher:
            inference_only_models.append(self.gram_teacher)
            inference_only_models_process_groups.append(default_process_group)
        if self.cfg.distillation.enabled:
            inference_only_models.append(self.teacher)
            inference_only_models_process_groups.append(default_process_group)
        ac_compile_parallelize(
            trained_model=self.student,
            inference_only_models=inference_only_models,
            cfg=self.cfg,
            trained_model_process_group=process_subgroup,
            inference_only_models_process_groups=inference_only_models_process_groups,
        )

    def broadcast_to_subgroups(self, tensor, over_dim, global_batch_size=None):
        """
        This is an operation that takes a tensor from the default process group, gathers it, stacks it, then scatters it within a smaller process subgroup
        """
        world_size = distributed.get_world_size()
        subgroup_size = distributed.get_subgroup_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        torch.distributed.all_gather(gathered, tensor)
        catted = torch.cat(gathered, dim=over_dim)
        if global_batch_size is not None:
            catted = catted.narrow(dim=over_dim, start=0, length=global_batch_size)

        return catted.chunk(subgroup_size, dim=over_dim)[distributed.get_subgroup_rank()].clone()
    
    def assert_params_finite(self, module, tag):
        for n, p in module.named_parameters():
            if p is None:
                continue
            if not torch.isfinite(p).all():
                raise RuntimeError(f"[{tag}] param {n} has non-finite values")

class SSLMetaTIPS(SSLMetaArch):
    """
    DINOv3 + CLIP
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward_backward(
            self, data, *, teacher_temp, iteration=0, **ignored_kwargs
        ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        metrics_dict = {}
        