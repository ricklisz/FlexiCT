"""Flexi_CT_3D: 3D CT ViT backbone (DINOv3 SSL on CT volumes)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .checkpoints import FLEXICT_3D_DEFAULT_CHECKPOINT, resolve_flexict_checkpoint
from .flexi_ct_2d import _BACKBONE_KWARGS, _load_teacher_into_backbone
from .models import flexi_ct_backbone_base


class Flexi_CT_3D(nn.Module):
    DEFAULT_CHECKPOINT = FLEXICT_3D_DEFAULT_CHECKPOINT

    def __init__(self, checkpoint_path: str | None = None, device: str | torch.device = "cuda"):
        super().__init__()
        self.backbone = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
        _load_teacher_into_backbone(self.backbone, resolve_flexict_checkpoint("3d", checkpoint_path))
        self.to(device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, 1, D, H, W] with D, H, W divisible by patch_size (8)."""
        out = self.backbone(x, is_training=True)
        return {
            "cls_token": out["x_norm_clstoken"],          # [B, D]
            "patch_tokens": out["x_norm_patchtokens"],    # [B, N, D]
        }
