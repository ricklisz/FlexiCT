"""Flexi_CT_2D: 2D CT ViT backbone (DINOv3 SSL on CT-2D slices)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .checkpoints import FLEXICT_2D_DEFAULT_CHECKPOINT, resolve_flexict_checkpoint
from .models import flexi_ct_backbone_base

_BACKBONE_KWARGS = dict(
    patch_size=8,
    in_chans=1,
    n_storage_tokens=4,
    qkv_bias=False,
    mask_k_bias=True,
    drop_path_rate=0.2,
    layerscale_init=1e-5,
)


def _load_teacher_into_backbone(backbone: nn.Module, ckpt_path: str) -> None:
    """Checkpoint has keys like `backbone.patch_embed_2D.*` plus DINO/iBOT head keys.
    Strip the prefix and drop the head keys before loading."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["teacher"]
    sd = {
        k[len("backbone."):]: v
        for k, v in sd.items()
        if k.startswith("backbone.") and "ibot" not in k and "dino_head" not in k
    }
    missing, unexpected = backbone.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing} unexpected={unexpected}")


class Flexi_CT_2D(nn.Module):
    DEFAULT_CHECKPOINT = FLEXICT_2D_DEFAULT_CHECKPOINT

    def __init__(self, checkpoint_path: str | None = None, device: str | torch.device = "cuda"):
        super().__init__()
        self.backbone = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
        _load_teacher_into_backbone(self.backbone, resolve_flexict_checkpoint("2d", checkpoint_path))
        self.to(device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, 1, H, W] with H, W divisible by patch_size (8)."""
        out = self.backbone(x, is_training=True)
        return {
            "cls_token": out["x_norm_clstoken"],          # [B, D]
            "patch_tokens": out["x_norm_patchtokens"],    # [B, N, D]
        }
