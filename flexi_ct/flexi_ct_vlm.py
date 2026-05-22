"""Flexi_CT_VLM: CT vision-language model (TIPS-style: Flexi CT encoder + Qwen3-Embedding)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoints import FLEXICT_VLM_DEFAULT_CHECKPOINT, resolve_flexict_checkpoint
from .flexi_ct_2d import _BACKBONE_KWARGS
from .models import Flexi_CT_VLM_Module, flexi_ct_backbone_base
from .text_tower import build_hf_text_model


class Flexi_CT_VLM(nn.Module):
    DEFAULT_CHECKPOINT = FLEXICT_VLM_DEFAULT_CHECKPOINT
    TEXT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        checkpoint_path: str | None = None,
        text_model_id: str | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        vision_model = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
        text_model = build_hf_text_model(
            model_name_or_path=text_model_id or self.TEXT_MODEL_ID,
            embed_dim=1024,
            pooling_type="last_token",
            use_flash_attention=False,
            torch_dtype="float32",
            freeze_backbone=False,
            use_projection=True,
            max_length=8192,
            padding_side="left",
        )
        text_model.init_weights(sharded=False)  # materializes HF backbone + projection
        self.model = Flexi_CT_VLM_Module(vision_model=vision_model, text_model=text_model, embed_dim=1024)
        self._load(resolve_flexict_checkpoint("vlm", checkpoint_path))
        self.tokenizer = text_model.get_tokenizer()
        self.to(device)

    def _load(self, ckpt_path: str) -> None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        vision_missing = [k for k in missing if k.startswith("vision_model.")]
        vision_unexp = [k for k in unexpected if k.startswith("vision_model.")]
        if vision_missing or vision_unexp:
            raise RuntimeError(
                f"Vision weights mismatch: missing={vision_missing[:5]} unexpected={vision_unexp[:5]}"
            )

    def encode_image(self, volume: torch.Tensor) -> torch.Tensor:
        """volume: [B, 1, D, H, W] with spatial dims divisible by patch_size (8). Returns [B, 1024]."""
        feats = self.model.vision_model(volume, is_training=True)
        joint = torch.cat([feats["x_norm_clstoken"], feats["x_norm_patchtokens"].mean(dim=1)], dim=-1)
        return F.normalize(self.model.vlm_vision_projection(joint), dim=-1)

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        batch = self.model.text_model.tokenize(prompts)
        feats = self.model.text_model(text_tokens=batch)
        return F.normalize(feats, dim=-1)

    def similarity(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Cosine similarity scaled by learned temperature. Shape: [B_img, B_text]."""
        return self.model.logit_scale.exp() * image_emb @ text_emb.T
