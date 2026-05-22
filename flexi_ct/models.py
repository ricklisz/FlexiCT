# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch.nn.functional as F
import torch
import torch.nn.init
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from .layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, RopePositionEmbedding3D, SelfAttentionBlock, SwiGLUFFN
from .utils import named_apply
import math
import numpy as np

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, PatchEmbedND):
        module.reset_parameters()
    if isinstance(module, PatchEmbed3D):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()

from torch.distributed._tensor import DTensor, distribute_tensor, Replicate
def _replicate_like(x: torch.Tensor, like: DTensor) -> DTensor:
    # Make 'x' a replicated DTensor on the same mesh as 'like'
    if isinstance(x, DTensor):
        return x
    return distribute_tensor(x, device_mesh=like.device_mesh, placements=[Replicate()])


class PatchEmbed3D(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HWD = img_size
        patch_HWD = patch_size

        self.img_size = image_HWD
        self.patch_size = patch_HWD
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_HWD, stride=patch_HWD)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, D, H, W = x.shape
        x = self.proj(x)  # B C D H W
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # B HWD C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1,  D, H, W, self.embed_dim)  # B D H W C
        return x

    def reset_parameters(self):
        # weight: [out_c, in_c/groups, kd, kh, kw]
        nn.init.kaiming_uniform_(self.proj.weight, a=0.0, mode="fan_in", nonlinearity="linear")
        if self.proj.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.proj.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.proj.bias, -bound, bound)

def _to_ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            assert len(x) == n
            return tuple(int(v) for v in x)
        return tuple([int(x)] * n)
    return parse

_to_2 = _to_ntuple(2)
_to_3 = _to_ntuple(3)
class PatchEmbedND(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.img_size = img_size
        self.base_patch_size = _to_2(patch_size) if dim == 2 else _to_3(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        if dim == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim,
                                  kernel_size=self.base_patch_size,
                                  stride=self.base_patch_size)
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim,
                                  kernel_size=self.base_patch_size,
                                  stride=self.base_patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self._pinv_cache = {}
        self._runtime_patch_size: Optional[Tuple[int, ...]] = None  
        
    def set_patch_size(self, new_patch_size: Union[int, Tuple[int, ...]] = 16) -> None:
        """Set runtime patch size (no parameter rebuild)."""
        self._runtime_patch_size = (
            _to_2(new_patch_size) if self.dim == 2 else _to_3(new_patch_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        psize = self._runtime_patch_size or self.base_patch_size

        if psize == self.base_patch_size:
            x = self.proj(x)
        else:
            # Resample to a temporary weight (Tensor or DTensor) and use functional conv
            W = self._resample_conv_weight(self.proj.weight, psize)
            b = self.proj.bias
            if self.dim == 2:
                x = F.conv2d(x, W, b, stride=psize, padding=0)
            else:
                x = F.conv3d(x, W, b, stride=psize, padding=0)

        if self.dim == 2:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # B HW C
            x = self.norm(x)
            if not self.flatten_embedding:
                x = x.reshape(B, H, W, self.embed_dim)  # B H W C
        else:
            B, C, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # B DHW C
            x = self.norm(x)
            if not self.flatten_embedding:
                x = x.reshape(B, D, H, W, self.embed_dim)  # B D H W C
        return x

    # ---- resampling that is Tensor/DTensor/FSDP-friendly ----
    def _resample_conv_weight(self, weight: torch.Tensor, target_size: Tuple[int, ...]) -> torch.Tensor:
        old_spatial = tuple(weight.shape[2:])
        if old_spatial == tuple(target_size):
            return weight

        # Build or fetch pseudoinverse (∏old, ∏new) on same device
        pinv = self._get_or_build_pinv(old_spatial, target_size, weight.device, torch.float32)

        # If weight is DTensor, replicate pinv on same mesh so mm is DTensor x DTensor
        if isinstance(weight, DTensor):
            pinv = distribute_tensor(pinv, device_mesh=weight.device_mesh, placements=[Replicate()])

        c_out, c_in = weight.shape[:2]
        old_total = int(np.prod(old_spatial))

        w = weight.to(torch.float32).reshape(c_out, c_in, old_total)
        w = w @ pinv  # -> (c_out, c_in, ∏new)
        w = w.reshape(c_out, c_in, *target_size).to(weight.dtype)
        return w

    def _get_or_build_pinv(
        self,
        old_size: Tuple[int, ...],
        new_size: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (self.dim, old_size, new_size, device.type)
        pinv = self._pinv_cache.get(key)
        if pinv is not None:
            return pinv.to(device=device, dtype=dtype)

        old_total = int(np.prod(old_size))
        new_total = int(np.prod(new_size))
        eye = torch.eye(old_total, device=device, dtype=dtype)
        basis = eye.reshape(old_total, 1, *old_size)

        if self.dim == 2:
            out = F.interpolate(basis, size=new_size, mode="bicubic", antialias=True, align_corners=False)
            R = out.squeeze(1).permute(1, 2, 0).reshape(new_total, old_total)
        else:
            out = F.interpolate(basis, size=new_size, mode="trilinear", align_corners=False)
            R = out.squeeze(1).permute(1, 2, 3, 0).reshape(new_total, old_total)
        pinv = torch.linalg.pinv(R).to(dtype)  # (∏old, ∏new)
        self._pinv_cache[key] = pinv.detach()
        return pinv

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.proj.weight, a=0.0, mode="fan_in", nonlinearity="linear")
        if self.proj.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.proj.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.proj.bias, -bound, bound)


class Flexi_CT_Core(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed_2D = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.patch_embed_3D = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        # Optional, but very helpful:
        assert embed_dim % (6 * num_heads) == 0, \
            f"embed_dim ({embed_dim}) must be divisible by 6*num_heads ({6*num_heads}) for 3D RoPE"

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed_2D = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        self.rope_embed_3D = RopePositionEmbedding3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed_2D._init_weights()
        self.rope_embed_3D._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        if x.dim() == 5:
            x = self.patch_embed_3D(x)
            B, D, H, W, _ = x.shape
            x = x.flatten(1, 3)
        else:
            x = self.patch_embed_2D(x)
            B, H, W,  _ = x.shape
            D = None
            x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, ((D, H, W) if D is not None else (H, W))

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
            
        if len(rope[0]) == 3:
            rope_sincos = [self.rope_embed_3D(D=D, H=H, W=W) for D, H, W in rope]
        else:
            rope_sincos = [self.rope_embed_2D(H=H, W=W) for H, W in rope]
                
        for _, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos)
            
        n_storage_tokens = self.n_storage_tokens
        norm = self.norm
        cls_norm = self.cls_norm
        local_cls_norm = self.local_cls_norm
        untie_cls_and_patch_norms = self.untie_cls_and_patch_norms
        untie_global_and_local_cls_norm = self.untie_global_and_local_cls_norm
        training = self.training
    
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if untie_cls_and_patch_norms or untie_global_and_local_cls_norm:
                if untie_global_and_local_cls_norm and training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = local_cls_norm(x[:, : n_storage_tokens + 1])
                elif untie_cls_and_patch_norms:
                    x_norm_cls_reg = cls_norm(x[:, : n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = norm(x[:, : n_storage_tokens + 1])
                x_norm_patch = norm(x[:, n_storage_tokens + 1 :])
            else:
                x_norm = norm(x)
                x_norm_cls_reg = x_norm[:, : n_storage_tokens + 1]
                x_norm_patch = x_norm[:, n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, hw_tuple = self.prepare_tokens_with_masks(x)
        if len(hw_tuple) == 3:
            D, H, W = hw_tuple
            rope_sincos = self.rope_embed_3D(D=D, H=H, W=W)
        else:
            H, W = hw_tuple
            rope_sincos = self.rope_embed_2D(H=H, W=W)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            if x.dim() == 5:
                B, _, d, h, w = x.shape
                outputs = [
                    out.reshape(B, d // self.patch_size, h // self.patch_size, w // self.patch_size, -1).permute(0, 4, 1, 2, 3).contiguous()
                    for out in outputs
                ]
            else:
                B, _, h, w = x.shape
                outputs = [
                    out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    @torch.no_grad()
    def inflate_patch_embed3d_from_2d(
        self,
        mode: str = "avg",   # "avg" or "center"
    ) -> None:
        """
        Initialize PatchEmbed3D weights by inflating PatchEmbed (2D) weights.

        Args:
            pe2d: 2D patch embed module with Conv2d `proj` of shape [C_out, C_in, kH, kW].
            pe3d: 3D patch embed module with Conv3d `proj` of shape [C_out, C_in, kD, kH, kW].
            mode:
                - "avg":   copy the 2D kernel into each temporal slice and divide by kD (I3D-style).
                - "center": copy into the center slice only; others set to 0.
        """
        assert isinstance(self.patch_embed_2D.proj, nn.Conv2d) and isinstance(self.patch_embed_3D.proj, nn.Conv3d), \
            "pe2d.proj must be Conv2d and pe3d.proj must be Conv3d"

        w2 = self.patch_embed_2D.proj.weight.data      # [C_out, C_in, kH2, kW2]
        b2 = self.patch_embed_2D.proj.bias.data if self.patch_embed_2D.proj.bias is not None else None

        w3 = self.patch_embed_3D.proj.weight.data      # [C_out, C_in, kD3, kH3, kW3]
        b3 = self.patch_embed_3D.proj.bias.data if self.patch_embed_3D.proj.bias is not None else None

        C_out2, C_in2, kH2, kW2 = w2.shape
        C_out3, C_in3, kD3, kH3, kW3 = w3.shape

        # Basic sanity checks
        assert C_out2 == C_out3, f"out_channels mismatch: {C_out2} vs {C_out3}"
        assert C_in2  == C_in3,  f"in_channels mismatch: {C_in2} vs {C_in3}"
        assert kH2    == kH3 and kW2 == kW3, \
            f"spatial kernel mismatch: (kH,kW)=({kH2},{kW2}) vs ({kH3},{kW3})"

        # Inflate: start from zeros
        w3.zero_()

        if mode == "avg":
            # Copy into every temporal slice and average across time
            # So the sum over temporal slices reproduces the 2D response.
            for t in range(kD3):
                w3[:, :, t, :, :].copy_(w2 / kD3)
        elif mode == "center":
            center = kD3 // 2
            w3[:, :, center, :, :].copy_(w2)
        else:
            raise ValueError(f"Unknown mode='{mode}', expected 'avg' or 'center'.")

        # Copy bias if present (identical)
        if b2 is not None and b3 is not None:
            b3.copy_(b2)


class Flexi_CT_Backbone(Flexi_CT_Core):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        # Call parent class constructor with all required parameters
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            n_storage_tokens=n_storage_tokens,
            mask_k_bias=mask_k_bias,
            untie_cls_and_patch_norms=untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
            device=device,
            **ignored_kwargs,
        )
        
        self.patch_embed_2D = PatchEmbedND(
            dim = 2,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.patch_embed_3D = PatchEmbedND(
            dim = 3,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

class Flexi_CT_VLM_Module(nn.Module):
    def __init__(
        self,
        *,
        vision_model: nn.Module,
        text_model: nn.Module,
        embed_dim = 1024,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        # Call parent class constructor with all required parameters
        self.vision_model = vision_model
        self.text_model = text_model
        self.logit_scale = nn.Parameter(torch.empty(1))
        self.vlm_embed_dim = embed_dim
        self.vlm_vision_projection = nn.Linear(2*vision_model.embed_dim, self.vlm_embed_dim, bias=False)
        self.device = device

    def forward(self, images: torch.Tensor, text) -> torch.Tensor:
        vision_features = self.vision_model(images, is_training = True)
        cls_token = vision_features["x_norm_clstoken"] # [B, D]
        patch_tokens = vision_features["x_norm_patchtokens"]  # [B, P, D]
        # Mean pool patch tokens (like DINOTxt)
        mean_patch_token = torch.mean(patch_tokens, dim=1)  # [B, D]
        # Concatenate CLS + mean(patch) along channel dimension (like DINOTxt)
        image_features = torch.cat([cls_token, mean_patch_token], dim=-1)  # [B, 2*D]
        # Project vision features to VLM embedding space
        image_features = self.vlm_vision_projection(image_features)  # [B, vlm_embed_dim]

        # Normalize image features
        image_features = F.normalize(image_features, dim=-1)
        text_features = self.text_model(**text)    
        # Normalize text features
        text_features = F.normalize(text_features, dim=-1)

        return self.logit_scale.exp(),  image_features,  text_features


def flexi_ct_backbone_base(patch_size=8, **kwargs):
    model = Flexi_CT_Backbone(
        patch_size=patch_size,
        embed_dim=864,
        depth=16,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model

