# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional, Sequence, Dict

import torch
from torch import Tensor, nn

from dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale  # , DropPath

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


class SelfAttentionBlockMoEMLP(SelfAttentionBlock):
    """
    A SelfAttentionBlock whose FFN is a Mixture-of-Experts keyed by modality.
    Attention path is unchanged. FFN path is routed per-sample using `modalities_list`.

    Usage:
        - Construct with a set of modality names and it will create one MLP expert per modality.
        - In forward, pass `modalities_list` aligned to each view's batch.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        modality_names: Sequence[str],                 # e.g. ["CT","MRI","PET"]
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        # Build the standard block first
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            mask_k_bias=mask_k_bias,
            device=device,
        )

        # Replace the shared FFN with Identity and create per-modality experts.
        # (We keep all other attributes from the parent.)
        self.mlp = nn.Identity()

        self.modality_names = list(modality_names)
        if len(self.modality_names) == 0:
            raise ValueError("`modality_names` must be a non-empty sequence.")

        hidden = int(dim * ffn_ratio)
        self.mlp_experts = nn.ModuleDict({
            m: ffn_layer(
                in_features=dim,
                hidden_features=hidden,
                act_layer=act_layer,
                drop=drop,
                bias=ffn_bias,
                device=device,
            )
            for m in self.modality_names
        })

    # -------------------------
    # Routing helpers
    # -------------------------
    @staticmethod
    def _group_indices_by_modality(mods: List[str]) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for i, m in enumerate(mods):
            buckets.setdefault(m, []).append(i)
        return buckets
    
    def _mlp_route_view(
        self,
        norm2_view: Tensor,                 # [Bv, T, D]
        mods_view: List[str],               # len == Bv
    ) -> Tensor:
        """Apply the correct expert per sample for one view, preserving batch order."""
        Bv, T, D = norm2_view.shape
        out = torch.empty_like(norm2_view)

        buckets = self._group_indices_by_modality(mods_view)
        for m, idxs in buckets.items():
            if m not in self.mlp_experts:
                raise KeyError(f"Unknown modality '{m}'. Available: {list(self.mlp_experts.keys())}")
            x_sel = norm2_view[idxs, :, :]         # [b_m, T, D]
            out_sel = self.mlp_experts[m](x_sel)   # [b_m, T, D]
            if out_sel.shape != x_sel.shape:
                raise RuntimeError(f"Expert '{m}' returned shape {tuple(out_sel.shape)}; "
                                   f"expected {tuple(x_sel.shape)}.")
            out[idxs, :, :] = out_sel
        return out
    
    def _forward_list(
        self,
        x_list: List[Tensor],
        rope_list=None,
        *,
        modalities_list,   # List[str] aligned with x_list; order matches each view's batch
    ) -> List[Tensor]:
        if rope_list is None:
            rope_list = [None for _ in x_list]

        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1.0 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / s for b, s in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            # -------- Attention (subset) --------
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:s]
                for x, b, s in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[idxs] for x, idxs in zip(x_list, indices_1_list)]

            rope_subset_list = [
                self._maybe_index_rope(r, idxs) if r is not None else None
                for r, idxs in zip(rope_list, indices_1_list)
            ]

            flat_1, shapes_1, ntok_1 = cat_keep_shapes(x_subset_1_list)
            norm1_list = uncat_with_shapes(self.norm1(flat_1), shapes_1, ntok_1)
            residual_1_list = self.attn.forward_list(norm1_list, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x, dim=0, index=idxs, source=self.ls1(resid), alpha=scale
                )
                for x, resid, idxs, scale in zip(x_list, residual_1_list, indices_1_list, residual_scale_factors)
            ]

            # -------- MLP (subset, routed) --------
            indices_2_list = [
                (torch.randperm(b, device=x.device))[:s]
                for x, b, s in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x_attn[idxs] for x_attn, idxs in zip(x_attn_list, indices_2_list)]

            flat_2, shapes_2, ntok_2 = cat_keep_shapes(x_subset_2_list)
            norm2_list = uncat_with_shapes(self.norm2(flat_2), shapes_2, ntok_2)

            residual_2_list = []
            for norm2_view, mods_view, idxs in zip(norm2_list, modalities_list, indices_2_list):
                # order is guaranteed to match; just index
                mods_subset = [mods_view[i] for i in idxs.tolist()]
                residual_2_list.append(self._mlp_route_view(norm2_view, mods_subset))

            x_out = [
                torch.index_add(
                    x_attn, dim=0, index=idxs, source=self.ls2(resid2), alpha=scale
                )
                for x_attn, resid2, idxs, scale in zip(x_attn_list, residual_2_list, indices_2_list, residual_scale_factors)
            ]
            return x_out

        else:
            out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                norm2 = self.norm2(x_attn)
                mlp_out = self._mlp_route_view(norm2, modalities_list)
                out.append(x_attn + self.ls2(mlp_out))
            return out
        
    def forward(
        self,
        x_or_x_list,
        rope_or_rope_list=None,
        *,
        modalities_list,   # List[str] for single view OR List[List[str]] for multi-view
    ):
        # Single-tensor path: accept List[str] and wrap to List[List[str]]
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list(
                [x_or_x_list],
                rope_list=[rope_or_rope_list],
                modalities_list=modalities_list,
            )[0]

        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list, modalities_list=modalities_list)
        else:
            raise AssertionError
