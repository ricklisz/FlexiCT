# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods
        
class RopePositionEmbedding3D(nn.Module):
    """
    RoPE positional embedding for 3D grids with no mixing across axes (axial),
    and no learnable weights.

    Supports two parametrizations of the rope parameters: either using `base`
    or `min_period` + `max_period`.

    Returns (sin, cos) each shaped [D*H*W, D_head], suitable for applying to q/k.
    """
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # For 3 axes, per-axis rotary block is D_head//6
        assert embed_dim % (6 * num_heads) == 0, \
            "For 3D RoPE, embed_dim must be divisible by 6 * num_heads."
        both_periods = (min_period is not None) and (max_period is not None)
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Keep a persistent buffer so teacher.load_state_dict(student.state_dict()) works.
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 6, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        per_axis_block = self.D_head // 6  # 3 axes × per_axis_block = D_head//2

        if self.base is not None:
            # Classic exponential spectrum
            periods = self.base ** (
                2 * torch.arange(per_axis_block, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [per_axis_block]
        else:
            # Linearly spaced in log-period between min_period and max_period
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, per_axis_block, device=device, dtype=dtype)
            periods = base**exponents               # [1, base]
            periods = periods / base                # [1/base, 1]
            periods = periods * self.max_period     # [min_period, max_period]
        self.periods.data = periods

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            D, H, W: depth, height, width (integers)

        Returns:
            (sin, cos): two tensors of shape [D*H*W, D_head]
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # --- Build normalized coords in [-1, +1] for each axis ---
        if self.normalize_coords == "max":
            m = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / m  # [D]
            coords_h = torch.arange(0.5, H, **dd) / m  # [H]
            coords_w = torch.arange(0.5, W, **dd) / m  # [W]
        elif self.normalize_coords == "min":
            m = min(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / m
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Meshgrid in (D, H, W) order; last dim stacks (d, h, w)
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"),
            dim=-1
        )  # [D, H, W, 3]
        coords = coords.flatten(0, 2)               # [DHW, 3]
        coords = 2.0 * coords - 1.0                 # shift [0,1] -> [-1,1]

        # --- Optional data-aug on coords (train-time only) ---
        if self.training and self.shift_coords is not None:
            shift_dhw = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift_dhw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_dhw = torch.empty(3, **dd).uniform_(jitter_min, jitter_max).exp()
            coords = coords * jitter_dhw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords = coords * rescale

        # --- Build rotary angles (axial, no mixing) ---
        # periods: [P] where P = D_head//6
        # coords:  [N, 3], broadcast => [N, 3, P]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [N, 3, P]
        angles = angles.flatten(1, 2)  # [N, 3*P] == [N, D_head//2]
        angles = angles.tile(2)        # [N, D_head]

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return (sin, cos)