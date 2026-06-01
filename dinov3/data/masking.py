# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random
from typing import Optional, Tuple
import numpy as np

class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)

class RCCMaskingGenerator:
    """
    Region Collaborative Cutout (RCC) for DINOv2-style masking.

    - Works on PATCH grid (H x W), returns bool mask with True = masked.
    - Follows the original RCC logic:
        * one bbox per grid cell (normalized coords)
        * process boxes largest-first
        * "recover" a box if its masked ratio already exceeds cut_ratio
        * then cut a single sub-rectangle in that box to approach cut_ratio
    - If under target globally, fills the remainder randomly (like DINOv2).

    Args (DINOv2-compatible where relevant):
        input_size: int or (H, W) in patches
        num_masking_patches: default target per call if __call__ arg is None/<=0
        min_num_patches, max_num_patches, min_aspect, max_aspect: kept for parity (unused)
    RCC knobs:
        grid_num: number of grid cells per axis (default 3 => 3x3)
        box_area_scale: normalized area per bbox (wrt full image), e.g. (0.12, 0.30)
        box_aspect: aspect ratio range for bbox sampling
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
        *,
        grid_num: int = 3,
        box_area_scale=(0.12, 0.30),
        box_aspect=(0.5, 2.0),
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or (1.0 / min_aspect)
        # kept for repr parity with DINOv2
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        # RCC knobs
        self.grid_num = grid_num
        self.box_area_scale = box_area_scale
        self.box_aspect = box_aspect

    def __repr__(self):
        return (
            f"RCCGenerator(H={self.height}, W={self.width}, "
            f"num={self.num_masking_patches}, grid={self.grid_num}, "
            f"box_area={self.box_area_scale}, box_ar={self.box_aspect})"
        )

    def get_shape(self):
        return self.height, self.width

    def __call__(self, num_masking_patches=0):
        H, W = self.get_shape()
        total = H * W

        # None => use default; 0 => mask nothing
        if num_masking_patches is None:
            target = int(self.num_masking_patches or 0)
        else:
            target = int(num_masking_patches)

        target = max(0, min(target, total))
        if target == 0:
            return np.zeros((H, W), dtype=bool)

        cut_ratio = target / float(total)

        # RCC works on a "keep" mask (True = keep/unmasked)
        keep = np.ones((H, W), dtype=bool)

        # 1) make one bbox per grid cell (normalized coords)
        bboxes = self._get_bboxs_in_grid(
            scale=self.box_area_scale, ratio=self.box_aspect, grid_num=self.grid_num
        )

        # sort boxes by area (desc), then apply RCC per box
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        order = np.argsort(-areas)

        for bid in order:
            x1, y1, x2, y2 = self._to_patch_box(bboxes[bid], W, H)
            if x2 < x1 or y2 < y1:
                continue

            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            area = box_w * box_h

            # current masked ratio in this box (masked = ~keep)
            cur_keep = int(keep[y1:y2 + 1, x1:x2 + 1].sum())
            cur_masked = area - cur_keep
            cur_ratio = cur_masked / max(1, area)

            # "recover" over-cut boxes (as in the original RCC)
            if cur_ratio > cut_ratio:
                keep[y1:y2 + 1, x1:x2 + 1] = True  # restore to unmasked
                cur_ratio = 0.0
                cur_masked = 0

            # how many more to mask in this box to reach cut_ratio
            desired_masked = int(round(cut_ratio * area))
            need = desired_masked - cur_masked
            if need <= 0:
                continue

            # carve ONE rectangle inside the box with area ~need
            # use the box aspect as guidance (matches original)
            box_ar = box_w / max(1, box_h)
            w = max(1, int(round(math.sqrt(need * box_ar))))
            h = max(1, int(round(math.sqrt(need / max(1e-6, box_ar)))))
            w = min(w, box_w)
            h = min(h, box_h)

            # random top-left inside the bbox
            rx1 = x1 + random.randint(0, box_w - w)
            ry1 = y1 + random.randint(0, box_h - h)
            rx2 = rx1 + w - 1
            ry2 = ry1 + h - 1

            keep[ry1:ry2 + 1, rx1:rx2 + 1] = False  # mask that sub-rect

        # final mask (True = masked)
        mask = ~keep
        return mask

    # ---------- RCC helpers ----------
    @staticmethod
    def _get_bboxs_in_grid(scale=(0.12, 0.30), ratio=(0.5, 2.0), grid_num=3):
        """
        One bbox per grid cell in normalized coords [x1,y1,x2,y2].
        """
        K = grid_num * grid_num
        bboxes = np.zeros((K, 4), dtype=np.float32)
        for gid in range(K):
            gx, gy = gid % grid_num, gid // grid_num
            gw = 1.0 / grid_num
            gh = 1.0 / grid_num
            # shrink cell slightly so boxes don’t always hug borders
            gx1, gx2 = gx / grid_num + gw / 4.0, (gx + 1) / grid_num - gw / 4.0
            gy1, gy2 = gy / grid_num + gh / 4.0, (gy + 1) / grid_num - gh / 4.0

            # sample bbox area/aspect in normalized space (wrt full image)
            target_area = random.uniform(*scale)
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            ar = math.exp(random.uniform(*log_ratio))
            w = math.sqrt(target_area * ar)
            h = math.sqrt(target_area / max(1e-6, ar))

            xc = random.uniform(gx1, gx2)
            yc = random.uniform(gy1, gy2)

            x1, y1 = max(0.0, xc - w / 2.0), max(0.0, yc - h / 2.0)
            x2, y2 = min(1.0, xc + w / 2.0), min(1.0, yc + h / 2.0)
            bboxes[gid] = (x1, y1, x2, y2)
        return bboxes

    @staticmethod
    def _to_patch_box(b, W, H):
        """
        Convert normalized bbox to inclusive patch coords (fixes the W/H mixup seen in some RCC snippets).
        """
        x1n, y1n, x2n, y2n = b
        x1 = int(round(x1n * (W - 1))); x2 = int(round(x2n * (W - 1)))
        y1 = int(round(y1n * (H - 1))); y2 = int(round(y2n * (H - 1)))
        return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)

    # ---------- DINOv2 parity ----------
    @staticmethod
    def complete_mask_randomly(mask, num_masking_patches):
        """
        If RCC undershoots the target, fill the remainder by masking random unmasked patches.
        (Exact DINOv2 behavior to hit the global count.)
        """
        shape = mask.shape
        flat = mask.reshape(-1)
        need = int(num_masking_patches) - int(flat.sum())
        if need > 0:
            idx = np.where(~flat)[0]
            need = min(need, len(idx))
            if need > 0:
                pick = np.random.choice(idx, size=need, replace=False)
                flat[pick] = True
        return flat.reshape(shape)

class MaskingGenerator3D:
    def __init__(
        self,
        input_size,
        num_masking_voxels=None,
        min_num_voxels=8,
        max_num_voxels=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        """
        Args:
            input_size (tuple or int): A 3-tuple (D, H, W) specifying the dimensions of the volume.
                If a single int is provided, it will assume a cube.
            num_masking_voxels (int): Total number of voxels to mask.
            min_num_voxels (int): Minimum volume (number of voxels) for a single cuboid patch.
            max_num_voxels (int): Maximum volume for a single cuboid patch (defaults to num_masking_voxels).
            min_aspect (float): Minimum aspect ratio for the cuboid dimensions.
            max_aspect (float): Maximum aspect ratio for the cuboid dimensions. If not provided,
                                defaults to 1/min_aspect.
                                
        The aspect ratios are used to control the relative sizes between dimensions.
        We will sample two independent aspect ratios (for example, d/h and d/w) in log-space.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size, input_size)
        if len(input_size) != 3:
            raise ValueError("input_size must be a tuple of three ints (depth, height, width)")
            
        self.depth, self.height, self.width = input_size

        self.num_voxels = self.depth * self.height * self.width
        self.num_masking_voxels = num_masking_voxels

        self.min_num_voxels = min_num_voxels
        self.max_num_voxels = num_masking_voxels if max_num_voxels is None else max_num_voxels

        max_aspect = max_aspect or 1 / min_aspect
        # We use the same aspect range for both sampled ratios.
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = (
            f"MaskingGenerator3D({self.depth}, {self.height}, {self.width} -> "
            f"[{self.min_num_voxels} ~ {self.max_num_voxels}], "
            f"max = {self.num_masking_voxels}, "
            f"aspect (log) range = {self.log_aspect_ratio[0]:.3f} ~ {self.log_aspect_ratio[1]:.3f})"
        )
        return repr_str

    def get_shape(self):
        return self.depth, self.height, self.width

    def _mask(self, mask, max_mask_voxels):
        delta = 0
        # Try up to 10 times to generate a cuboid patch
        for _ in range(10):
            # Choose a random target volume between the minimum and allowed maximum for this iteration
            target_volume = random.uniform(self.min_num_voxels, max_mask_voxels)
            # Sample two aspect ratios in log-space. They will roughly control d/h and d/w.
            ratio_dh = math.exp(random.uniform(*self.log_aspect_ratio))
            ratio_dw = math.exp(random.uniform(*self.log_aspect_ratio))
            # Compute cuboid dimensions such that: d * h * w ~ target_volume.
            # One valid solution is:
            #   d = (target_volume * ratio_dh * ratio_dw)^(1/3)
            #   h = (target_volume * (ratio_dw / ratio_dh))^(1/3)
            #   w = (target_volume / ratio_dw)^(1/3)
            d = int(round((target_volume * ratio_dh * ratio_dw) ** (1/3)))
            h = int(round((target_volume * (ratio_dw / ratio_dh)) ** (1/3)))
            w = int(round((target_volume / ratio_dw) ** (1/3)))

            # Ensure the dimensions are at least 1
            d, h, w = max(d, 1), max(h, 1), max(w, 1)

            # Check if the cuboid fits within the volume dimensions
            if d < self.depth and h < self.height and w < self.width:
                # Randomly choose a starting location within the volume
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                front = random.randint(0, self.depth - d)

                # Count how many voxels are already masked in the selected cuboid
                num_masked = mask[front : front + d, top : top + h, left : left + w].sum()
                # We allow masking only the voxels that are not yet masked, as long as
                # the number of new voxels (d * h * w - num_masked) is positive and does not exceed max_mask_voxels.
                if 0 < d * h * w - num_masked <= max_mask_voxels:
                    # Mask the unmasked voxels in the cuboid.
                    for f in range(front, front + d):
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if not mask[f, i, j]:
                                    mask[f, i, j] = True
                                    delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_voxels=0):
        """
        Generates a 3D mask of shape (depth, height, width), where True indicates a masked voxel.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_voxels:
            remaining_voxels = num_masking_voxels - mask_count
            # Limit the cuboid patch volume to be no larger than both remaining_voxels and self.max_num_voxels.
            max_mask_voxels = min(remaining_voxels, self.max_num_voxels)
            delta = self._mask(mask, max_mask_voxels)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask
    
class RCCMaskingGenerator3D:
    """
    Region Collaborative Cutout (RCC) for DINOv2-style masking in 3D.

    - Works on PATCH grid (D x H x W), returns bool mask with True = masked.
    - Follows the original RCC logic extended to 3D:
        * one bbox per grid cell in normalized coords [x1, y1, z1, x2, y2, z2]
        * process boxes largest-first (by volume)
        * "recover" a box if its masked ratio already exceeds cut_ratio
        * then cut a single sub-cuboid in that box to approach cut_ratio
    - Optionally, if under target globally, you can call complete_mask_randomly
      (same behavior as your 2D version, generalized to 3D).

    Args (DINOv2-compatible where relevant):
        input_size: int or (D, H, W) in patches
        num_masking_patches: default target per call if __call__ arg is None/<=0
        min_num_patches, max_num_patches, min_aspect, max_aspect: kept for parity (unused)
    RCC knobs:
        grid_num: number of grid cells per axis (default 3 => 3x3x3)
        box_area_scale: in 3D, interpreted as normalized volume fraction per bbox
                        (wrt full volume), e.g. (0.12, 0.30)
        box_aspect: aspect ratio range used to guide width/height of the box
                    (depth is then chosen to match volume)
    """

    def __init__(
        self,
        input_size,
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
        *,
        grid_num: int = 3,
        box_area_scale: Tuple[float, float] = (0.12, 0.30),
        box_aspect: Tuple[float, float] = (0.5, 2.0),
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size, input_size)
        if len(input_size) != 3:
            raise ValueError("input_size must be int or a tuple (D, H, W) in patches.")

        self.depth, self.height, self.width = input_size

        self.num_patches = self.depth * self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or (1.0 / min_aspect)
        # kept for repr parity with DINOv2
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        # RCC knobs
        self.grid_num = grid_num
        # In 3D we interpret this as *volume* fraction in [0,1]
        self.box_area_scale = box_area_scale
        # Used to guide width/height; depth is derived to match volume
        self.box_aspect = box_aspect

    def __repr__(self):
        return (
            f"RCCGenerator3D(D={self.depth}, H={self.height}, W={self.width}, "
            f"num={self.num_masking_patches}, grid={self.grid_num}, "
            f"box_vol={self.box_area_scale}, box_ar={self.box_aspect})"
        )

    def get_shape(self):
        return self.depth, self.height, self.width

    # ---------- main call ----------

    def __call__(self, num_masking_patches: Optional[int] = None):
        D, H, W = self.get_shape()
        total = D * H * W

        # IMPORTANT: match MaskingGenerator3D semantics
        #   - None => use default (self.num_masking_patches)
        #   - 0    => mask nothing
        if num_masking_patches is None:
            target = int(self.num_masking_patches or 0)
        else:
            target = int(num_masking_patches)

        target = max(0, min(target, total))
        if target == 0:
            return np.zeros((D, H, W), dtype=bool)

        cut_ratio = target / float(total)

        keep = np.ones((D, H, W), dtype=bool)

        bboxes = self._get_bboxs_in_grid_3d(
            scale=self.box_area_scale,
            ratio=self.box_aspect,
            grid_num=self.grid_num,
        )

        volumes = (
            (bboxes[:, 3] - bboxes[:, 0])
            * (bboxes[:, 4] - bboxes[:, 1])
            * (bboxes[:, 5] - bboxes[:, 2])
        )
        order = np.argsort(-volumes)

        for bid in order:
            x1, y1, z1, x2, y2, z2 = self._to_voxel_box(bboxes[bid], W, H, D)
            if x2 < x1 or y2 < y1 or z2 < z1:
                continue

            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            box_d = z2 - z1 + 1
            volume = box_w * box_h * box_d

            cur_keep = int(keep[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1].sum())
            cur_masked = volume - cur_keep
            cur_ratio = cur_masked / max(1, volume)

            if cur_ratio > cut_ratio:
                keep[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = True
                cur_masked = 0

            desired_masked = int(round(cut_ratio * volume))
            need = desired_masked - cur_masked
            if need <= 0:
                continue

            box_ar_hw = box_w / max(1, box_h)
            w = max(1, int(round(math.sqrt(need * box_ar_hw))))
            h = max(1, int(round(math.sqrt(need / max(1e-6, box_ar_hw)))))
            w = min(w, box_w)
            h = min(h, box_h)

            base_area = max(1, w * h)
            d = int(math.ceil(need / base_area))
            d = max(1, min(d, box_d))

            rx1 = x1 + random.randint(0, box_w - w)
            ry1 = y1 + random.randint(0, box_h - h)
            rz1 = z1 + random.randint(0, box_d - d)
            rx2 = rx1 + w - 1
            ry2 = ry1 + h - 1
            rz2 = rz1 + d - 1

            keep[rz1 : rz2 + 1, ry1 : ry2 + 1, rx1 : rx2 + 1] = False

        mask = ~keep
        # # Enforce exact count WITHOUT changing RCC sampling above
        # cur = int(mask.sum())
        # if cur < target:
        #     mask = self.complete_mask_randomly(mask, target)
        # elif cur > target:
        #     mask = self._trim_mask_randomly(mask, target)
        return mask
    # ---------- RCC helpers (3D) ----------

    @staticmethod
    def _get_bboxs_in_grid_3d(
        scale=(0.12, 0.30),
        ratio=(0.5, 2.0),
        grid_num: int = 3,
    ):
        """
        One bbox per 3D grid cell in normalized coords [x1, y1, z1, x2, y2, z2].

        - scale: interpreted as normalized *volume* fraction wrt full volume.
        - ratio: aspect ratio range guiding width/height footprint; depth is
                 chosen to match volume.
        """
        K = grid_num * grid_num * grid_num
        bboxes = np.zeros((K, 6), dtype=np.float32)

        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for gid in range(K):
            # grid indices
            gx = gid % grid_num
            gy = (gid // grid_num) % grid_num
            gz = gid // (grid_num * grid_num)

            gw = 1.0 / grid_num
            gh = 1.0 / grid_num
            gd = 1.0 / grid_num

            # shrunken cell bounds
            gx1, gx2 = gx / grid_num + gw / 4.0, (gx + 1) / grid_num - gw / 4.0
            gy1, gy2 = gy / grid_num + gh / 4.0, (gy + 1) / grid_num - gh / 4.0
            gz1, gz2 = gz / grid_num + gd / 4.0, (gz + 1) / grid_num - gd / 4.0

            # sample target volume and in-plane aspect ratio
            target_vol = random.uniform(*scale)
            ar_hw = math.exp(random.uniform(*log_ratio))

            # derive footprint (w,h) and depth such that w*h*d ~ target_vol
            # in normalized coordinates (wrt full volume)
            # We treat target_vol as fraction of full volume; for the cell,
            # part of that will be taken, but we clamp later anyway.
            # Approximate local normalized dims:
            # base volume of full space is 1, so:
            #   w*h*d ≈ target_vol
            #   w/h = ar_hw
            # => w = sqrt(ar_hw * A), h = sqrt(A / ar_hw), d = target_vol / A
            # Choose A to be cubic root: A ≈ target_vol^(2/3)
            A = target_vol ** (2.0 / 3.0)
            w = math.sqrt(A * ar_hw)
            h = math.sqrt(A / max(1e-6, ar_hw))
            d = target_vol / max(1e-6, A)

            # center of bbox in this cell
            xc = random.uniform(gx1, gx2)
            yc = random.uniform(gy1, gy2)
            zc = random.uniform(gz1, gz2)

            x1 = max(0.0, xc - w / 2.0)
            y1 = max(0.0, yc - h / 2.0)
            z1 = max(0.0, zc - d / 2.0)
            x2 = min(1.0, xc + w / 2.0)
            y2 = min(1.0, yc + h / 2.0)
            z2 = min(1.0, zc + d / 2.0)

            bboxes[gid] = (x1, y1, z1, x2, y2, z2)

        return bboxes

    @staticmethod
    def _to_voxel_box(b, W: int, H: int, D: int):
        """
        Convert normalized bbox [x1,y1,z1,x2,y2,z2] to inclusive voxel coords
        for a (D, H, W) grid. We map:
          x -> width index (axis 2)
          y -> height index (axis 1)
          z -> depth index (axis 0)
        """
        x1n, y1n, z1n, x2n, y2n, z2n = b

        x1 = int(round(x1n * (W - 1)))
        x2 = int(round(x2n * (W - 1)))
        y1 = int(round(y1n * (H - 1)))
        y2 = int(round(y2n * (H - 1)))
        z1 = int(round(z1n * (D - 1)))
        z2 = int(round(z2n * (D - 1)))

        return (
            max(0, x1),
            max(0, y1),
            max(0, z1),
            min(W - 1, x2),
            min(H - 1, y2),
            min(D - 1, z2),
        )
        
    @staticmethod
    def complete_mask_randomly(mask: np.ndarray, num_masking_patches: int):
        """
        If RCC undershoots the target, fill the remainder by masking random
        unmasked voxels. Works for any mask shape (D,H,W).
        """
        shape = mask.shape
        flat = mask.reshape(-1)
        need = int(num_masking_patches) - int(flat.sum())
        if need > 0:
            idx = np.where(~flat)[0]
            need = min(need, len(idx))
            if need > 0:
                pick = np.random.choice(idx, size=need, replace=False)
                flat[pick] = True
        return flat.reshape(shape)
    
    @staticmethod
    def _trim_mask_randomly(mask: np.ndarray, num_masking_patches: int):
        """
        If RCC overshoots the target, unmask random masked voxels until
        exactly num_masking_patches are masked. Works for any (D,H,W).
        """
        shape = mask.shape
        flat = mask.reshape(-1)
        extra = int(flat.sum()) - int(num_masking_patches)
        if extra > 0:
            idx = np.where(flat)[0]  # masked positions
            extra = min(extra, len(idx))
            if extra > 0:
                drop = np.random.choice(idx, size=extra, replace=False)
                flat[drop] = False
        return flat.reshape(shape)
