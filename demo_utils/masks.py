"""Mask utilities for Demo 1."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_hw_numpy(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a 2D numpy image from ``[H, W]`` or single-channel ``[C, H, W]`` input."""

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image or single-channel image, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def make_foreground_mask(
    image: np.ndarray | torch.Tensor,
    *,
    min_object_frac: float = 0.01,
    closing_radius: int = 3,
    use_otsu: bool = True,
    manual_thresh: float | None = None,
    fill_internal_holes: bool = True,
    clear_border_first: bool = True,
) -> np.ndarray:
    """Build the notebook-style foreground mask for a prepared CT slice.

    The ground-truth notebook computes a body/subject mask from the displayed
    CT itself, fills interior holes, and keeps the largest cleaned component.
    This helper accepts HU, z-scored, or display-normalized slices; it
    min-maxes finite values before thresholding so the thresholds remain
    stable across those representations.
    """

    arr = _to_hw_numpy(image)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=bool)

    values = arr[finite]
    lo = float(values.min())
    hi = float(values.max())
    if hi > lo:
        img = (arr - lo) / (hi - lo)
    else:
        img = np.zeros_like(arr, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    height, width = img.shape

    if manual_thresh is not None:
        threshold = float(manual_thresh)
    elif use_otsu:
        try:
            from skimage.filters import threshold_otsu

            threshold = max(float(threshold_otsu(img[finite])), 0.03)
        except Exception:
            threshold = float(np.quantile(img[finite], 0.60))
    else:
        threshold = float(np.quantile(img[finite], 0.60))

    mask = img > threshold

    try:
        from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
        from skimage.segmentation import clear_border

        if clear_border_first:
            mask = clear_border(mask)
        if closing_radius > 0:
            mask = binary_closing(mask, footprint=disk(int(closing_radius)))
        if fill_internal_holes:
            try:
                from scipy.ndimage import binary_fill_holes

                mask = binary_fill_holes(mask)
            except Exception:
                mask = remove_small_holes(mask, area_threshold=int(height * width))
        mask = remove_small_holes(mask, area_threshold=max(1, int(0.002 * height * width)))
        mask = remove_small_objects(mask, min_size=max(64, int(min_object_frac * height * width)))
    except Exception:
        if fill_internal_holes:
            try:
                from scipy.ndimage import binary_fill_holes

                mask = binary_fill_holes(mask)
            except Exception:
                pass

    try:
        from skimage.measure import label

        labels = label(mask)
        if labels.max() > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == int(sizes.argmax())
    except Exception:
        pass

    return mask.astype(bool, copy=False)


def mask_to_token_grid(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Max-pool a resized image-space mask onto the patch-token grid."""

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {arr.shape}")
    height, width = arr.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"mask shape {arr.shape} must be divisible by patch_size={patch_size}")
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False))[None, None]
    pooled = F.max_pool2d(tensor, kernel_size=patch_size, stride=patch_size)
    return (pooled[0, 0].cpu().numpy() > 0.5).astype(bool, copy=False)
