"""CT and mask preprocessing for the 2D Demo 1 pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .masks import make_foreground_mask


WINDOWS: dict[str, tuple[float, float]] = {
    "lung": (-600.0, 1500.0),
    "soft_tissue": (50.0, 400.0),
    "bone": (400.0, 1800.0),
}


@dataclass(frozen=True)
class PadGeometry:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    pad_before: tuple[int, int]
    pad_after: tuple[int, int]


@dataclass(frozen=True)
class ZScoreStats:
    mean: float
    std: float
    clip_min: float
    clip_max: float
    pad_value: float


@dataclass(frozen=True)
class SceneArrays:
    model_image: np.ndarray
    display_image: np.ndarray
    foreground_mask: np.ndarray
    mask: np.ndarray | None
    metadata: dict[str, Any]


def center_pad_to_square(slice_2d: np.ndarray, fill_value: float = 0.0) -> tuple[np.ndarray, PadGeometry]:
    """Center-pad a 2D array to square without cropping."""

    arr = np.asarray(slice_2d)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {arr.shape}")
    height, width = arr.shape
    side = max(height, width)
    pad_y_total = side - height
    pad_x_total = side - width
    before_y = pad_y_total // 2
    before_x = pad_x_total // 2
    after_y = pad_y_total - before_y
    after_x = pad_x_total - before_x
    padded = np.pad(
        arr,
        ((before_y, after_y), (before_x, after_x)),
        mode="constant",
        constant_values=fill_value,
    )
    geom = PadGeometry(
        original_shape=(int(height), int(width)),
        padded_shape=(int(side), int(side)),
        pad_before=(int(before_y), int(before_x)),
        pad_after=(int(after_y), int(after_x)),
    )
    return padded.astype(arr.dtype, copy=False), geom


def resize_slice(slice_2d: np.ndarray, output_size: int = 512, *, is_mask: bool = False) -> np.ndarray:
    """Resize one 2D slice to ``output_size`` square using linear or nearest interpolation."""

    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    arr = np.asarray(slice_2d)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {arr.shape}")
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False))[None, None]
    if is_mask:
        resized = F.interpolate(tensor, size=(output_size, output_size), mode="nearest")
        return resized[0, 0].cpu().numpy().astype(arr.dtype, copy=False)
    resized = F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy().astype(np.float32, copy=False)


def resize_rgb(image: np.ndarray, output_size: int = 512, *, is_mask: bool = False) -> np.ndarray:
    """Resize ``[H, W, C]`` arrays channel-wise."""

    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected [H, W, C], got shape {arr.shape}")
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False)).permute(2, 0, 1)[None]
    if is_mask:
        resized = F.interpolate(tensor, size=(output_size, output_size), mode="nearest")
    else:
        resized = F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def window_ct_slice(slice_hu: np.ndarray, window_name: str = "lung") -> np.ndarray:
    """Apply a CT display window and return float values in ``[0, 1]``."""

    if window_name not in WINDOWS:
        known = ", ".join(sorted(WINDOWS))
        raise ValueError(f"Unknown CT window {window_name!r}. Available: {known}")
    center, width = WINDOWS[window_name]
    lower = center - width / 2.0
    upper = center + width / 2.0
    arr = np.asarray(slice_hu, dtype=np.float32)
    return np.clip((arr - lower) / (upper - lower), 0.0, 1.0).astype(np.float32, copy=False)


def zscore_clipped_volume(
    volume_hu: np.ndarray,
    clip_range: tuple[float, float] = (-1000.0, 1000.0),
    *,
    background_hu: float = -1000.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, ZScoreStats]:
    """Clip a HU volume and compute volume-level z-score normalization."""

    clip_min, clip_max = float(clip_range[0]), float(clip_range[1])
    clipped = np.clip(np.asarray(volume_hu, dtype=np.float32), clip_min, clip_max)
    mean = float(clipped.mean())
    std = float(clipped.std())
    if std < eps:
        std = 1.0
    z = (clipped - mean) / std
    pad_value = (float(background_hu) - mean) / std
    stats = ZScoreStats(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max, pad_value=float(pad_value))
    return z.astype(np.float32, copy=False), stats


def _import_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - exercised by runtime users.
        raise RuntimeError("SimpleITK is required for Demo 1 CT loading. Install requirements.txt.") from exc
    return sitk


def resample_sitk_image(
    image: Any,
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    *,
    is_mask: bool = False,
    default_value: float = -1000.0,
) -> Any:
    """Orient an image to LPS and resample it to ``target_spacing`` in SimpleITK XYZ order."""

    sitk = _import_sitk()
    oriented = sitk.DICOMOrient(image, "LPS")
    original_spacing = tuple(float(v) for v in oriented.GetSpacing())
    original_size = tuple(int(v) for v in oriented.GetSize())
    new_size = tuple(
        max(1, int(round(size * spacing / float(target))))
        for size, spacing, target in zip(original_size, original_spacing, target_spacing)
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(v) for v in target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(oriented.GetDirection())
    resampler.SetOutputOrigin(oriented.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(0.0 if is_mask else default_value))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resampler.Execute(oriented)


def load_resampled_lps_image(
    path: str | Path,
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    *,
    is_mask: bool = False,
    default_value: float = -1000.0,
) -> Any:
    """Load and resample a medical image object in LPS orientation."""

    sitk = _import_sitk()
    image = sitk.ReadImage(str(Path(path).expanduser()))
    return resample_sitk_image(
        image,
        target_spacing=target_spacing,
        is_mask=is_mask,
        default_value=default_value,
    )


def _sitk_array_and_metadata(image: Any, *, is_mask: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    sitk = _import_sitk()
    array = sitk.GetArrayFromImage(image)
    dtype = np.int32 if is_mask else np.float32
    metadata = {
        "spacing_xyz": [float(v) for v in image.GetSpacing()],
        "size_xyz": [int(v) for v in image.GetSize()],
        "array_shape_zyx": [int(v) for v in array.shape],
        "direction": [float(v) for v in image.GetDirection()],
        "origin_xyz": [float(v) for v in image.GetOrigin()],
    }
    return array.astype(dtype, copy=False), metadata


def load_resampled_lps_array(
    path: str | Path,
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    *,
    is_mask: bool = False,
    default_value: float = -1000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a NIfTI/medical image, orient to LPS, resample, and return ``[Z, Y, X]``."""

    resampled = load_resampled_lps_image(
        path,
        target_spacing=target_spacing,
        is_mask=is_mask,
        default_value=default_value,
    )
    return _sitk_array_and_metadata(resampled, is_mask=is_mask)


def resample_mask_to_reference(mask_path: str | Path, reference_image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a mask and nearest-neighbor resample it onto an already-resampled CT grid."""

    sitk = _import_sitk()
    mask_image = sitk.DICOMOrient(sitk.ReadImage(str(Path(mask_path).expanduser())), "LPS")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled = resampler.Execute(mask_image)
    return _sitk_array_and_metadata(resampled, is_mask=True)


def prepare_scene_arrays(
    *,
    ct_path: str | Path,
    slice_index: int,
    window_name: str,
    input_size: int = 512,
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    clip_range: tuple[float, float] = (-1000.0, 1000.0),
    background_hu: float = -1000.0,
    mask_path: str | Path | None = None,
) -> SceneArrays:
    """Create model, display, and optional mask arrays for one configured scene."""

    reference_image = load_resampled_lps_image(
        ct_path,
        target_spacing=target_spacing,
        is_mask=False,
        default_value=background_hu,
    )
    volume_hu, ct_metadata = _sitk_array_and_metadata(reference_image, is_mask=False)
    if slice_index < 0 or slice_index >= volume_hu.shape[0]:
        raise IndexError(f"slice_index={slice_index} outside resampled depth {volume_hu.shape[0]}")

    z_volume, z_stats = zscore_clipped_volume(volume_hu, clip_range=clip_range, background_hu=background_hu)
    model_slice = z_volume[slice_index]
    display_slice = window_ct_slice(volume_hu[slice_index], window_name=window_name)

    model_padded, pad_geom = center_pad_to_square(model_slice, fill_value=z_stats.pad_value)
    display_padded, _ = center_pad_to_square(display_slice, fill_value=0.0)
    model_image = resize_slice(model_padded, output_size=input_size, is_mask=False)
    display_image = resize_slice(display_padded, output_size=input_size, is_mask=False)
    foreground_mask = make_foreground_mask(model_image).astype(np.uint8, copy=False)

    mask_image: np.ndarray | None = None
    mask_metadata: dict[str, Any] | None = None
    if mask_path is not None:
        mask_volume, mask_metadata = resample_mask_to_reference(mask_path, reference_image)
        if mask_volume.shape != volume_hu.shape:
            raise ValueError(f"mask shape {mask_volume.shape} does not match CT shape {volume_hu.shape}")
        mask_slice = (mask_volume[slice_index] > 0).astype(np.uint8)
        mask_padded, _ = center_pad_to_square(mask_slice, fill_value=0)
        mask_image = (resize_slice(mask_padded, output_size=input_size, is_mask=True) > 0).astype(np.uint8)

    metadata = {
        "ct_path": str(ct_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "slice_index": int(slice_index),
        "window": window_name,
        "input_size": int(input_size),
        "target_spacing_xyz": [float(v) for v in target_spacing],
        "ct_metadata": ct_metadata,
        "mask_metadata": mask_metadata,
        "zscore": asdict(z_stats),
        "pad_geometry": asdict(pad_geom),
        "volume_shape_zyx": [int(v) for v in volume_hu.shape],
    }
    return SceneArrays(
        model_image=model_image.astype(np.float32, copy=False),
        display_image=np.clip(display_image, 0.0, 1.0).astype(np.float32, copy=False),
        foreground_mask=foreground_mask.astype(np.uint8, copy=False),
        mask=mask_image,
        metadata=metadata,
    )
