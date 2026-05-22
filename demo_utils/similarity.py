"""Query-patch similarity and top-k helpers for Demo 1."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .preprocessing import resize_slice


@dataclass(frozen=True)
class TopKMatch:
    rank: int
    row: int
    col: int
    score: float
    bbox_xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def click_xy_to_token(click_xy: tuple[int, int] | list[int], *, input_size: int, patch_size: int) -> tuple[int, int]:
    """Map image-space ``(x, y)`` coordinates to token-grid ``(row, col)``."""

    if input_size <= 0 or patch_size <= 0:
        raise ValueError("input_size and patch_size must be positive")
    if input_size % patch_size:
        raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch_size}")
    x, y = int(click_xy[0]), int(click_xy[1])
    max_coord = input_size - 1
    x = int(np.clip(x, 0, max_coord))
    y = int(np.clip(y, 0, max_coord))
    grid = input_size // patch_size
    row = min(grid - 1, y // patch_size)
    col = min(grid - 1, x // patch_size)
    return int(row), int(col)


def click_xy_to_patch_index(click_xy: tuple[int, int] | list[int], *, input_size: int, patch_size: int) -> int:
    """Map image-space ``(x, y)`` coordinates to a flattened patch index."""

    row, col = click_xy_to_token(click_xy, input_size=input_size, patch_size=patch_size)
    grid = input_size // patch_size
    return int(row * grid + col)


def cosine_similarity_grid(embeddings_grid: np.ndarray, query_token: tuple[int, int], eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity from one query token to all tokens in the same slice."""

    grid = np.asarray(embeddings_grid, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError(f"embeddings_grid must be [H, W, D], got {grid.shape}")
    row, col = query_token
    if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        raise IndexError(f"query_token={query_token} outside token grid {grid.shape[:2]}")
    flat = grid.reshape(-1, grid.shape[-1])
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normalized = flat / np.maximum(norms, eps)
    query = normalized[row * grid.shape[1] + col]
    return (normalized @ query).reshape(grid.shape[:2]).astype(np.float32, copy=False)


def exclude_query_neighborhood(
    similarity: np.ndarray,
    query_token: tuple[int, int],
    *,
    radius: int = 1,
) -> np.ndarray:
    scores = np.asarray(similarity, dtype=np.float32).copy()
    row, col = query_token
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    r0 = max(0, row - radius)
    r1 = min(scores.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(scores.shape[1], col + radius + 1)
    scores[r0:r1, c0:c1] = -np.inf
    return scores


def token_bbox(row: int, col: int, *, patch_size: int, input_size: int) -> tuple[int, int, int, int]:
    x0 = int(col * patch_size)
    y0 = int(row * patch_size)
    x1 = min(input_size, x0 + patch_size)
    y1 = min(input_size, y0 + patch_size)
    return x0, y0, x1, y1


def top_k_from_similarity(
    similarity: np.ndarray,
    *,
    query_token: tuple[int, int],
    patch_size: int,
    top_k: int = 6,
    exclude_radius: int = 1,
    input_size: int = 512,
) -> list[TopKMatch]:
    """Return top-k tokens after excluding the query neighborhood."""

    if top_k <= 0:
        return []
    scores = exclude_query_neighborhood(similarity, query_token, radius=exclude_radius)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return []
    flat_indices = np.flatnonzero(finite)
    order = np.argsort(scores.reshape(-1)[flat_indices])[::-1]
    selected = flat_indices[order[:top_k]]

    matches: list[TopKMatch] = []
    grid_w = scores.shape[1]
    for rank, flat_idx in enumerate(selected, start=1):
        row = int(flat_idx // grid_w)
        col = int(flat_idx % grid_w)
        matches.append(
            TopKMatch(
                rank=rank,
                row=row,
                col=col,
                score=float(scores[row, col]),
                bbox_xyxy=token_bbox(row, col, patch_size=patch_size, input_size=input_size),
            )
        )
    return matches


def upsample_similarity(similarity: np.ndarray, *, input_size: int = 512) -> np.ndarray:
    # Nearest-neighbor upsampling so the 32x32 token grid renders as crisp 8x8
    # blocks (matches demo/test.ipynb's `plt.imshow` on the raw token grid).
    return resize_slice(np.asarray(similarity, dtype=np.float32), output_size=input_size, is_mask=True)


def normalize_similarity_for_display(similarity_image: np.ndarray) -> np.ndarray:
    arr = np.asarray(similarity_image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def extract_context_crops(
    display_image: np.ndarray,
    matches: list[TopKMatch],
    *,
    context_pixels: int = 32,
    crop_size: int = 160,
) -> np.ndarray:
    """Extract fixed-size grayscale RGB crops around each match bbox."""

    image = np.asarray(display_image, dtype=np.float32)
    crops: list[np.ndarray] = []
    height, width = image.shape
    for match in matches:
        x0, y0, x1, y1 = match.bbox_xyxy
        x0 = max(0, x0 - context_pixels)
        y0 = max(0, y0 - context_pixels)
        x1 = min(width, x1 + context_pixels)
        y1 = min(height, y1 + context_pixels)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((1, 1), dtype=np.float32)
        resized = resize_slice(crop, output_size=crop_size, is_mask=False)
        crops.append(np.repeat(resized[..., None], 3, axis=-1))
    if not crops:
        return np.zeros((0, crop_size, crop_size, 3), dtype=np.float32)
    return np.stack(crops).astype(np.float32, copy=False)
