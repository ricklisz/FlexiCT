#!/usr/bin/env python3
"""ISUP-grade retrieval on C4KC-KiTS (ccRCC only) — two-pass adaptive crop + RRF.

Task
----
* Population : clear-cell RCC (ccRCC) patients with known ISUP grade.
* Label      : Low (ISUP 1–2) vs High (ISUP 3–4) — binary.
* Crops      : two per-patient adaptive cubic crops centred on the tumor.
               pass 1 crop_size = bbox_max_extent_mm × 1.5
               pass 2 crop_size = bbox_max_extent_mm × 2.0
               each crop is boundary-clamped, then resized to 160³ for the
               model.
* Ranking    : cosine similarity per crop scale + reciprocal rank fusion.
* Metrics    : Recall@K, Precision@K, mAP with 95 % bootstrap CIs vs
               random-retrieval baseline.

Adaptive + fusion rationale
---------------------------
The 1.5× crop focuses on core lesion morphology; the 2.0× crop includes more
peri-tumoral context. Reciprocal Rank Fusion combines both rankings to retain
retrievals that are consistently strong across scales.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from math import comb as _comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flexi_ct import Flexi_CT_3D, Flexi_CT_VLM
from flexi_ct.checkpoints import resolve_retrieval_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CSV_DIR = Path(__file__).resolve().parent / "csv"
CLINICAL_CSV = (
    os.environ.get(
        "C4KC_CLINICAL_CSV",
        str(CSV_DIR / "C4KC-KiTS_final.csv"),
    )
)
IMAGE_DIR = os.environ.get("C4KC_IMAGE_DIR", str(ROOT / "data" / "C4KC"))

ISUP_TO_GROUP: Dict[int, str] = {1: "Low", 2: "Low", 3: "High", 4: "High"}

TARGET_SIZE = (160, 160, 160)

# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────
def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# SimpleITK helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resample_sitk(
    image: sitk.Image,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [int(round(sz * ospc / nspc))
                for sz, ospc, nspc in zip(orig_size, orig_spacing, spacing)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(spacing)
    f.SetSize(new_size)
    f.SetOutputDirection(image.GetDirection())
    f.SetOutputOrigin(image.GetOrigin())
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(default_value)
    return f.Execute(image)


def _orient_sitk(image: sitk.Image, axcodes: str = "LPS") -> sitk.Image:
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation(axcodes)
    return f.Execute(image)


# ──────────────────────────────────────────────────────────────────────────────
# Seg-derived geometry: centroid (world) + bbox max extent at 2 mm iso
# ──────────────────────────────────────────────────────────────────────────────
def compute_seg_geometry(
    seg_path: str,
) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    """Return (centroid_world_LPS, bbox_max_extent_mm) for the Mass label (2).

    Both quantities are derived after resampling the seg to 2 mm isotropic, so
    bbox_max_extent_mm is the bounding-box size in 2 mm voxels and is directly
    usable as the crop-size basis for this retrieval script.

    Falls back to label 1 (Kidney) when no Mass voxels are present.
    Returns (None, None) if the file is missing or the mask is empty.
    """
    if not Path(seg_path).exists():
        return None, None

    try:
        seg = sitk.ReadImage(str(seg_path))
    except Exception:
        return None, None

    # Resample to 2 mm iso with nearest-neighbour to preserve labels
    seg_iso = _resample_sitk(seg, spacing=(2.0, 2.0, 2.0),
                              interpolator=sitk.sitkNearestNeighbor,
                              default_value=0.0)
    seg_iso = _orient_sitk(seg_iso, "LPS")

    arr = sitk.GetArrayFromImage(seg_iso)          # (Z, Y, X)
    coords = np.argwhere(arr == 2)                 # Mass
    if len(coords) == 0:
        coords = np.argwhere(arr == 1)             # Kidney fallback
    if len(coords) == 0:
        return None, None

    # Centroid in voxel space (z, y, x), convert to world (x, y, z)
    centroid_vox = coords.mean(axis=0)             # (z, y, x)
    world = seg_iso.TransformContinuousIndexToPhysicalPoint(
        (float(centroid_vox[2]), float(centroid_vox[1]), float(centroid_vox[0]))
    )
    centroid_world = (float(world[0]), float(world[1]), float(world[2]))

    # Bounding-box max extent in 2 mm voxels
    bbox_extents = (coords.max(axis=0) - coords.min(axis=0) + 1).astype(float)
    bbox_max_mm = float(bbox_extents.max())

    return centroid_world, bbox_max_mm


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive cubic crop
# ──────────────────────────────────────────────────────────────────────────────
def adaptive_crop(
    ct_path: str,
    centroid_world: Tuple[float, float, float],
    crop_size: int,
) -> torch.Tensor:
    """Load CT → resample to 2 mm iso → orient LPS → clip/z-score →
    crop a cube of *crop_size* voxels around *centroid_world* →
    boundary-clamp if needed → resize to TARGET_SIZE via trilinear
    interpolation.

    Returns a (1, D, H, W) float32 tensor ready for the model.
    """
    # ── Load and preprocess ──────────────────────────────────────────────────
    ct = sitk.ReadImage(ct_path)
    ct = _resample_sitk(ct, spacing=(2.0, 2.0, 2.0),
                        interpolator=sitk.sitkLinear, default_value=-1000.0)
    ct = _orient_sitk(ct, "LPS")

    arr = sitk.GetArrayFromImage(ct).astype(np.float32)   # (Z, Y, X)
    arr = np.clip(arr, -1000.0, 1000.0)
    std = arr.std()
    arr = (arr - arr.mean()) / (std if std > 1e-6 else 1.0)

    # ── Map centroid world → voxel index in resampled LPS volume ─────────────
    # TransformPhysicalPointToContinuousIndex returns (x, y, z)
    cx, cy, cz = ct.TransformPhysicalPointToContinuousIndex(centroid_world)
    # numpy layout is (Z, Y, X)
    iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))

    # ── Crop with boundary clamping ───────────────────────────────────────────
    half = crop_size // 2
    Z, Y, X = arr.shape

    z0 = max(0, iz - half);  z1 = min(Z, iz - half + crop_size)
    y0 = max(0, iy - half);  y1 = min(Y, iy - half + crop_size)
    x0 = max(0, ix - half);  x1 = min(X, ix - half + crop_size)

    crop = arr[z0:z1, y0:y1, x0:x1]

    # ── Boundary-clamped crop; resize below restores fixed model shape ────────
    t = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)   # (1,1,Z,Y,X)
    t = F.interpolate(t, size=TARGET_SIZE, mode="trilinear", align_corners=True)
    return t.squeeze(0)                                             # (1, D, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class C4KCAdaptiveDataset(Dataset):
    """C4KC-KiTS ccRCC dataset with per-patient adaptive tumor crops.

    Crop size is determined at init time from each patient's segmentation mask:
        crop_size_vox = round(bbox_max_extent_mm × crop_scale)
    The crop is centred on the Mass (label 2) centroid in world coordinates.
    """

    def __init__(
        self,
        clinical_csv: str = CLINICAL_CSV,
        image_dir: str = IMAGE_DIR,
        crop_scale: float = 1.5,
    ):
        df = pd.read_csv(clinical_csv)

        # ── Filter: ccRCC only, valid ISUP grade ─────────────────────────────
        df = df[df["tumor_histologic_subtype"] == "clear_cell_rcc"].copy()
        df = df[df["tumor_isup_grade"].notna()].copy()
        df["isup_grade"] = df["tumor_isup_grade"].astype(int)
        df = df[df["isup_grade"].isin(ISUP_TO_GROUP)].copy()
        df["isup_group"] = df["isup_grade"].map(ISUP_TO_GROUP)
        df = df.reset_index(drop=True)

        # ── Resolve file paths ────────────────────────────────────────────────
        df["ct_path"] = df["patient_id"].apply(
            lambda pid: str(Path(image_dir) / f"{pid}_ct.nii.gz")
        )
        df["seg_path"] = df["patient_id"].apply(
            lambda pid: str(Path(image_dir) / f"{pid}_seg.nii.gz")
        )

        exists = df["ct_path"].apply(lambda p: Path(p).exists())
        if (~exists).sum() > 0:
            missing = df.loc[~exists, "patient_id"].tolist()
            print(f"[Dataset] Dropping {(~exists).sum()} missing CT patients: {missing}")
            df = df[exists].reset_index(drop=True)

        # ── Precompute centroid + adaptive crop size ──────────────────────────
        print("[Dataset] Computing tumor centroids and bounding boxes …")
        cx_list, cy_list, cz_list, crop_list = [], [], [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Seg geometry"):
            centroid, bbox_mm = compute_seg_geometry(row["seg_path"])
            if centroid is None:
                cx_list.append(float("nan"))
                cy_list.append(float("nan"))
                cz_list.append(float("nan"))
                crop_list.append(float("nan"))
            else:
                cx_list.append(centroid[0])
                cy_list.append(centroid[1])
                cz_list.append(centroid[2])
                crop_list.append(bbox_mm * crop_scale)

        df["centroid_X"] = cx_list
        df["centroid_Y"] = cy_list
        df["centroid_Z"] = cz_list
        df["adaptive_crop_size"] = crop_list

        # Drop rows with no valid geometry
        valid = df["adaptive_crop_size"].notna()
        if (~valid).sum() > 0:
            print(f"[Dataset] Dropping {(~valid).sum()} patients with no valid seg geometry.")
            df = df[valid].reset_index(drop=True)

        self.df = df
        self.crop_scale = crop_scale

        if is_main_process():
            cs = df["adaptive_crop_size"]
            print(f"[Dataset] {len(df)} ccRCC patients  |  ISUP: {Counter(df['isup_grade'].tolist())}")
            print(f"[Dataset] Adaptive crop size (vox): "
                  f"min={cs.min():.1f}  median={cs.median():.1f}  max={cs.max():.1f}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        crop_size = max(1, int(round(float(row["adaptive_crop_size"]))))
        centroid_world = (
            float(row["centroid_X"]),
            float(row["centroid_Y"]),
            float(row["centroid_Z"]),
        )
        image = adaptive_crop(row["ct_path"], centroid_world, crop_size)
        return {
            "image":      image,
            "patient_id": row["patient_id"],
            "isup_grade": int(row["isup_grade"]),
            "isup_group": row["isup_group"],
            "crop_size":  crop_size,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Vision model (identical to rank_fusion.py)
# ──────────────────────────────────────────────────────────────────────────────
class VisionEmbeddingExtractor(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        projection: nn.Module,
        use_hier: bool = False,
        interaction_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.projection = projection
        self.use_hier = use_hier
        self.interaction_indices = interaction_indices or []

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_hier and self.interaction_indices:
            layer_outputs = self.vision_model.get_intermediate_layers(
                x, n=self.interaction_indices,
                return_class_token=True, norm=True,
            )
            h_parts = [
                torch.cat([cls, patches.mean(dim=1)], dim=-1)
                for patches, cls in layer_outputs
            ]
            return F.normalize(torch.cat(h_parts, dim=-1), dim=-1)

        out = self.vision_model(x, is_training=True)
        h = torch.cat([out["x_norm_clstoken"],
                       out["x_norm_patchtokens"].mean(dim=1)], dim=-1)
        return F.normalize(self.projection(h), dim=-1)


def load_vision_model(
    ckpt_path: str | None,
    device: str = "cuda",
    ckpt_type: str = "vlm",
    use_hier: bool = False,
    interaction_indices: Optional[List[int]] = None,
) -> VisionEmbeddingExtractor:
    ckpt_path = resolve_retrieval_checkpoint(ckpt_type, ckpt_path)
    if ckpt_type == "vlm":
        vlm = Flexi_CT_VLM(checkpoint_path=ckpt_path, device="cpu")
        vision_model = vlm.model.vision_model
        projection = vlm.model.vlm_vision_projection
    elif ckpt_type == "teacher":
        teacher = Flexi_CT_3D(checkpoint_path=ckpt_path, device="cpu")
        vision_model = teacher.backbone
        projection = nn.Identity()
    else:
        raise ValueError(f"Unknown ckpt_type='{ckpt_type}'")

    model = VisionEmbeddingExtractor(vision_model, projection,
                                     use_hier=use_hier,
                                     interaction_indices=interaction_indices)
    model.to(device).eval()
    mode_str = f"hier blocks {interaction_indices}" if use_hier else ckpt_type
    print(f"Loaded local FlexiCT-backed vision model from {ckpt_path} (mode={mode_str})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Stratified 50/50 split (by individual ISUP grade for within-class balance)
# ──────────────────────────────────────────────────────────────────────────────
def stratified_split(grades: List[int], seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    grade_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(grades):
        grade_to_idx[g].append(i)
    query_idx, support_idx = [], []
    for grade in sorted(grade_to_idx):
        idxs = np.array(grade_to_idx[grade])
        rng.shuffle(idxs)
        n_q = len(idxs) // 2
        query_idx.extend(idxs[:n_q].tolist())
        support_idx.extend(idxs[n_q:].tolist())
    return query_idx, support_idx


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
class IndexedDataset(Dataset):
    def __init__(self, base: C4KCAdaptiveDataset, indices: List[int]):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.base[self.indices[idx]]
        return s["image"], s["patient_id"]


def _collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def extract_features(
    model: VisionEmbeddingExtractor,
    dataset: C4KCAdaptiveDataset,
    indices: List[int],
    device: str = "cuda",
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, List[str]]:
    sub = IndexedDataset(dataset, indices)
    world_size, rank = get_world_size(), get_rank()
    sampler = (
        torch.utils.data.DistributedSampler(sub, num_replicas=world_size,
                                             rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    loader = DataLoader(sub, batch_size=batch_size, sampler=sampler,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True, collate_fn=_collate,
                        persistent_workers=num_workers > 0, drop_last=False)

    local_feats, local_pids = [], []
    for images, pids in tqdm(loader, desc="Extracting", disable=rank != 0):
        images = images.to(device, non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            feat = model(images)
        local_feats.append(feat.float().cpu())
        local_pids.extend(pids)

    local_feats = torch.cat(local_feats, dim=0)

    if world_size > 1:
        local_feats = local_feats.to(device)
        gathered = [torch.zeros_like(local_feats) for _ in range(world_size)]
        dist.all_gather(gathered, local_feats)
        all_feats = torch.cat(gathered, dim=0).cpu()
        all_pids_nested: List[List[str]] = [None] * world_size  # type: ignore
        dist.all_gather_object(all_pids_nested, local_pids)
        all_pids = [p for sub_ in all_pids_nested for p in sub_]
        n = len(indices)
        return all_feats[:n], all_pids[:n]

    return local_feats, local_pids


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(sim_matrices: List[torch.Tensor], rrf_k: int = 60) -> np.ndarray:
    """Fuse multiple [Q, S] similarity matrices using Reciprocal Rank Fusion."""
    q, s = sim_matrices[0].shape
    fused = np.zeros((q, s), dtype=np.float64)
    arange_s = torch.arange(s).unsqueeze(0).expand(q, -1)

    for sim in sim_matrices:
        sorted_idx = sim.argsort(dim=1, descending=True)
        ranks = torch.zeros_like(sorted_idx)
        ranks.scatter_(1, sorted_idx, arange_s)
        fused += 1.0 / (rrf_k + ranks.numpy())

    return fused


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────────────────────
def _sorted_idx(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1)


def compute_recall_at_k(scores, q_groups, s_groups, k):
    eff_k = min(k, scores.shape[1])
    topk = np.argpartition(scores, -eff_k, axis=1)[:, -eff_k:]
    s_arr = np.array(s_groups)
    return np.array([np.any(s_arr[topk[i]] == qg) for i, qg in enumerate(q_groups)])


def compute_per_group_recall(hits, q_groups):
    arr = np.array(q_groups)
    return {g: float(hits[arr == g].mean()) for g in sorted(set(q_groups))
            if (arr == g).any()}


def compute_precision_at_k(scores, q_groups, s_groups, k):
    eff_k = min(k, scores.shape[1])
    top = _sorted_idx(scores)[:, :eff_k]
    s_arr = np.array(s_groups)
    return np.array([(s_arr[top[i]] == qg).sum() / eff_k
                     for i, qg in enumerate(q_groups)])


def compute_map(scores, q_groups, s_groups):
    ranked_all = _sorted_idx(scores)
    s_arr = np.array(s_groups)
    aps = np.zeros(len(q_groups))
    for i, qg in enumerate(q_groups):
        rel = (s_arr[ranked_all[i]] == qg)
        n_rel = rel.sum()
        if n_rel == 0:
            continue
        cum = np.cumsum(rel).astype(np.float64)
        p_at_r = cum / np.arange(1, len(rel) + 1)
        aps[i] = (p_at_r * rel).sum() / n_rel
    return aps


def _random_recall_at_k(S, R, K):
    K = min(K, S)
    if R <= 0 or S <= 0:
        return 0.0
    if K >= S or R >= S or S - R < K:
        return 1.0
    return 1.0 - _comb(S - R, K) / _comb(S, K)


def random_baseline_recall(q_groups, s_groups, k):
    S = len(s_groups)
    s_counts = Counter(s_groups)
    per: Dict[str, List[float]] = defaultdict(list)
    all_v = []
    for qg in q_groups:
        p = _random_recall_at_k(S, s_counts.get(qg, 0), k)
        all_v.append(p); per[qg].append(p)
    return {"overall": float(np.mean(all_v)),
            **{g: float(np.mean(vs)) for g, vs in per.items()}}


def random_baseline_precision(q_groups, s_groups, k):
    S = len(s_groups)
    s_counts = Counter(s_groups)
    per: Dict[str, List[float]] = defaultdict(list)
    all_v = []
    for qg in q_groups:
        p = s_counts.get(qg, 0) / S
        all_v.append(p); per[qg].append(p)
    return {"overall": float(np.mean(all_v)),
            **{g: float(np.mean(vs)) for g, vs in per.items()}}


def random_baseline_map(q_groups, s_groups):
    S = len(s_groups)
    s_counts = Counter(s_groups)
    per: Dict[str, List[float]] = defaultdict(list)
    all_v = []
    for qg in q_groups:
        p = s_counts.get(qg, 0) / S
        all_v.append(p); per[qg].append(p)
    return {"overall": float(np.mean(all_v)),
            **{g: float(np.mean(vs)) for g, vs in per.items()}}


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean()
                     for _ in range(n_bootstrap)])
    alpha = (1.0 - ci) / 2.0
    return {"mean": float(values.mean()),
            "ci_lower": float(np.quantile(boot, alpha)),
            "ci_upper": float(np.quantile(boot, 1.0 - alpha))}


# ══════════════════════════════════════════════════════════════════════════════
# Main retrieval pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_retrieval(args: argparse.Namespace) -> Dict:
    setup_distributed()
    rank = get_rank()
    device = (f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
              if torch.cuda.is_available() else "cpu")
    args.checkpoint = resolve_retrieval_checkpoint(args.ckpt_type, args.checkpoint)

    # ── 1. Datasets (two adaptive scales) ────────────────────────────────────
    if is_main_process():
        print(f"Building dataset (crop_scale={args.crop_scale_small}) …")
    dataset_small = C4KCAdaptiveDataset(
        clinical_csv=args.clinical_csv,
        image_dir=args.image_dir,
        crop_scale=args.crop_scale_small,
    )
    if is_main_process():
        print(f"Building dataset (crop_scale={args.crop_scale_large}) …")
    dataset_large = C4KCAdaptiveDataset(
        clinical_csv=args.clinical_csv,
        image_dir=args.image_dir,
        crop_scale=args.crop_scale_large,
    )

    # ── 2. Labels and split ───────────────────────────────────────────────────
    if not dataset_small.df["patient_id"].equals(dataset_large.df["patient_id"]):
        raise RuntimeError("Scale-specific datasets are misaligned by patient_id.")

    grades_all = dataset_small.df["isup_grade"].tolist()
    groups_all = dataset_small.df["isup_group"].tolist()

    q_local, s_local = stratified_split(grades_all, seed=args.seed)

    query_grades   = [grades_all[i] for i in q_local]
    support_grades = [grades_all[i] for i in s_local]
    query_groups   = [groups_all[i] for i in q_local]
    support_groups = [groups_all[i] for i in s_local]

    if is_main_process():
        print(f"\nQuery   set: {len(q_local)}  grades={Counter(query_grades)}"
              f"  groups={Counter(query_groups)}")
        print(f"Support set: {len(s_local)}  grades={Counter(support_grades)}"
              f"  groups={Counter(support_groups)}")
        # Crop size stats per group
        for scale_name, scale_ds in [
            (f"{args.crop_scale_small:.1f}x", dataset_small),
            (f"{args.crop_scale_large:.1f}x", dataset_large),
        ]:
            print(f"  Scale {scale_name}:")
            df = scale_ds.df
            for grp in ["Low", "High"]:
                cs = df.loc[df["isup_group"] == grp, "adaptive_crop_size"]
                print(f"    {grp} crop_size: min={cs.min():.1f}  "
                      f"median={cs.median():.1f}  max={cs.max():.1f}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    if is_main_process():
        print("\nLoading model …")
    model = load_vision_model(
        args.checkpoint, device=device, ckpt_type=args.ckpt_type,
        use_hier=args.use_hier, interaction_indices=args.interaction_indices,
    )

    # ── 4. Feature extraction (two adaptive-crop passes) ─────────────────────
    if is_main_process():
        print(f"\n── Pass 1: crop scale {args.crop_scale_small} ──")
        print("Extracting query features …")
    q_feats_small, q_pids = extract_features(
        model, dataset_small, q_local, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if is_main_process():
        print("Extracting support features …")
    s_feats_small, s_pids = extract_features(
        model, dataset_small, s_local, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    if is_main_process():
        print(f"\n── Pass 2: crop scale {args.crop_scale_large} ──")
        print("Extracting query features …")
    q_feats_large, _ = extract_features(
        model, dataset_large, q_local, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if is_main_process():
        print("Extracting support features …")
    s_feats_large, _ = extract_features(
        model, dataset_large, s_local, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ── 5. Cosine similarity + rank fusion ───────────────────────────────────
    sim_small = q_feats_small @ s_feats_small.T   # [Q, S]
    sim_large = q_feats_large @ s_feats_large.T   # [Q, S]
    scores = reciprocal_rank_fusion([sim_small, sim_large], rrf_k=args.rrf_k)

    if is_main_process():
        print(f"\nRRF-fused score matrix: {scores.shape} (rrf_k={args.rrf_k})")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    results: Dict = {}
    if is_main_process():
        k_values = [1, 3, 5]
        groups_arr = np.array(query_groups)

        results = {
            "config": {
                "checkpoint": args.checkpoint,
                "ckpt_type": args.ckpt_type,
                "use_hier": args.use_hier,
                "interaction_indices": args.interaction_indices,
                "crop_scale_small": args.crop_scale_small,
                "crop_scale_large": args.crop_scale_large,
                "rrf_k": args.rrf_k,
                "pixdim": 1.0,
                "seed": args.seed,
                "label": "ISUP_grade_binary (Low=1-2, High=3-4)",
                "population": "ccRCC only",
                "crop_strategy": "two-pass adaptive (bbox_max_extent × scale) + RRF",
            },
            "split": {
                "n_query": len(q_local),
                "n_support": len(s_local),
                "query_grade_dist":   dict(Counter(query_grades)),
                "support_grade_dist": dict(Counter(support_grades)),
                "query_group_dist":   dict(Counter(query_groups)),
                "support_group_dist": dict(Counter(support_groups)),
            },
            "metrics": {},
            "random_baseline": {},
            "per_query": [],
        }

        # Recall@K
        print("\n── Recall@K ──")
        for k in k_values:
            eff_k = min(k, len(s_local))
            hits = compute_recall_at_k(scores, query_groups, support_groups, eff_k)
            stats = bootstrap_ci(hits.astype(float), n_bootstrap=args.n_bootstrap,
                                 seed=args.seed)
            per_group = compute_per_group_recall(hits, query_groups)
            rand = random_baseline_recall(query_groups, support_groups, eff_k)
            results["metrics"][f"recall@{k}"] = {"overall": stats, "per_group": per_group}
            results["random_baseline"][f"recall@{k}"] = rand
            print(f"  Recall@{k:>2d}: {stats['mean']:.4f}  "
                  f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]  "
                  f"(random: {rand['overall']:.4f})")
            for g in sorted(per_group):
                print(f"    {g:>6s}: {per_group[g]:.4f}  (random: {rand.get(g,0.):.4f})")

        # mAP
        print("\n── Mean Average Precision ──")
        ap_vals = compute_map(scores, query_groups, support_groups)
        map_stats = bootstrap_ci(ap_vals, n_bootstrap=args.n_bootstrap, seed=args.seed)
        map_per_group = {
            g: bootstrap_ci(ap_vals[groups_arr == g], n_bootstrap=args.n_bootstrap,
                            seed=args.seed)
            for g in sorted(set(query_groups)) if (groups_arr == g).any()
        }
        rand_map = random_baseline_map(query_groups, support_groups)
        results["metrics"]["mAP"] = {"overall": map_stats, "per_group": map_per_group}
        results["random_baseline"]["mAP"] = rand_map
        print(f"  mAP:     {map_stats['mean']:.4f}  "
              f"[{map_stats['ci_lower']:.4f}, {map_stats['ci_upper']:.4f}]  "
              f"(random: {rand_map['overall']:.4f})")
        for g in sorted(map_per_group):
            gs = map_per_group[g]
            print(f"    {g:>6s}: {gs['mean']:.4f}  "
                  f"[{gs['ci_lower']:.4f}, {gs['ci_upper']:.4f}]  "
                  f"(random: {rand_map.get(g,0.):.4f})")

        # Precision@K
        print("\n── Precision@K ──")
        prec_store: Dict[str, np.ndarray] = {}
        for k in k_values:
            eff_k = min(k, len(s_local))
            prec = compute_precision_at_k(scores, query_groups, support_groups, eff_k)
            prec_store[f"prec@{k}"] = prec
            stats = bootstrap_ci(prec, n_bootstrap=args.n_bootstrap, seed=args.seed)
            per_group_prec = {
                g: float(prec[groups_arr == g].mean())
                for g in sorted(set(query_groups)) if (groups_arr == g).any()
            }
            rand_prec = random_baseline_precision(query_groups, support_groups, eff_k)
            results["metrics"][f"precision@{k}"] = {"overall": stats,
                                                     "per_group": per_group_prec}
            results["random_baseline"][f"precision@{k}"] = rand_prec
            print(f"  Prec@{k:>2d}:  {stats['mean']:.4f}  "
                  f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]  "
                  f"(random: {rand_prec['overall']:.4f})")
            for g in sorted(per_group_prec):
                print(f"    {g:>6s}: {per_group_prec[g]:.4f}  "
                      f"(random: {rand_prec.get(g,0.):.4f})")

        # Per-query detail
        ranked_all   = _sorted_idx(scores)
        s_groups_arr = np.array(support_groups)
        for i in range(len(q_local)):
            entry = {
                "patient_id":  q_pids[i],
                "isup_grade":  query_grades[i],
                "isup_group":  query_groups[i],
                "crop_size_scale_small": int(round(float(
                    dataset_small.df.iloc[q_local[i]]["adaptive_crop_size"]
                ))),
                "crop_size_scale_large": int(round(float(
                    dataset_large.df.iloc[q_local[i]]["adaptive_crop_size"]
                ))),
                "AP":          float(ap_vals[i]),
                "top_retrieved_pids":   [s_pids[j] for j in ranked_all[i, :10]],
                "top_retrieved_groups": s_groups_arr[ranked_all[i, :10]].tolist(),
            }
            for k in k_values:
                eff_k = min(k, len(s_local))
                entry[f"hit@{k}"]  = bool(query_groups[i] in
                                          s_groups_arr[ranked_all[i, :eff_k]].tolist())
                entry[f"prec@{k}"] = float(prec_store[f"prec@{k}"][i])
            results["per_query"].append(entry)

        # Save
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "retrieval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_path}")

    cleanup_distributed()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C4KC-KiTS ISUP-grade retrieval (ccRCC) — two-pass adaptive crop + RRF"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help=("Explicit model checkpoint. If omitted, uses FLEXICT_CHECKPOINT, "
                         "then FLEXICT_VLM_CHECKPOINT for --ckpt_type vlm or "
                         "FLEXICT_3D_CHECKPOINT for --ckpt_type teacher."))
    p.add_argument("--ckpt_type", type=str, default="vlm", choices=["vlm", "teacher"])
    p.add_argument("--use_hier", action="store_true", default=False)
    p.add_argument("--interaction_indices", type=int, nargs="+", default=[3, 7, 11, 15])
    p.add_argument("--clinical_csv", type=str, default=CLINICAL_CSV)
    p.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    p.add_argument("--output_dir", type=str,
                   default=str(ROOT / "results/retrieval/C4KC_ISUP_grade/ours_adaptive_crop_rrf"))
    p.add_argument("--crop_scale_small", type=float, default=1.5,
                   help="Small-pass multiplier for adaptive crop size (default: 1.5).")
    p.add_argument("--crop_scale_large", type=float, default=2.0,
                   help="Large-pass multiplier for adaptive crop size (default: 2.0).")
    p.add_argument("--rrf_k", type=int, default=20,
                   help="Reciprocal Rank Fusion k constant (default: 20).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    run_retrieval(parse_args())
