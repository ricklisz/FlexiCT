#!/usr/bin/env python3
"""Linear probing for ISUP-grade classification on C4KC-KiTS (ccRCC only).

Preprocessing
-------------
Identical to ours_C4KC_retrieval_v2.py:
  * Population  : clear-cell RCC (ccRCC) with known ISUP grade.
  * Crop centroid: Mass (label 2) centroid from the seg mask at 1 mm iso;
                   falls back to Kidney (label 1) when no Mass voxels present.
  * Crop size   : bbox_max_extent_mm × crop_scale — no fixed bounds.
  * Intensity   : clip to [−1000, 1000] → z-score whole volume →
                  crop adaptive cube → zero-pad to cube → resize to 160³.
  * Padding fill: 0.0 (= mean value of the z-scored volume).

Label
-----
  Binary: Low (ISUP 1–2) vs High (ISUP 3–4).

Evaluation
----------
  Identical to ours_linear_adaptive.py:
  * Stratified k-fold CV by individual ISUP grade (1/2/3/4).
  * Grid search over C values; best C selected by balanced accuracy.
  * Repeated CV for 95 % CIs.
  * Metrics: balanced accuracy, macro F1, AUC (OvR), PR-AUC.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flexi_ct import Flexi_CT_3D, Flexi_CT_VLM
from flexi_ct.checkpoints import resolve_retrieval_checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# Paths & constants
# ══════════════════════════════════════════════════════════════════════════════
CSV_DIR = Path(__file__).resolve().parent / "csv"
CLINICAL_CSV = (
    os.environ.get(
        "C4KC_CLINICAL_CSV",
        str(CSV_DIR / "C4KC-KiTS_final.csv"),
    )
)
IMAGE_DIR = os.environ.get("C4KC_IMAGE_DIR", str(ROOT / "data" / "C4KC"))

ISUP_TO_GROUP: Dict[int, str] = {1: "Low", 2: "Low", 3: "High", 4: "High"}
GROUP_TO_IDX:  Dict[str, int] = {"Low": 0, "High": 1}
IDX_TO_GROUP:  Dict[int, str] = {v: k for k, v in GROUP_TO_IDX.items()}
NUM_CLASSES = len(GROUP_TO_IDX)

TARGET_SIZE = (160, 160, 160)


# ══════════════════════════════════════════════════════════════════════════════
# Distributed helpers
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# SimpleITK helpers
# ══════════════════════════════════════════════════════════════════════════════
def _resample_sitk(
    image: sitk.Image,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()
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


# ══════════════════════════════════════════════════════════════════════════════
# Seg geometry: centroid (world, LPS) + bbox max extent (mm at 1 mm iso)
# ══════════════════════════════════════════════════════════════════════════════
def compute_seg_geometry(
    seg_path: str,
) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    """Return (centroid_world_LPS, bbox_max_extent_mm) for the Mass label (2).

    Both quantities are derived after resampling the seg to 1 mm isotropic so
    that bbox_max_extent_mm equals the bounding-box size in voxels at 1 mm iso,
    making it directly usable as a crop size in voxels.

    Falls back to label 1 (Kidney) when no Mass voxels are present.
    Returns (None, None) if the file is missing or the mask is empty.
    """
    if not Path(seg_path).exists():
        return None, None
    try:
        seg = sitk.ReadImage(str(seg_path))
    except Exception:
        return None, None

    seg_iso = _resample_sitk(seg, spacing=(1.0, 1.0, 1.0),
                              interpolator=sitk.sitkNearestNeighbor,
                              default_value=0.0)
    seg_iso = _orient_sitk(seg_iso, "LPS")

    arr    = sitk.GetArrayFromImage(seg_iso)   # (Z, Y, X)
    coords = np.argwhere(arr == 2)             # Mass
    if len(coords) == 0:
        coords = np.argwhere(arr == 1)         # Kidney fallback
    if len(coords) == 0:
        return None, None

    centroid_vox = coords.mean(axis=0)         # (z, y, x)
    world = seg_iso.TransformContinuousIndexToPhysicalPoint(
        (float(centroid_vox[2]), float(centroid_vox[1]), float(centroid_vox[0]))
    )
    centroid_world = (float(world[0]), float(world[1]), float(world[2]))

    bbox_extents = (coords.max(axis=0) - coords.min(axis=0) + 1).astype(float)
    return centroid_world, float(bbox_extents.max())


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive cubic crop  (identical to ours_C4KC_retrieval_v2.py)
# ══════════════════════════════════════════════════════════════════════════════
def adaptive_crop(
    ct_path: str,
    centroid_world: Tuple[float, float, float],
    crop_size: int,
) -> torch.Tensor:
    """Load CT → resample 1 mm iso → orient LPS → clip → z-score whole volume
    → crop adaptive cube → zero-pad to cube → resize to TARGET_SIZE.

    Padding fill = 0.0 (= mean of the z-scored volume).
    Returns a (1, D, H, W) float32 tensor.
    """
    ct = sitk.ReadImage(ct_path)
    ct = _resample_sitk(ct, spacing=(1.0, 1.0, 1.0),
                        interpolator=sitk.sitkLinear, default_value=-1000.0)
    ct = _orient_sitk(ct, "LPS")

    arr = sitk.GetArrayFromImage(ct).astype(np.float32)   # (Z, Y, X)
    arr = np.clip(arr, -1000.0, 1000.0)

    # Z-score the whole volume
    std = arr.std()
    arr = (arr - arr.mean()) / (std if std > 1e-6 else 1.0)

    # World → voxel
    cx, cy, cz = ct.TransformPhysicalPointToContinuousIndex(centroid_world)
    iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))

    half = crop_size // 2
    Z, Y, X = arr.shape
    z0 = max(0, iz - half);  z1 = min(Z, iz - half + crop_size)
    y0 = max(0, iy - half);  y1 = min(Y, iy - half + crop_size)
    x0 = max(0, ix - half);  x1 = min(X, ix - half + crop_size)

    crop = arr[z0:z1, y0:y1, x0:x1]

    # Zero-pad to restore (crop_size, crop_size, crop_size); 0.0 == z-score mean
    pad = (
        (0, crop_size - crop.shape[0]),
        (0, crop_size - crop.shape[1]),
        (0, crop_size - crop.shape[2]),
    )
    if any(p[1] > 0 for p in pad):
        crop = np.pad(crop, pad, mode="constant", constant_values=0.0)

    t = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)   # (1,1,Z,Y,X)
    t = F.interpolate(t, size=TARGET_SIZE, mode="trilinear", align_corners=True)
    return t.squeeze(0)                                             # (1, D, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class C4KCAdaptiveDataset(Dataset):
    """C4KC-KiTS ccRCC dataset with per-patient adaptive tumor crops.

    Crop centroid : Mass (label 2) centroid from the seg mask at 1 mm iso,
                    falling back to Kidney (label 1).
    Crop size     : bbox_max_extent_mm × crop_scale voxels.
    Patients with no valid seg geometry are dropped.
    """

    def __init__(
        self,
        clinical_csv: str   = CLINICAL_CSV,
        image_dir:    str   = IMAGE_DIR,
        crop_scale:   float = 1.5,
    ):
        df = pd.read_csv(clinical_csv)

        # Filter: ccRCC only, valid ISUP grade
        df = df[df["tumor_histologic_subtype"] == "clear_cell_rcc"].copy()
        df = df[df["tumor_isup_grade"].notna()].copy()
        df["isup_grade"] = df["tumor_isup_grade"].astype(int)
        df = df[df["isup_grade"].isin(ISUP_TO_GROUP)].copy()
        df["isup_group"] = df["isup_grade"].map(ISUP_TO_GROUP)
        df = df.reset_index(drop=True)

        # Resolve file paths
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

        # Precompute centroid + adaptive crop size
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

        df["centroid_X"]        = cx_list
        df["centroid_Y"]        = cy_list
        df["centroid_Z"]        = cz_list
        df["adaptive_crop_size"] = crop_list

        valid = df["adaptive_crop_size"].notna()
        if (~valid).sum() > 0:
            print(f"[Dataset] Dropping {(~valid).sum()} patients with no valid seg geometry.")
            df = df[valid].reset_index(drop=True)

        self.df         = df
        self.crop_scale = crop_scale

        if is_main_process():
            cs = df["adaptive_crop_size"]
            print(f"[Dataset] {len(df)} ccRCC patients  |  "
                  f"ISUP: {Counter(df['isup_grade'].tolist())}")
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


# ══════════════════════════════════════════════════════════════════════════════
# Vision embedding extractor
# ══════════════════════════════════════════════════════════════════════════════
class VisionEmbeddingExtractor(nn.Module):
    def __init__(self, vision_model, projection,
                 use_hier=False, interaction_indices=None):
        super().__init__()
        self.vision_model        = vision_model
        self.projection          = projection
        self.use_hier            = use_hier
        self.interaction_indices = interaction_indices or []

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_hier and self.interaction_indices:
            layer_outputs = self.vision_model.get_intermediate_layers(
                x, n=self.interaction_indices,
                return_class_token=True, norm=True,
            )
            h_parts = []
            for patch_tokens, cls_token in layer_outputs:
                h_parts.append(torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1))
            return F.normalize(torch.cat(h_parts, dim=-1), dim=-1)

        out        = self.vision_model(x, is_training=True)
        cls_token  = out["x_norm_clstoken"]
        mean_patch = out["x_norm_patchtokens"].mean(dim=1)
        h          = self.projection(torch.cat([cls_token, mean_patch], dim=-1))
        return F.normalize(h, dim=-1)


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


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════
class IndexedDataset(Dataset):
    def __init__(self, base: C4KCAdaptiveDataset, indices: List[int]):
        self.base    = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.base[self.indices[idx]]
        return s["image"], s["patient_id"]


def _collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def extract_features(
    model:       VisionEmbeddingExtractor,
    dataset:     C4KCAdaptiveDataset,
    indices:     List[int],
    device:      str = "cuda",
    batch_size:  int = 4,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, List[str]]:
    sub        = IndexedDataset(dataset, indices)
    world_size = get_world_size()
    rank       = get_rank()
    sampler    = (
        torch.utils.data.DistributedSampler(
            sub, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    loader = DataLoader(
        sub, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn,
        persistent_workers=num_workers > 0, drop_last=False,
    )

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
        gathered    = [torch.zeros_like(local_feats) for _ in range(world_size)]
        dist.all_gather(gathered, local_feats)
        all_feats   = torch.cat(gathered, dim=0).cpu()
        all_pids_nested: List[List[str]] = [None] * world_size  # type: ignore
        dist.all_gather_object(all_pids_nested, local_pids)
        all_pids = [p for sub_ in all_pids_nested for p in sub_]
        n        = len(indices)
        return all_feats[:n], all_pids[:n]

    return local_feats, local_pids


# ══════════════════════════════════════════════════════════════════════════════
# Stratified K-fold by individual ISUP grade (1/2/3/4)
# ══════════════════════════════════════════════════════════════════════════════
def stratified_kfold(
    isup_grades: List[int], n_folds: int = 5, seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Stratify by individual ISUP grade so each grade is balanced across folds."""
    rng = np.random.default_rng(seed)
    grade_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(isup_grades):
        grade_to_indices[g].append(i)

    n = len(isup_grades)
    fold_assignments = np.full(n, -1, dtype=int)
    for grade in sorted(grade_to_indices):
        idxs = np.array(grade_to_indices[grade])
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            fold_assignments[idx] = j % n_folds

    folds = []
    for f in range(n_folds):
        test_idx  = np.where(fold_assignments == f)[0].tolist()
        train_idx = np.where(fold_assignments != f)[0].tolist()
        folds.append((train_idx, test_idx))
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════
def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1];  neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    concordant = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(concordant / (len(pos) * len(neg)))


def compute_ovr_auc(y_true, probas, num_classes=NUM_CLASSES):
    valid = [_binary_auc(probas[:, c], (y_true == c).astype(int))
             for c in range(num_classes)]
    valid = [v for v in valid if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _binary_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order         = np.argsort(-scores)
    sorted_labels = labels[order].astype(float)
    tp_cum        = np.cumsum(sorted_labels)
    precision     = tp_cum / np.arange(1, len(sorted_labels) + 1)
    recall        = tp_cum / n_pos
    delta_recall  = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * delta_recall))


def compute_ovr_pr_auc(y_true, probas, num_classes=NUM_CLASSES):
    valid = [_binary_pr_auc(probas[:, c], (y_true == c).astype(int))
             for c in range(num_classes)]
    valid = [v for v in valid if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_metrics(y_true, y_pred, probas, class_names):
    correct = y_true == y_pred
    per_class_acc, per_class_f1 = {}, {}
    per_class_auc, per_class_pr_auc, per_class_count = {}, {}, {}

    for c, name in enumerate(class_names):
        mask                  = y_true == c
        per_class_count[name] = int(mask.sum())
        binary_labels         = (y_true == c).astype(int)
        if mask.sum() == 0:
            per_class_acc[name] = per_class_f1[name] = 0.0
            per_class_auc[name] = per_class_pr_auc[name] = float("nan")
            continue
        per_class_acc[name] = float(correct[mask].mean())
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[name]     = (2 * prec * rec / (prec + rec)
                                   if (prec + rec) > 0 else 0.0)
        per_class_auc[name]    = _binary_auc(probas[:, c], binary_labels)
        per_class_pr_auc[name] = _binary_pr_auc(probas[:, c], binary_labels)

    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    return {
        "accuracy":           float(correct.mean()),
        "balanced_accuracy":  float(np.mean(list(per_class_acc.values()))),
        "macro_f1":           float(np.mean(list(per_class_f1.values()))),
        "macro_auc":          compute_ovr_auc(y_true, probas),
        "macro_pr_auc":       compute_ovr_pr_auc(y_true, probas),
        "per_class_accuracy": per_class_acc,
        "per_class_f1":       per_class_f1,
        "per_class_auc":      per_class_auc,
        "per_class_pr_auc":   per_class_pr_auc,
        "per_class_count":    per_class_count,
        "confusion_matrix":   confusion.tolist(),
        "pred_histogram":     {class_names[c]: int((y_pred == c).sum())
                               for c in range(len(class_names))},
        "n_test":             len(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════════
# sklearn pipeline
# ══════════════════════════════════════════════════════════════════════════════
def make_logreg(C: float, solver: str, max_iter: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            multi_class="multinomial", solver=solver,
            class_weight="balanced", C=C,
            max_iter=max_iter, random_state=0,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_cv(feats, labels, isup_grades, class_names,
                seed, n_folds, C_values, solver, max_iter):
    folds      = stratified_kfold(isup_grades, n_folds=n_folds, seed=seed)
    n          = len(labels)
    oof_probas = {C: np.zeros((n, NUM_CLASSES)) for C in C_values}

    for train_idx, test_idx in folds:
        X_train = feats[train_idx];  X_test = feats[test_idx]
        y_train = labels[train_idx]
        for C in C_values:
            clf = make_logreg(C, solver, max_iter)
            clf.fit(X_train, y_train)
            oof_probas[C][test_idx] = clf.predict_proba(X_test)

    results = {}
    best_ba = -1.0
    for C in C_values:
        preds = oof_probas[C].argmax(axis=1)
        m     = compute_metrics(labels, preds, oof_probas[C], class_names)
        results[f"C={C}"] = m
        if m["balanced_accuracy"] > best_ba:
            best_ba = m["balanced_accuracy"]
            results["best"] = {**m, "C": C}
    return results


def evaluate_repeated_cv(feats, labels, isup_grades, class_names,
                         n_repeats, n_folds, C_values, solver,
                         max_iter, base_seed=0):
    best_metrics = defaultdict(list)
    for rep in range(n_repeats):
        r = evaluate_cv(feats, labels, isup_grades, class_names,
                        base_seed + rep, n_folds, C_values, solver, max_iter)
        for mname in ["balanced_accuracy", "macro_f1", "macro_auc", "macro_pr_auc"]:
            best_metrics[mname].append(r["best"][mname])

    def summarize(vals):
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "ci_lower": float(np.quantile(arr, 0.025)),
                "ci_upper": float(np.quantile(arr, 0.975))}

    return {"best": {k: summarize(v) for k, v in best_metrics.items()},
            "n_repeats": n_repeats, "n_folds": n_folds}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run_linear_probe(args: argparse.Namespace) -> Dict:
    setup_distributed()
    rank   = get_rank()
    device = (f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
              if torch.cuda.is_available() else "cpu")
    args.checkpoint = resolve_retrieval_checkpoint(args.ckpt_type, args.checkpoint)

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    if is_main_process():
        print(f"Building dataset (crop_scale={args.crop_scale}) …")
    dataset = C4KCAdaptiveDataset(
        clinical_csv = args.clinical_csv,
        image_dir    = args.image_dir,
        crop_scale   = args.crop_scale,
    )

    # ── 2. Labels ─────────────────────────────────────────────────────────────
    isup_grades_all = dataset.df["isup_grade"].tolist()
    groups_all      = dataset.df["isup_group"].tolist()
    all_indices     = list(range(len(dataset)))
    labels          = np.array([GROUP_TO_IDX[g] for g in groups_all])

    if is_main_process():
        print(f"Total patients:  {len(all_indices)}")
        print(f"ISUP grade dist: {Counter(isup_grades_all)}")
        print(f"Group dist:      {Counter(groups_all)}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    if is_main_process():
        print("\nLoading model …")
    model = load_vision_model(
        args.checkpoint, device=device, ckpt_type=args.ckpt_type,
        use_hier=args.use_hier, interaction_indices=args.interaction_indices,
    )

    # ── 4. Feature extraction ─────────────────────────────────────────────────
    if is_main_process():
        print(f"\nExtracting features (crop_scale={args.crop_scale}) …")
    feats, patient_ids = extract_features(
        model, dataset, all_indices, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    feats_np = feats.numpy()
    if is_main_process():
        print(f"Features shape: {feats_np.shape}")

    # ── 5. Evaluate (rank 0 only) ─────────────────────────────────────────────
    results: Dict = {}
    if is_main_process():
        class_names = [IDX_TO_GROUP[i] for i in range(NUM_CLASSES)]
        C_values    = args.C_values

        results = {
            "config": {
                "checkpoint":          args.checkpoint,
                "ckpt_type":           args.ckpt_type,
                "use_hier":            args.use_hier,
                "interaction_indices": args.interaction_indices,
                "crop_scale":          args.crop_scale,
                "C_values":            C_values,
                "solver":              args.solver,
                "max_iter":            args.max_iter,
                "n_folds":             args.n_folds,
                "n_repeats":           args.n_repeats,
                "primary_seed":        args.seed,
                "feature_dim":         feats_np.shape[1],
                "label":               "ISUP_grade_binary (Low=1-2, High=3-4)",
                "population":          "ccRCC only",
                "crop_strategy":       "adaptive bbox_max_extent × crop_scale",
                "intensity":           "clip[-1000,1000] → z-score whole vol → crop → pad 0.0",
            },
            "split_info": {
                "n_valid":        len(all_indices),
                "isup_grade_dist": dict(Counter(isup_grades_all)),
                "group_dist":     dict(Counter(groups_all)),
            },
        }

        # Primary CV
        primary = evaluate_cv(
            feats_np, labels, isup_grades_all, class_names,
            args.seed, args.n_folds, C_values, args.solver, args.max_iter,
        )
        results["primary_cv"] = primary
        best = primary["best"]

        print(f"\n{'='*70}")
        print(f"  [{args.n_folds}-fold CV, seed={args.seed}] "
              f"Best linear probe (C={best['C']}):")
        print(f"{'='*70}")
        print(f"    Balanced Accuracy: {best['balanced_accuracy']:.4f}")
        print(f"    Macro F1:          {best['macro_f1']:.4f}")
        print(f"    Macro AUC (OvR):   {best['macro_auc']:.4f}")
        print(f"    Macro PR AUC:      {best['macro_pr_auc']:.4f}")
        print(f"    Pred histogram:    {best['pred_histogram']}")
        for g in class_names:
            print(f"      {g:>6s}: acc={best['per_class_accuracy'][g]:.4f}  "
                  f"f1={best['per_class_f1'][g]:.4f}  "
                  f"auc={best['per_class_auc'].get(g, float('nan')):.4f}  "
                  f"pr={best['per_class_pr_auc'].get(g, float('nan')):.4f}  "
                  f"(n={best['per_class_count'][g]})")

        cm = np.array(best["confusion_matrix"])
        print("    Confusion matrix:")
        header = "          " + "  ".join(f"{g:>8s}" for g in class_names)
        print(f"    {header}")
        for r, g in enumerate(class_names):
            row_str = "  ".join(f"{cm[r, c]:>8d}" for c in range(NUM_CLASSES))
            print(f"      {g:>8s}  {row_str}")

        print(f"\n  C sweep (seed={args.seed}):")
        for C in C_values:
            m = primary[f"C={C}"]
            print(f"    C={C:<10.4f}  "
                  f"BalAcc={m['balanced_accuracy']:.4f}  "
                  f"F1={m['macro_f1']:.4f}  "
                  f"AUC={m['macro_auc']:.4f}  "
                  f"PR={m['macro_pr_auc']:.4f}")

        # Repeated CV
        repeated = evaluate_repeated_cv(
            feats_np, labels, isup_grades_all, class_names,
            n_repeats=args.n_repeats, n_folds=args.n_folds,
            C_values=C_values, solver=args.solver, max_iter=args.max_iter,
        )
        results["repeated_cv"] = repeated

        print(f"\n  [{args.n_repeats}× {args.n_folds}-fold CV summary]")
        s = repeated["best"]
        parts = []
        for mname, short in [
            ("balanced_accuracy", "BalAcc"), ("macro_f1", "F1"),
            ("macro_auc", "AUC"), ("macro_pr_auc", "PR"),
        ]:
            v = s[mname]
            parts.append(f"{short}={v['mean']:.4f} "
                         f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")
        print(f"    {'  '.join(parts)}")

        out_dir  = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "linear_probe_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    cleanup_distributed()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear probing for C4KC-KiTS ISUP grade (ccRCC) — adaptive tumor crop"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help=("Explicit model checkpoint. If omitted, uses FLEXICT_CHECKPOINT, "
                         "then FLEXICT_VLM_CHECKPOINT for --ckpt_type vlm or "
                         "FLEXICT_3D_CHECKPOINT for --ckpt_type teacher."))
    p.add_argument("--ckpt_type", type=str, default="vlm", choices=["vlm", "teacher"])
    p.add_argument("--use_hier", action="store_true", default=False)
    p.add_argument("--interaction_indices", type=int, nargs="+", default=[3, 7, 11, 15])
    p.add_argument("--clinical_csv", type=str, default=CLINICAL_CSV)
    p.add_argument("--image_dir",    type=str, default=IMAGE_DIR)
    p.add_argument("--output_dir",   type=str,
                   default=str(ROOT / "results/retrieval/linear_probe/C4KC_ISUP_grade/ours_adaptive_crop"))
    p.add_argument("--crop_scale",   type=float, default=1.5,
                   help="Multiplier applied to bbox_max_extent_mm to get crop_size "
                        "(default: 1.5).")
    p.add_argument("--C_values", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--solver",      type=str,  default="lbfgs",
                   choices=["lbfgs", "saga"])
    p.add_argument("--max_iter",    type=int,  default=20000)
    p.add_argument("--n_folds",     type=int,  default=5)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--n_repeats",   type=int,  default=20,
                   help="Number of repeated CVs for robust CIs (default: 20).")
    p.add_argument("--batch_size",  type=int,  default=4)
    p.add_argument("--num_workers", type=int,  default=4)
    return p.parse_args()


if __name__ == "__main__":
    run_linear_probe(parse_args())
