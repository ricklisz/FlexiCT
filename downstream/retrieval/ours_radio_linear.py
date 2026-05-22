#!/usr/bin/env python3
"""Linear probing for T-stage classification on RADIO using FlexiMedDINOv3_VLM.

Extracts frozen VLM vision embeddings, then trains a multinomial logistic
regression classifier (sklearn) per fold in repeated stratified 5-fold CV.
Supports single-pass or two-pass feature extraction (two crop sizes →
concatenate + L2-norm).

Grid-searches over regularisation strength C (inverse of L2 penalty).
Features are StandardScaler-normalised before fitting.

Metrics: balanced accuracy, macro F1, macro AUC (OvR), macro PR AUC (OvR)
with 95 % confidence intervals from repeated CV.

T-stage groups:
    Early       : Tis, T1a, T1b
    Intermediate: T2a, T2b
    Advanced    : T3, T4
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms as monai_transforms
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flexi_ct import Flexi_CT_3D, Flexi_CT_VLM
from flexi_ct.checkpoints import resolve_retrieval_checkpoint
try:
    from .preprocess import SeedBasedPatchCropd
    from .radio_utils import normalize_radio_image_paths, validate_radio_image_paths
except ImportError:
    from preprocess import SeedBasedPatchCropd
    from radio_utils import normalize_radio_image_paths, validate_radio_image_paths


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
# Paths & constants
# ══════════════════════════════════════════════════════════════════════════════
CSV_DIR = Path(__file__).resolve().parent / "csv"
ANNOTATIONS_CSV = (
    os.environ.get(
        "RADIO_ANNOTATIONS_CSV",
        str(CSV_DIR / "radio_annotations_fixed.csv"),
    )
)
METADATA_CSV = (
    os.environ.get(
        "RADIO_METADATA_CSV",
        str(CSV_DIR / "radio_nsclc_radiogenomics.csv"),
    )
)

T_STAGE_TO_GROUP: Dict[str, str] = {
    "Tis": "Early",
    "T1a": "Early",
    "T1b": "Early",
    "T2a": "Intermediate",
    "T2b": "Intermediate",
    "T3": "Advanced",
    "T4": "Advanced",
}

GROUP_TO_IDX: Dict[str, int] = {
    "Early": 0,
    "Intermediate": 1,
    "Advanced": 2,
}
IDX_TO_GROUP: Dict[int, str] = {v: k for k, v in GROUP_TO_IDX.items()}
NUM_CLASSES = len(GROUP_TO_IDX)


# ══════════════════════════════════════════════════════════════════════════════
# Data preprocessing
# ══════════════════════════════════════════════════════════════════════════════
class ClipAndZScored(monai_transforms.MapTransform):
    def __init__(self, keys, clip_min: float = -1000, clip_max: float = 1000,
                 allow_missing_keys: bool = False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            t = torch.clamp(d[key], self.clip_min, self.clip_max)
            mean = float(t.mean())
            std = float(t.std())
            if std < 1e-6:
                t = t - mean
            else:
                t = (t - mean) / std
            d[key] = t
        return d


class RadioRetrievalDataset(Dataset):
    def __init__(
        self,
        annotations_csv: str = ANNOTATIONS_CSV,
        metadata_csv: str = METADATA_CSV,
        pixdim: float = 1.0,
        orientation: str = "LPS",
        crop: bool = False,
        roi_size: Union[Tuple[int, int, int], int] = (50, 50, 50),
        coord_orientation: str = "LPS",
        pad_size: Optional[Union[Tuple[int, int, int], int]] = None,
    ):
        self.annotations = normalize_radio_image_paths(pd.read_csv(annotations_csv))
        validate_radio_image_paths(self.annotations, annotations_csv)
        metadata = pd.read_csv(metadata_csv)
        self.metadata_lookup = metadata.set_index("Case ID")

        if isinstance(roi_size, int):
            roi_size = (roi_size, roi_size, roi_size)
        if isinstance(pad_size, int):
            pad_size = (pad_size, pad_size, pad_size)

        steps: list = [
            monai_transforms.LoadImaged(
                keys=["image_path"], image_only=True, reader="NibabelReader"
            ),
            monai_transforms.EnsureChannelFirstd(keys=["image_path"]),
            ClipAndZScored(keys=["image_path"], clip_min=-1000, clip_max=1000),
            monai_transforms.Spacingd(
                keys=["image_path"], pixdim=pixdim,
                padding_mode="zeros", mode="linear",
                align_corners=True, diagonal=True,
            ),
            monai_transforms.Orientationd(
                keys=["image_path"], axcodes=orientation
            ),
        ]

        if crop:
            steps.append(SeedBasedPatchCropd(
                keys=["image_path"], roi_size=roi_size[::-1],
                coord_orientation=coord_orientation, global_coordinates=True,
            ))
        steps.append(monai_transforms.Transposed(
            keys=["image_path"], indices=(0, 3, 2, 1)
        ))

        if pad_size is not None:
            steps.append(monai_transforms.SpatialPadd(
                keys=["image_path"], spatial_size=pad_size
            ))

        steps.append(monai_transforms.Resized(
            keys=["image_path"], spatial_size=(160, 160, 160), mode="bilinear"
        ))

        self.transform = monai_transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        row = self.annotations.iloc[idx]
        data_dict = {
            "image_path": row["image_path"],
            "coordX": float(row["coordX"]),
            "coordY": float(row["coordY"]),
            "coordZ": float(row["coordZ"]),
        }
        data_dict = self.transform(data_dict)

        case_id = row["Case ID"]
        metadata = (
            self.metadata_lookup.loc[case_id].to_dict()
            if case_id in self.metadata_lookup.index else {}
        )
        return {
            "image": data_dict["image_path"],
            "coordX": data_dict["coordX"],
            "coordY": data_dict["coordY"],
            "coordZ": data_dict["coordZ"],
            "case_id": case_id,
            "metadata": metadata,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Vision embedding extractor
# ══════════════════════════════════════════════════════════════════════════════
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
            h_parts = []
            for patch_tokens, cls_token in layer_outputs:
                mean_patch = patch_tokens.mean(dim=1)
                h_parts.append(torch.cat([cls_token, mean_patch], dim=-1))
            h = torch.cat(h_parts, dim=-1)
            return F.normalize(h, dim=-1)

        out = self.vision_model(x, is_training=True)
        cls_token = out["x_norm_clstoken"]
        mean_patch = out["x_norm_patchtokens"].mean(dim=1)
        h = torch.cat([cls_token, mean_patch], dim=-1)
        h = self.projection(h)
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

    model = VisionEmbeddingExtractor(
        vision_model, projection,
        use_hier=use_hier, interaction_indices=interaction_indices,
    )
    model.to(device).eval()
    mode_str = f"hier blocks {interaction_indices}" if use_hier else ckpt_type
    print(f"Loaded local FlexiCT-backed vision model from {ckpt_path} (mode={mode_str})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════
class FeatureExtractionDataset(Dataset):
    def __init__(self, base_dataset: RadioRetrievalDataset, indices: List[int]):
        self.base = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample = self.base[self.indices[idx]]
        img = sample["image"]
        if hasattr(img, "as_tensor"):
            img = img.as_tensor()
        return img, sample["case_id"]


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    case_ids = [b[1] for b in batch]
    return images, case_ids


@torch.no_grad()
def extract_features(
    model: VisionEmbeddingExtractor,
    dataset: RadioRetrievalDataset,
    indices: List[int],
    device: str = "cuda",
    batch_size: int = 1,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, List[str]]:
    feat_ds = FeatureExtractionDataset(dataset, indices)
    world_size = get_world_size()
    rank = get_rank()

    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            feat_ds, num_replicas=world_size, rank=rank, shuffle=False,
        )
    else:
        sampler = None

    loader = DataLoader(
        feat_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn,
        persistent_workers=num_workers > 0, drop_last=False,
    )

    local_features: List[torch.Tensor] = []
    local_case_ids: List[str] = []

    for images, cids in tqdm(loader, desc="Extracting", disable=rank != 0):
        images = images.to(device, non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            feat = model(images)
        local_features.append(feat.float().cpu())
        local_case_ids.extend(cids)

    local_features = torch.cat(local_features, dim=0)

    if world_size > 1:
        local_features = local_features.to(device)
        gathered = [torch.zeros_like(local_features) for _ in range(world_size)]
        dist.all_gather(gathered, local_features)
        all_features = torch.cat(gathered, dim=0).cpu()

        all_case_ids_nested: List[List[str]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(all_case_ids_nested, local_case_ids)
        all_case_ids = [cid for sublist in all_case_ids_nested for cid in sublist]

        n = len(indices)
        return all_features[:n], all_case_ids[:n]

    return local_features, local_case_ids


# ══════════════════════════════════════════════════════════════════════════════
# Stratified K-fold
# ══════════════════════════════════════════════════════════════════════════════
def stratified_kfold(
    t_stages: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    rng = np.random.default_rng(seed)
    stage_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, stage in enumerate(t_stages):
        stage_to_indices[stage].append(i)

    n = len(t_stages)
    fold_assignments = np.full(n, -1, dtype=int)

    for stage in sorted(stage_to_indices):
        idxs = np.array(stage_to_indices[stage])
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            fold_assignments[idx] = j % n_folds

    folds = []
    for f in range(n_folds):
        test_idx = np.where(fold_assignments == f)[0].tolist()
        train_idx = np.where(fold_assignments != f)[0].tolist()
        folds.append((train_idx, test_idx))

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════════════
def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    concordant = 0.0
    for p in pos:
        concordant += (p > neg).sum() + 0.5 * (p == neg).sum()
    return float(concordant / (len(pos) * len(neg)))


def compute_ovr_auc(
    y_true: np.ndarray, probas: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    valid = []
    for c in range(num_classes):
        auc = _binary_auc(probas[:, c], (y_true == c).astype(int))
        if not np.isnan(auc):
            valid.append(auc)
    return float(np.mean(valid)) if valid else float("nan")


def _binary_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order].astype(float)
    tp_cum = np.cumsum(sorted_labels)
    precision = tp_cum / np.arange(1, len(sorted_labels) + 1)
    recall = tp_cum / n_pos
    recall_with_zero = np.concatenate([[0.0], recall])
    delta_recall = np.diff(recall_with_zero)
    return float(np.sum(precision * delta_recall))


def compute_ovr_pr_auc(
    y_true: np.ndarray, probas: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    valid = []
    for c in range(num_classes):
        ap = _binary_pr_auc(probas[:, c], (y_true == c).astype(int))
        if not np.isnan(ap):
            valid.append(ap)
    return float(np.mean(valid)) if valid else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probas: np.ndarray,
    class_names: List[str],
) -> Dict:
    correct = y_true == y_pred
    accuracy = float(correct.mean())

    per_class_acc, per_class_f1, per_class_count = {}, {}, {}
    per_class_auc, per_class_pr_auc = {}, {}

    for c, name in enumerate(class_names):
        mask = y_true == c
        per_class_count[name] = int(mask.sum())
        binary_labels = (y_true == c).astype(int)
        if mask.sum() == 0:
            per_class_acc[name] = 0.0
            per_class_f1[name] = 0.0
            per_class_auc[name] = float("nan")
            per_class_pr_auc[name] = float("nan")
            continue
        per_class_acc[name] = float(correct[mask].mean())

        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[name] = (
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )
        per_class_auc[name] = _binary_auc(probas[:, c], binary_labels)
        per_class_pr_auc[name] = _binary_pr_auc(probas[:, c], binary_labels)

    balanced_accuracy = float(np.mean(list(per_class_acc.values())))
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    macro_auc = compute_ovr_auc(y_true, probas)
    macro_pr_auc = compute_ovr_pr_auc(y_true, probas)

    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    pred_hist = {class_names[c]: int((y_pred == c).sum())
                 for c in range(len(class_names))}

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "per_class_accuracy": per_class_acc,
        "per_class_f1": per_class_f1,
        "per_class_auc": per_class_auc,
        "per_class_pr_auc": per_class_pr_auc,
        "per_class_count": per_class_count,
        "confusion_matrix": confusion.tolist(),
        "pred_histogram": pred_hist,
        "n_test": len(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════════
# sklearn logistic regression (StandardScaler + LogisticRegression)
# ══════════════════════════════════════════════════════════════════════════════
def make_logreg(C: float, solver: str, max_iter: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            multi_class="multinomial",
            solver=solver,
            class_weight="balanced",
            C=C,
            max_iter=max_iter,
            random_state=0,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_cv(
    feats: np.ndarray,
    labels: np.ndarray,
    t_stages: List[str],
    class_names: List[str],
    seed: int,
    n_folds: int,
    C_values: List[float],
    solver: str,
    max_iter: int,
) -> Dict:
    folds = stratified_kfold(t_stages, n_folds=n_folds, seed=seed)
    n = len(labels)

    oof_probas = {C: np.zeros((n, NUM_CLASSES)) for C in C_values}

    for train_idx, test_idx in folds:
        X_train = feats[train_idx]
        X_test = feats[test_idx]
        y_train = labels[train_idx]

        for C in C_values:
            clf = make_logreg(C, solver, max_iter)
            clf.fit(X_train, y_train)
            oof_probas[C][test_idx] = clf.predict_proba(X_test)

    results: Dict = {}
    best_ba = -1.0
    for C in C_values:
        preds = oof_probas[C].argmax(axis=1)
        m = compute_metrics(labels, preds, oof_probas[C], class_names)
        results[f"C={C}"] = m
        if m["balanced_accuracy"] > best_ba:
            best_ba = m["balanced_accuracy"]
            results["best"] = {**m, "C": C}

    return results


def evaluate_repeated_cv(
    feats: np.ndarray,
    labels: np.ndarray,
    t_stages: List[str],
    class_names: List[str],
    n_repeats: int,
    n_folds: int,
    C_values: List[float],
    solver: str,
    max_iter: int,
    base_seed: int = 0,
) -> Dict:
    best_metrics = defaultdict(list)

    for rep in range(n_repeats):
        seed = base_seed + rep
        r = evaluate_cv(
            feats, labels, t_stages, class_names, seed,
            n_folds, C_values, solver, max_iter,
        )
        for metric_name in [
            "balanced_accuracy", "macro_f1", "macro_auc", "macro_pr_auc",
        ]:
            best_metrics[metric_name].append(r["best"][metric_name])

    def summarize(vals):
        arr = np.array(vals)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lower": float(np.quantile(arr, 0.025)),
            "ci_upper": float(np.quantile(arr, 0.975)),
        }

    return {
        "best": {k: summarize(v) for k, v in best_metrics.items()},
        "n_repeats": n_repeats,
        "n_folds": n_folds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run_linear_probe(args: argparse.Namespace) -> Dict:
    setup_distributed()
    rank = get_rank()
    device = (
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available() else "cpu"
    )
    args.checkpoint = resolve_retrieval_checkpoint(args.ckpt_type, args.checkpoint)

    # ── 1. Resolve crop sizes ─────────────────────────────────────────────────
    roi_sizes = args.roi_sizes
    two_pass = len(roi_sizes) == 2
    if is_main_process():
        mode = (f"two-pass ({roi_sizes[0]}, {roi_sizes[1]})"
                if two_pass else f"single-pass ({roi_sizes[0]})")
        print(f"Feature extraction mode: {mode}")

    # ── 2. Build dataset(s) ───────────────────────────────────────────────────
    datasets = []
    for roi in roi_sizes:
        if is_main_process():
            print(f"Building dataset (roi_size={roi}) ...")
        ds = RadioRetrievalDataset(
            annotations_csv=args.annotations_csv,
            metadata_csv=args.metadata_csv,
            pixdim=args.pixdim,
            crop=True,
            roi_size=roi,
            pad_size=roi,
        )
        datasets.append(ds)

    # ── 3. Resolve T-stage for every sample ───────────────────────────────────
    base_ds = datasets[0]
    case_ids_all = base_ds.annotations["Case ID"].tolist()
    t_stages_all: List[str] = []
    valid_mask: List[bool] = []
    for cid in case_ids_all:
        if cid in base_ds.metadata_lookup.index:
            stage = str(
                base_ds.metadata_lookup.loc[cid, "Pathological T stage"]
            ).strip()
        else:
            stage = ""
        t_stages_all.append(stage)
        valid_mask.append(stage in T_STAGE_TO_GROUP)

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_t_stages = [t_stages_all[i] for i in valid_indices]
    valid_groups = [T_STAGE_TO_GROUP[s] for s in valid_t_stages]
    labels = np.array([GROUP_TO_IDX[g] for g in valid_groups])

    if is_main_process():
        print(f"Valid T-stage: {len(valid_indices)} / {len(case_ids_all)}")
        print(f"T-stage dist: {Counter(valid_t_stages)}")
        print(f"Group dist:   {Counter(valid_groups)}")

    # ── 4. Load model ─────────────────────────────────────────────────────────
    if is_main_process():
        print("\nLoading model ...")
    model = load_vision_model(
        args.checkpoint, device=device, ckpt_type=args.ckpt_type,
        use_hier=args.use_hier,
        interaction_indices=args.interaction_indices,
    )

    # ── 5. Extract features ───────────────────────────────────────────────────
    feat_list = []
    for i, ds in enumerate(datasets):
        if is_main_process():
            print(f"\nExtracting features (roi_size={roi_sizes[i]}) ...")
        feats, case_ids = extract_features(
            model, ds, valid_indices, device=device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        feat_list.append(feats)
        if is_main_process():
            print(f"  Features shape: {feats.shape}")

    if two_pass:
        feats_all = F.normalize(
            torch.cat(feat_list, dim=-1), dim=-1,
        ).numpy()
        if is_main_process():
            print(f"Fused features shape: {feats_all.shape}")
    else:
        feats_all = feat_list[0].numpy()

    # ── 6. Run evaluation (rank 0 only) ───────────────────────────────────────
    results: Dict = {}
    if is_main_process():
        class_names = [IDX_TO_GROUP[i] for i in range(NUM_CLASSES)]
        C_values = args.C_values
        n_folds = args.n_folds
        solver = args.solver
        max_iter = args.max_iter

        results = {
            "config": {
                "checkpoint": args.checkpoint,
                "ckpt_type": args.ckpt_type,
                "use_hier": args.use_hier,
                "interaction_indices": args.interaction_indices,
                "roi_sizes": roi_sizes,
                "C_values": C_values,
                "solver": solver,
                "max_iter": max_iter,
                "n_folds": n_folds,
                "n_repeats": args.n_repeats,
                "primary_seed": args.seed,
                "feature_dim": feats_all.shape[1],
            },
            "split_info": {
                "n_valid": len(valid_indices),
                "t_stage_dist": dict(Counter(valid_t_stages)),
                "group_dist": dict(Counter(valid_groups)),
            },
        }

        # ── Primary CV ────────────────────────────────────────────────────────
        primary = evaluate_cv(
            feats_all, labels, valid_t_stages, class_names,
            args.seed, n_folds, C_values, solver, max_iter,
        )
        results["primary_cv"] = primary

        best = primary["best"]
        print(f"\n{'='*70}")
        print(f"  [{n_folds}-fold CV, seed={args.seed}] "
              f"Best linear probe (C={best['C']}):")
        print(f"{'='*70}")
        print(f"    Balanced Accuracy: {best['balanced_accuracy']:.4f}")
        print(f"    Macro F1:          {best['macro_f1']:.4f}")
        print(f"    Macro AUC (OvR):   {best['macro_auc']:.4f}")
        print(f"    Macro PR AUC:      {best['macro_pr_auc']:.4f}")
        print(f"    Pred histogram:    {best['pred_histogram']}")
        for g in class_names:
            print(
                f"      {g:>14s}: acc={best['per_class_accuracy'][g]:.4f}  "
                f"f1={best['per_class_f1'][g]:.4f}  "
                f"auc={best['per_class_auc'].get(g, float('nan')):.4f}  "
                f"pr={best['per_class_pr_auc'].get(g, float('nan')):.4f}  "
                f"(n={best['per_class_count'][g]})"
            )
        cm = np.array(best["confusion_matrix"])
        print(f"    Confusion matrix:")
        header = "            " + "  ".join(f"{g:>12s}" for g in class_names)
        print(f"    {header}")
        for r, g in enumerate(class_names):
            row_str = "  ".join(f"{cm[r, c]:>12d}" for c in range(NUM_CLASSES))
            print(f"      {g:>12s}  {row_str}")

        # ── All C results ─────────────────────────────────────────────────────
        print(f"\n  C sweep (seed={args.seed}):")
        for C in C_values:
            m = primary[f"C={C}"]
            print(
                f"    C={C:<10.4f}  "
                f"BalAcc={m['balanced_accuracy']:.4f}  "
                f"F1={m['macro_f1']:.4f}  "
                f"AUC={m['macro_auc']:.4f}  "
                f"PR={m['macro_pr_auc']:.4f}"
            )

        # ── Repeated CV for robust CIs ────────────────────────────────────────
        repeated = evaluate_repeated_cv(
            feats_all, labels, valid_t_stages, class_names,
            n_repeats=args.n_repeats, n_folds=n_folds,
            C_values=C_values, solver=solver, max_iter=max_iter,
        )
        results["repeated_cv"] = repeated

        print(f"\n  [{args.n_repeats}x {n_folds}-fold CV summary (best C per repeat)]")
        s = repeated["best"]
        parts = []
        for mname, short in [
            ("balanced_accuracy", "BalAcc"),
            ("macro_f1", "F1"),
            ("macro_auc", "AUC"),
            ("macro_pr_auc", "PR"),
        ]:
            v = s[mname]
            parts.append(
                f"{short}={v['mean']:.4f} "
                f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
            )
        print(f"    {'  '.join(parts)}")

        # ── Save ──────────────────────────────────────────────────────────────
        out_dir = Path(args.output_dir)
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
        description="Linear probing (sklearn LogisticRegression) for T-stage "
                    "on RADIO with FlexiMedDINOv3_VLM"
    )
    p.add_argument(
        "--checkpoint", type=str,
        default=None,
        help=(
            "Explicit model checkpoint. If omitted, uses FLEXICT_CHECKPOINT, "
            "then FLEXICT_VLM_CHECKPOINT for --ckpt_type vlm or "
            "FLEXICT_3D_CHECKPOINT for --ckpt_type teacher."
        ),
    )
    p.add_argument(
        "--ckpt_type", type=str, default="vlm", choices=["vlm", "teacher"],
    )
    p.add_argument("--use_hier", action="store_true", default=True)
    p.add_argument(
        "--interaction_indices", type=int, nargs="+", default=[3, 7, 11, 15],
    )
    p.add_argument(
        "--output_dir", type=str,
        default=str(ROOT / "results/linear_probe/radio_T_stage/ours_sklearn_hier"),
    )
    p.add_argument("--annotations_csv", type=str, default=ANNOTATIONS_CSV)
    p.add_argument("--metadata_csv", type=str, default=METADATA_CSV)
    p.add_argument(
        "--roi_sizes", type=int, nargs="+", default=[32, 64],
        help="Crop size(s). One value = single-pass; two values = two-pass "
             "(features are concatenated + L2-normalised)",
    )
    p.add_argument(
        "--C_values", type=float, nargs="+",
        default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
        help="Regularisation strengths to grid-search (inverse of L2 penalty)",
    )
    p.add_argument(
        "--solver", type=str, default="lbfgs",
        choices=["lbfgs", "saga"],
        help="LogisticRegression solver",
    )
    p.add_argument("--max_iter", type=int, default=20000,
                   help="Max iterations for the solver")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_repeats", type=int, default=2,
                   help="Number of repeated CVs for robust CIs")
    p.add_argument("--pixdim", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    run_linear_probe(parse_args())
