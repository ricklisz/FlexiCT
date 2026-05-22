#!/usr/bin/env python3
"""Two-pass lesion-centric T-stage retrieval with Reciprocal Rank Fusion.

Extracts FlexiMedDINOv3_VLM vision embeddings at two crop scales (small and
large), ranks support items independently under each scale via cosine
similarity, then fuses the two rankings with Reciprocal Rank Fusion (RRF).
Evaluates Recall@K, Precision@K, and mAP on the fused ranking.
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


class ClipAndZScored(monai_transforms.MapTransform):
    """Clip HU values to [clip_min, clip_max] then z-score normalise per image."""

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


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# T-stage group mapping
# ──────────────────────────────────────────────────────────────────────────────
T_STAGE_TO_GROUP: Dict[str, str] = {
    "Tis": "Early",
    "T1a": "Early",
    "T1b": "Early",
    "T2a": "Intermediate",
    "T2b": "Intermediate",
    "T3": "Advanced",
    "T4": "Advanced",
}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class RadioRetrievalDataset(Dataset):
    """CT lesion dataset for the RADIO retrieval task."""

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
                keys=["image_path"],
                pixdim=pixdim,
                padding_mode="zeros",
                mode="linear",
                align_corners=True,
                diagonal=True,
            ),
            monai_transforms.Orientationd(
                keys=["image_path"], axcodes=orientation
            ),
        ]

        if crop:
            steps.append(
                SeedBasedPatchCropd(
                    keys=["image_path"],
                    roi_size=roi_size[::-1],
                    coord_orientation=coord_orientation,
                    global_coordinates=True,
                )
            )
            steps.append(
                monai_transforms.Transposed(
                    keys=["image_path"], indices=(0, 3, 2, 1)
                )
            )

        if pad_size is not None:
            steps.append(
                monai_transforms.SpatialPadd(
                    keys=["image_path"], spatial_size=pad_size
                )
            )

        steps.append(
            monai_transforms.Resized(
                keys=["image_path"], spatial_size=(160, 160, 160), mode="bilinear"
            )
        )

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
            if case_id in self.metadata_lookup.index
            else {}
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
# Vision-only embedding extractor
# ══════════════════════════════════════════════════════════════════════════════
class VisionEmbeddingExtractor(nn.Module):
    """Wraps the VLM vision pathway: CLS + mean(patch) → projection → L2-norm.

    When *use_hier* is True, intermediate block outputs at *interaction_indices*
    are extracted, concatenated, and L2-normalised (projection is bypassed).
    """

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
                x,
                n=self.interaction_indices,
                return_class_token=True,
                norm=True,
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
        raise ValueError(f"Unknown ckpt_type='{ckpt_type}' (use 'vlm' or 'teacher')")

    model = VisionEmbeddingExtractor(
        vision_model, projection,
        use_hier=use_hier,
        interaction_indices=interaction_indices,
    )
    model.to(device).eval()
    mode_str = f"hier blocks {interaction_indices}" if use_hier else ckpt_type
    print(f"Loaded local FlexiCT-backed vision model from {ckpt_path} (mode={mode_str})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Stratified 50/50 split by individual T-stage
# ══════════════════════════════════════════════════════════════════════════════
def stratified_split(
    case_ids: List[str],
    t_stages: List[str],
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)

    stage_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, stage in enumerate(t_stages):
        stage_to_indices[stage].append(i)

    query_indices: List[int] = []
    support_indices: List[int] = []

    for stage in sorted(stage_to_indices):
        idxs = np.array(stage_to_indices[stage])
        rng.shuffle(idxs)
        n_query = len(idxs) // 2
        query_indices.extend(idxs[:n_query].tolist())
        support_indices.extend(idxs[n_query:].tolist())

    return query_indices, support_indices


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader-friendly dataset wrapper
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


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (DataLoader + optional DDP)
# ══════════════════════════════════════════════════════════════════════════════
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
        feat_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        persistent_workers=num_workers > 0,
        drop_last=False,
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
# Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════
def reciprocal_rank_fusion(
    sim_matrices: List[torch.Tensor],
    rrf_k: int = 60,
) -> np.ndarray:
    """Fuse multiple [Q, S] similarity matrices via Reciprocal Rank Fusion.

    For each query, every system produces a ranking of the S support items.
    The fused score for support item j given query i is:

        score(i, j) = Σ_sys  1 / (rrf_k + rank_sys(i, j))

    where rank is 0-indexed (0 = most similar).

    Returns:
        fused_scores: ``[Q, S]`` numpy array (higher is better).
    """
    Q, S = sim_matrices[0].shape
    fused = np.zeros((Q, S), dtype=np.float64)

    arange_S = torch.arange(S).unsqueeze(0).expand(Q, -1)

    for sim in sim_matrices:
        sorted_indices = sim.argsort(dim=1, descending=True)       # [Q, S]
        ranks = torch.zeros_like(sorted_indices)
        ranks.scatter_(1, sorted_indices, arange_S)                # [Q, S]
        fused += 1.0 / (rrf_k + ranks.numpy())

    return fused


# ══════════════════════════════════════════════════════════════════════════════
# Metrics (operate on a precomputed [Q, S] score matrix)
# ══════════════════════════════════════════════════════════════════════════════
from math import comb as _comb


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return [Q, k] array of column indices with the highest scores per row."""
    Q, S = scores.shape
    k = min(k, S)
    return np.argpartition(scores, -k, axis=1)[:, -k:]


def _sorted_indices(scores: np.ndarray) -> np.ndarray:
    """Return [Q, S] array of column indices sorted descending by score."""
    return np.argsort(-scores, axis=1)


def compute_recall_at_k(
    scores: np.ndarray,
    query_groups: List[str],
    support_groups: List[str],
    k: int,
) -> np.ndarray:
    """Per-query binary hits: 1 if >= 1 same-group support appears in top-K."""
    effective_k = min(k, scores.shape[1])
    topk = _topk_indices(scores, effective_k)
    support_arr = np.array(support_groups)
    hits = np.zeros(len(query_groups), dtype=bool)
    for i, qg in enumerate(query_groups):
        hits[i] = np.any(support_arr[topk[i]] == qg)
    return hits


def compute_per_group_recall(
    hits: np.ndarray,
    query_groups: List[str],
) -> Dict[str, float]:
    groups_arr = np.array(query_groups)
    out: Dict[str, float] = {}
    for g in sorted(set(query_groups)):
        mask = groups_arr == g
        out[g] = float(hits[mask].mean()) if mask.any() else 0.0
    return out


def random_recall_at_k_single(S: int, R: int, K: int) -> float:
    K = min(K, S)
    if R <= 0 or S <= 0:
        return 0.0
    if K >= S or R >= S:
        return 1.0
    if S - R < K:
        return 1.0
    return 1.0 - _comb(S - R, K) / _comb(S, K)


def compute_random_baseline(
    query_groups: List[str],
    support_groups: List[str],
    k: int,
) -> Dict[str, float]:
    S = len(support_groups)
    support_counts = Counter(support_groups)

    per_group_expected: Dict[str, List[float]] = defaultdict(list)
    all_expected: List[float] = []
    for qg in query_groups:
        R = support_counts.get(qg, 0)
        p = random_recall_at_k_single(S, R, k)
        all_expected.append(p)
        per_group_expected[qg].append(p)

    out = {"overall": float(np.mean(all_expected))}
    for g in sorted(per_group_expected):
        out[g] = float(np.mean(per_group_expected[g]))
    return out


def compute_precision_at_k(
    scores: np.ndarray,
    query_groups: List[str],
    support_groups: List[str],
    k: int,
) -> np.ndarray:
    """Per-query Precision@K: (# relevant in top-K) / K."""
    effective_k = min(k, scores.shape[1])
    sorted_idx = _sorted_indices(scores)[:, :effective_k]
    support_arr = np.array(support_groups)
    prec = np.zeros(len(query_groups), dtype=np.float64)
    for i, qg in enumerate(query_groups):
        prec[i] = (support_arr[sorted_idx[i]] == qg).sum() / effective_k
    return prec


def compute_random_precision(
    query_groups: List[str],
    support_groups: List[str],
    k: int,
) -> Dict[str, float]:
    S = len(support_groups)
    support_counts = Counter(support_groups)

    per_group: Dict[str, List[float]] = defaultdict(list)
    all_prec: List[float] = []
    for qg in query_groups:
        p = support_counts.get(qg, 0) / S
        all_prec.append(p)
        per_group[qg].append(p)

    out = {"overall": float(np.mean(all_prec))}
    for g in sorted(per_group):
        out[g] = float(np.mean(per_group[g]))
    return out


def compute_map(
    scores: np.ndarray,
    query_groups: List[str],
    support_groups: List[str],
) -> np.ndarray:
    """Per-query Average Precision (AP). Mean gives mAP."""
    sorted_idx = _sorted_indices(scores)
    support_arr = np.array(support_groups)
    aps = np.zeros(len(query_groups))
    for i, qg in enumerate(query_groups):
        ranked = support_arr[sorted_idx[i]]
        relevant = (ranked == qg)
        n_relevant = relevant.sum()
        if n_relevant == 0:
            continue
        cum_relevant = np.cumsum(relevant).astype(np.float64)
        precision_at_rank = cum_relevant / np.arange(1, len(relevant) + 1)
        aps[i] = (precision_at_rank * relevant).sum() / n_relevant
    return aps


def compute_random_map(
    query_groups: List[str],
    support_groups: List[str],
) -> Dict[str, float]:
    S = len(support_groups)
    support_counts = Counter(support_groups)

    per_group: Dict[str, List[float]] = defaultdict(list)
    all_ap: List[float] = []
    for qg in query_groups:
        p = support_counts.get(qg, 0) / S
        all_ap.append(p)
        per_group[qg].append(p)

    out = {"overall": float(np.mean(all_ap))}
    for g in sorted(per_group):
        out[g] = float(np.mean(per_group[g]))
    return out


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = sample.mean()

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(boot_means, alpha)),
        "ci_upper": float(np.quantile(boot_means, 1.0 - alpha)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main retrieval pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_retrieval(args: argparse.Namespace) -> Dict:
    setup_distributed()
    rank = get_rank()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    args.checkpoint = resolve_retrieval_checkpoint(args.ckpt_type, args.checkpoint)

    # ── 1. Build two datasets (small crop + large crop) ──────────────────────
    if is_main_process():
        print(f"Building small-crop dataset (roi={args.roi_size_small}) …")
    dataset_small = RadioRetrievalDataset(
        annotations_csv=args.annotations_csv,
        metadata_csv=args.metadata_csv,
        pixdim=args.pixdim,
        crop=True,
        roi_size=args.roi_size_small,
        pad_size=args.roi_size_small,
    )
    if is_main_process():
        print(f"Building large-crop dataset (roi={args.roi_size_large}) …")
    dataset_large = RadioRetrievalDataset(
        annotations_csv=args.annotations_csv,
        metadata_csv=args.metadata_csv,
        pixdim=args.pixdim,
        crop=True,
        roi_size=args.roi_size_large,
        pad_size=args.roi_size_large,
    )

    # ── 2. Resolve T-stage for every sample ──────────────────────────────────
    case_ids_all = dataset_small.annotations["Case ID"].tolist()
    t_stages_all: List[str] = []
    valid_mask: List[bool] = []
    for cid in case_ids_all:
        if cid in dataset_small.metadata_lookup.index:
            stage = str(dataset_small.metadata_lookup.loc[cid, "Pathological T stage"]).strip()
        else:
            stage = ""
        t_stages_all.append(stage)
        valid_mask.append(stage in T_STAGE_TO_GROUP)

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_case_ids = [case_ids_all[i] for i in valid_indices]
    valid_t_stages = [t_stages_all[i] for i in valid_indices]
    valid_groups = [T_STAGE_TO_GROUP[s] for s in valid_t_stages]

    if is_main_process():
        print(f"Samples with valid T-stage: {len(valid_indices)} / {len(case_ids_all)}")
        print(f"T-stage distribution: {Counter(valid_t_stages)}")
        print(f"Group distribution:   {Counter(valid_groups)}")

    # ── 3. Stratified 50/50 split (stratify by individual T-stage) ───────────
    q_local, s_local = stratified_split(valid_case_ids, valid_t_stages, seed=args.seed)

    query_indices = [valid_indices[i] for i in q_local]
    support_indices = [valid_indices[i] for i in s_local]
    query_stages = [valid_t_stages[i] for i in q_local]
    support_stages = [valid_t_stages[i] for i in s_local]
    query_groups = [T_STAGE_TO_GROUP[s] for s in query_stages]
    support_groups = [T_STAGE_TO_GROUP[s] for s in support_stages]

    if is_main_process():
        print(f"\nQuery  set: {len(query_indices)}  T-stages: {Counter(query_stages)}")
        print(f"Support set: {len(support_indices)}  T-stages: {Counter(support_stages)}")

    # ── 4. Load model ────────────────────────────────────────────────────────
    if is_main_process():
        print("\nLoading model …")
    model = load_vision_model(
        args.checkpoint, device=device, ckpt_type=args.ckpt_type,
        use_hier=args.use_hier,
        interaction_indices=args.interaction_indices,
    )

    # ── 5. Two-pass feature extraction ───────────────────────────────────────
    if is_main_process():
        print(f"\n── Pass 1: small crop (roi={args.roi_size_small}) ──")
        print("  Extracting query features …")
    query_feats_small, query_cids = extract_features(
        model, dataset_small, query_indices, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if is_main_process():
        print("  Extracting support features …")
    support_feats_small, support_cids = extract_features(
        model, dataset_small, support_indices, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    if is_main_process():
        print(f"\n── Pass 2: large crop (roi={args.roi_size_large}) ──")
        print("  Extracting query features …")
    query_feats_large, _ = extract_features(
        model, dataset_large, query_indices, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if is_main_process():
        print("  Extracting support features …")
    support_feats_large, _ = extract_features(
        model, dataset_large, support_indices, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ── 6. Reciprocal Rank Fusion ────────────────────────────────────────────
    sim_small = query_feats_small @ support_feats_small.T       # [Q, S]
    sim_large = query_feats_large @ support_feats_large.T       # [Q, S]

    fused_scores = reciprocal_rank_fusion(
        [sim_small, sim_large], rrf_k=args.rrf_k,
    )

    if is_main_process():
        print(f"\nRRF fusion complete (k={args.rrf_k}), score matrix shape: {fused_scores.shape}")

    # ── 7. Compute metrics (rank 0 only) ─────────────────────────────────────
    results: Dict = {}
    if is_main_process():
        k_values = [1, 3, 5]
        results = {
            "config": {
                "checkpoint": args.checkpoint,
                "ckpt_type": args.ckpt_type,
                "use_hier": args.use_hier,
                "interaction_indices": args.interaction_indices,
                "roi_size_small": args.roi_size_small,
                "roi_size_large": args.roi_size_large,
                "rrf_k": args.rrf_k,
                "pixdim": args.pixdim,
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
            },
            "split": {
                "n_query": len(query_indices),
                "n_support": len(support_indices),
                "query_t_stage_dist": dict(Counter(query_stages)),
                "support_t_stage_dist": dict(Counter(support_stages)),
                "query_group_dist": dict(Counter(query_groups)),
                "support_group_dist": dict(Counter(support_groups)),
            },
            "metrics": {},
            "random_baseline": {},
            "per_query": [],
        }

        # ── Recall@K ─────────────────────────────────────────────────────────
        print("\n── Recall@K ──")
        for k in k_values:
            effective_k = min(k, len(support_indices))
            hits = compute_recall_at_k(
                fused_scores, query_groups, support_groups, k=effective_k,
            )
            stats = bootstrap_ci(hits, n_bootstrap=args.n_bootstrap, seed=args.seed)
            per_group = compute_per_group_recall(hits, query_groups)
            rand_bl = compute_random_baseline(query_groups, support_groups, k=effective_k)

            results["metrics"][f"recall@{k}"] = {
                "overall": stats,
                "per_group": per_group,
            }
            results["random_baseline"][f"recall@{k}"] = rand_bl

            print(
                f"  Recall@{k:>2d}:  {stats['mean']:.4f}  "
                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]  "
                f"(random: {rand_bl['overall']:.4f})"
            )
            for g in sorted(per_group):
                rg = rand_bl.get(g, 0.0)
                print(f"    {g:>14s}: {per_group[g]:.4f}  (random: {rg:.4f})")

        # ── mAP ──────────────────────────────────────────────────────────────
        print("\n── Mean Average Precision ──")
        ap_values = compute_map(fused_scores, query_groups, support_groups)
        map_stats = bootstrap_ci(ap_values, n_bootstrap=args.n_bootstrap, seed=args.seed)

        groups_arr = np.array(query_groups)
        map_per_group: Dict[str, Dict[str, float]] = {}
        for g in sorted(set(query_groups)):
            mask = groups_arr == g
            if mask.any():
                map_per_group[g] = bootstrap_ci(
                    ap_values[mask], n_bootstrap=args.n_bootstrap, seed=args.seed
                )

        rand_map = compute_random_map(query_groups, support_groups)

        results["metrics"]["mAP"] = {
            "overall": map_stats,
            "per_group": {g: v for g, v in map_per_group.items()},
        }
        results["random_baseline"]["mAP"] = rand_map

        print(
            f"  mAP:     {map_stats['mean']:.4f}  "
            f"[{map_stats['ci_lower']:.4f}, {map_stats['ci_upper']:.4f}]  "
            f"(random: {rand_map['overall']:.4f})"
        )
        for g in sorted(map_per_group):
            gs = map_per_group[g]
            rg = rand_map.get(g, 0.0)
            print(
                f"    {g:>14s}: {gs['mean']:.4f}  "
                f"[{gs['ci_lower']:.4f}, {gs['ci_upper']:.4f}]  "
                f"(random: {rg:.4f})"
            )

        # ── Precision@K ───────────────────────────────────────────────────────
        prec_k_values = [1, 3, 5]
        print("\n── Precision@K ──")
        prec_per_query: Dict[str, np.ndarray] = {}
        for k in prec_k_values:
            effective_k = min(k, len(support_indices))
            prec_arr = compute_precision_at_k(
                fused_scores, query_groups, support_groups, k=effective_k,
            )
            prec_per_query[f"prec@{k}"] = prec_arr
            stats = bootstrap_ci(prec_arr, n_bootstrap=args.n_bootstrap, seed=args.seed)
            per_group_prec: Dict[str, float] = {}
            for g in sorted(set(query_groups)):
                mask = groups_arr == g
                per_group_prec[g] = float(prec_arr[mask].mean()) if mask.any() else 0.0
            rand_prec = compute_random_precision(
                query_groups, support_groups, k=effective_k,
            )

            results["metrics"][f"precision@{k}"] = {
                "overall": stats,
                "per_group": per_group_prec,
            }
            results["random_baseline"][f"precision@{k}"] = rand_prec

            print(
                f"  Prec@{k:>2d}:   {stats['mean']:.4f}  "
                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]  "
                f"(random: {rand_prec['overall']:.4f})"
            )
            for g in sorted(per_group_prec):
                rg = rand_prec.get(g, 0.0)
                print(f"    {g:>14s}: {per_group_prec[g]:.4f}  (random: {rg:.4f})")

        # ── Per-query detail ─────────────────────────────────────────────────
        largest_k = min(max(k_values + prec_k_values), len(support_indices))
        sorted_all = _sorted_indices(fused_scores)                  # [Q, S]
        support_groups_arr = np.array(support_groups)

        for i in range(len(query_indices)):
            topk_idx = sorted_all[i, :largest_k]
            retrieved_groups = support_groups_arr[topk_idx].tolist()
            entry = {
                "case_id": query_cids[i],
                "t_stage": query_stages[i],
                "group": query_groups[i],
                "AP": float(ap_values[i]),
                "top_retrieved_case_ids": [support_cids[j] for j in sorted_all[i, :10]],
                "top_retrieved_groups": support_groups_arr[sorted_all[i, :10]].tolist(),
            }
            for k in k_values:
                entry[f"hit@{k}"] = bool(query_groups[i] in retrieved_groups[:k])
            for k in prec_k_values:
                entry[f"prec@{k}"] = float(prec_per_query[f"prec@{k}"][i])
            results["per_query"].append(entry)

        # ── 8. Save ──────────────────────────────────────────────────────────
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "retrieval_results.json"
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
        description="Rank-fusion T-stage retrieval on RADIO with FlexiMedDINOv3_VLM"
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
        help="Checkpoint type: 'vlm' (full VLM) or 'teacher' (backbone only)",
    )
    p.add_argument(
        "--use_hier", action="store_true", default=False,
        help="Use hierarchical features from intermediate blocks (bypasses projection)",
    )
    p.add_argument(
        "--interaction_indices", type=int, nargs="+", default=[3, 7, 11, 15],
        help="0-indexed block indices to extract when --use_hier is set",
    )
    p.add_argument(
        "--output_dir", type=str,
        default=str(ROOT / "results/retrieval/radio_T_stage/ours_rank_fusion"),
        help="Directory to save retrieval_results.json",
    )
    p.add_argument(
        "--annotations_csv", type=str, default=ANNOTATIONS_CSV, help="Path to annotations CSV",
    )
    p.add_argument(
        "--metadata_csv", type=str, default=METADATA_CSV, help="Path to metadata CSV",
    )
    p.add_argument("--roi_size_small", type=int, default=32,
                   help="Cubic crop size for the small-scale pass")
    p.add_argument("--roi_size_large", type=int, default=64,
                   help="Cubic crop size for the large-scale pass")
    p.add_argument("--rrf_k", type=int, default=60,
                   help="Constant k for Reciprocal Rank Fusion (default 60)")
    p.add_argument("--pixdim", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for parallel I/O + preprocessing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    run_retrieval(parse_args())
