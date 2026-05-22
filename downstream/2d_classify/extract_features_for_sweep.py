"""Extract cached Flexi_CT_2D features for 2D classification sweeps."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

DEFAULT_FEATURES_DIR = str(ROOT / "features" / "2d_classify")
DEFAULT_CHECKPOINT = Flexi_CT_2D.DEFAULT_CHECKPOINT
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"

DATASETS = {
    "kits": {
        "hf_name": "raidium/CuriaBench",
        "hf_config": "kits",
        "case_id_col": "series_id",
        "feature_mode": "mask_or_cls",
    },
    "deep-lesion": {
        "hf_name": "raidium/CuriaBench",
        "hf_config": "deep-lesion-site",
        "case_id_col": "series_id",
        "feature_mode": "mask_or_cls",
    },
    "covidx": {
        "hf_name": "raidium/CuriaBench",
        "hf_config": "covidx-ct",
        "case_id_col": None,
        "feature_mode": "cls_patch_mean",
    },
}


def _set_patch_size(model: torch.nn.Module, patch_size: int) -> None:
    for module in model.modules():
        if hasattr(module, "set_patch_size"):
            module.set_patch_size(patch_size)
    if hasattr(model, "patch_size"):
        model.patch_size = patch_size


def _mask_to_patch_grid(mask: object, spatial_dim: int, device: torch.device) -> torch.Tensor | None:
    if mask is None:
        return None

    mask_np = np.asarray(mask)
    if mask_np.size == 0:
        return None
    while mask_np.ndim > 2:
        mask_np = mask_np.max(axis=0)
    if mask_np.ndim != 2:
        return None

    mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
    if torch.count_nonzero(mask_t).item() == 0:
        return None
    mask_t = mask_t.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(mask_t, size=(spatial_dim, spatial_dim), mode="nearest")
    resized = resized.squeeze(0).squeeze(0) > 0
    if torch.count_nonzero(resized).item() == 0:
        return None
    return resized


def _pool_with_optional_masks(
    cls_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    masks: list[object] | None,
) -> torch.Tensor:
    spatial_dim = int(math.sqrt(patch_tokens.shape[1]))
    if spatial_dim * spatial_dim != patch_tokens.shape[1]:
        raise ValueError(f"Patch token count is not square: {patch_tokens.shape[1]}")

    patch_grid = patch_tokens.reshape(
        patch_tokens.shape[0], spatial_dim, spatial_dim, patch_tokens.shape[-1]
    )
    pooled = []
    for idx in range(patch_tokens.shape[0]):
        mask = None if masks is None else _mask_to_patch_grid(masks[idx], spatial_dim, patch_tokens.device)
        if mask is None:
            pooled.append(cls_tokens[idx])
            continue
        selected = patch_grid[idx][mask]
        pooled.append(selected.mean(dim=0) if selected.numel() else cls_tokens[idx])
    return torch.stack(pooled, dim=0)


def _extract_batch_features(
    examples: dict,
    processor: AutoImageProcessor,
    model: Flexi_CT_2D,
    feature_mode: str,
    device: torch.device,
) -> dict:
    images = [
        np.clip(np.asarray(image, dtype=np.float32), -1000.0, 1000.0)
        for image in examples["image"]
    ]
    processed = processor(images, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device)
    if pixel_values.shape[1] == 3:
        pixel_values = pixel_values.mean(dim=1, keepdim=True)

    with torch.no_grad():
        out = model(pixel_values)

    cls_tokens = out["cls_token"]
    patch_tokens = out["patch_tokens"]
    if feature_mode == "cls_patch_mean":
        features = torch.cat([cls_tokens, patch_tokens.mean(dim=1)], dim=1)
    elif feature_mode == "mask_or_cls":
        masks = examples.get("mask")
        features = _pool_with_optional_masks(cls_tokens, patch_tokens, masks)
    elif feature_mode == "cls":
        features = cls_tokens
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    return {
        "pixel_values": features.detach().cpu().numpy().astype(np.float32),
        "labels": np.asarray(examples["target"], dtype=np.int64),
    }


def extract_and_save(
    dataset: str,
    output_dir: str,
    checkpoint: str | None,
    batch_size: int,
    patch_size: int,
) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_flexict_checkpoint("2d", checkpoint)
    ds_cfg = DATASETS[dataset]
    out_path = Path(output_dir) / "flexi" / dataset
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}  Device: {device}  Output: {out_path}")
    print(f"Checkpoint: {checkpoint_path}")
    model = Flexi_CT_2D(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    _set_patch_size(model.backbone, patch_size)

    processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)
    processor.crop_size = 512

    ds = load_dataset(ds_cfg["hf_name"], ds_cfg["hf_config"])
    case_id_col = ds_cfg["case_id_col"]

    for split in ("train", "val", "test"):
        if split not in ds:
            print(f"  [{split}] missing, skipping.")
            continue

        feat_path = out_path / f"{split}_features.npy"
        label_path = out_path / f"{split}_labels.npy"
        case_path = out_path / f"{split}_case_ids.npy"
        if feat_path.exists() and label_path.exists():
            print(f"  [{split}] exists, skipping.")
            continue

        split_ds = ds[split]
        mapped = split_ds.map(
            lambda examples: _extract_batch_features(
                examples=examples,
                processor=processor,
                model=model,
                feature_mode=ds_cfg["feature_mode"],
                device=device,
            ),
            batched=True,
            batch_size=batch_size,
            num_proc=None,
        )
        mapped.set_format("numpy", columns=["pixel_values", "labels"])

        features = np.asarray(mapped["pixel_values"], dtype=np.float32)
        labels = np.asarray(mapped["labels"], dtype=np.int64)
        np.save(feat_path, features)
        np.save(label_path, labels)

        if case_id_col and case_id_col in split_ds.column_names:
            case_ids = np.asarray([str(x) for x in split_ds[case_id_col]])
            np.save(case_path, case_ids)

        print(f"  [{split}] features {features.shape}, labels {labels.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[*DATASETS.keys(), "all"])
    parser.add_argument("--output_dir", default=DEFAULT_FEATURES_DIR)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit Flexi_CT_2D checkpoint. If omitted, uses FLEXICT_CHECKPOINT, "
            "then FLEXICT_2D_CHECKPOINT. No private-host default is bundled."
        ),
    )
    parser.add_argument(
        "--config_dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing copied reference YAML configs for provenance.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        parser.error(f"--config_dir does not exist: {config_dir}")

    datasets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        extract_and_save(
            dataset=dataset,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
        )


if __name__ == "__main__":
    main()
