"""Run data-efficiency linear-probe sweeps from cached Flexi_CT features."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_DIR = str(ROOT / "features" / "2d_classify")
DEFAULT_OUTPUT_CSV = str(ROOT / "results" / "2d_classify" / "data_efficiency_results.csv")

DATASETS = ["kits", "deep-lesion", "covidx"]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
LR_SWEEP = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
N_BOOTSTRAP = 10_000
SEED = 42

DATASET_HPARAMS = {
    "kits": {"epochs": 200, "batch_size": 64},
    "deep-lesion": {"epochs": 200, "batch_size": 64},
    "covidx": {"epochs": 200, "batch_size": 64},
}


def _parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    return [float(item) for item in value.split(",") if item.strip()]


def _auc(proba: np.ndarray, labels: np.ndarray) -> float:
    n_classes = proba.shape[1]
    try:
        if n_classes == 2:
            return float(roc_auc_score(labels, proba[:, 1]))
        return float(roc_auc_score(labels, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _compute_metrics(proba: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = proba.argmax(axis=1)
    return {
        "auc": _auc(proba, labels),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
    }


def bootstrap_ci(
    proba: np.ndarray,
    labels: np.ndarray,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    auc_boots = []
    bal_acc_boots = []
    n_samples = len(labels)

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        b_labels = labels[idx]
        if len(np.unique(b_labels)) < 2:
            continue
        metrics = _compute_metrics(proba[idx], b_labels)
        if not np.isnan(metrics["auc"]):
            auc_boots.append(metrics["auc"])
            bal_acc_boots.append(metrics["bal_acc"])

    def _ci(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    auc_lo, auc_hi = _ci(auc_boots)
    bal_acc_lo, bal_acc_hi = _ci(bal_acc_boots)
    return {
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
        "bal_acc_lo": bal_acc_lo,
        "bal_acc_hi": bal_acc_hi,
    }


def _aggregate_by_case(
    proba: np.ndarray,
    labels: np.ndarray,
    case_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_labels = {}
    for idx, case_id in enumerate(case_ids):
        key = str(case_id)
        grouped_probs[key].append(proba[idx])
        grouped_labels[key] = int(labels[idx])

    keys = list(grouped_probs)
    case_probs = np.stack([np.mean(grouped_probs[key], axis=0) for key in keys])
    case_labels = np.asarray([grouped_labels[key] for key in keys], dtype=np.int64)
    return case_probs, case_labels


def train_linear_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    lr: float,
    epochs: int,
    batch_size: int,
    test_case_ids: np.ndarray | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray] | None:
    model = nn.Linear(x_train.shape[-1], n_classes).to(device)
    nn.init.normal_(model.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.bias)

    optimizer = SGD(model.parameters(), lr=lr * batch_size / 256.0, momentum=0.9)
    steps_per_epoch = max(1, len(x_train) // batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=0)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)

    best_auc = -1.0
    best_result = None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(x_test_t), dim=-1).cpu().numpy()
        eval_proba, eval_labels = (
            _aggregate_by_case(proba, y_test, test_case_ids)
            if test_case_ids is not None
            else (proba, y_test)
        )
        auc = _auc(eval_proba, eval_labels)
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_result = (eval_proba, eval_labels)

    return best_result


def run_sweep(
    features_dir: str,
    output_csv: str,
    probs_dir: str | None,
    datasets: list[str],
    fractions: list[float],
    lrs: list[float],
    n_bootstrap: int,
    epochs_override: int | None,
) -> pd.DataFrame:
    features_root = Path(features_dir)
    probs_root = Path(probs_dir) if probs_dir else Path(output_csv).parent / "sweep_probabilities"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for dataset in datasets:
        feat_dir = features_root / "flexi" / dataset
        paths = {
            "train_features": feat_dir / "train_features.npy",
            "train_labels": feat_dir / "train_labels.npy",
            "test_features": feat_dir / "test_features.npy",
            "test_labels": feat_dir / "test_labels.npy",
        }
        if not all(path.exists() for path in paths.values()):
            print(f"[skip] flexi/{dataset}: cached features not found in {feat_dir}")
            continue

        x_train = np.load(paths["train_features"])
        y_train = np.load(paths["train_labels"])
        x_test = np.load(paths["test_features"])
        y_test = np.load(paths["test_labels"])
        case_path = feat_dir / "test_case_ids.npy"
        test_case_ids = np.load(case_path, allow_pickle=True) if case_path.exists() else None

        n_classes = len(np.unique(y_train))
        hp = DATASET_HPARAMS[dataset]
        epochs = epochs_override or hp["epochs"]
        batch_size = hp["batch_size"]
        print(f"\nflexi/{dataset}: train={len(y_train)} test={len(y_test)} classes={n_classes}")

        for frac in fractions:
            min_samples = max(2, n_classes)
            if int(len(y_train) * frac) < min_samples:
                print(f"  [skip] frac={frac:.0%}: too few samples")
                continue

            if frac < 1.0:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=SEED)
                idx, _ = next(splitter.split(x_train, y_train))
                x_sub = x_train[idx]
                y_sub = y_train[idx]
            else:
                x_sub = x_train
                y_sub = y_train

            best_lr = None
            best_auc = -1.0
            best_eval = None
            for lr in lrs:
                eval_result = train_linear_head(
                    x_train=x_sub,
                    y_train=y_sub,
                    x_test=x_test,
                    y_test=y_test,
                    n_classes=n_classes,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    test_case_ids=test_case_ids,
                    device=device,
                )
                if eval_result is None:
                    continue
                eval_proba, eval_labels = eval_result
                auc = _auc(eval_proba, eval_labels)
                print(f"    frac={frac:.0%} lr={lr} auc={auc:.4f}")
                if not np.isnan(auc) and auc > best_auc:
                    best_lr = lr
                    best_auc = auc
                    best_eval = eval_result

            if best_eval is None:
                continue

            eval_proba, eval_labels = best_eval
            metrics = _compute_metrics(eval_proba, eval_labels)
            ci = bootstrap_ci(eval_proba, eval_labels, n=n_bootstrap)
            probs_path = probs_root / "flexi" / dataset / f"frac_{frac:.2f}.npz"
            probs_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(probs_path, probs=eval_proba, labels=eval_labels, best_lr=best_lr)

            row = {
                "model": "flexi",
                "dataset": dataset,
                "fraction": frac,
                "auc": metrics["auc"],
                "auc_lo": ci["auc_lo"],
                "auc_hi": ci["auc_hi"],
                "bal_acc": metrics["bal_acc"],
                "bal_acc_lo": ci["bal_acc_lo"],
                "bal_acc_hi": ci["bal_acc_hi"],
            }
            results.append(row)
            print(
                f"  ** frac={frac:.0%} best_lr={best_lr} "
                f"auc={row['auc']:.4f} bal_acc={row['bal_acc']:.4f}"
            )

    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows -> {output_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--probs_dir", default=None)
    parser.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    parser.add_argument("--fractions", default=None, help="Comma-separated override, e.g. 0.1,1.0")
    parser.add_argument("--lrs", default=None, help="Comma-separated override, e.g. 0.01,0.1")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--epochs", type=int, default=None, help="Debug override for all datasets.")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    run_sweep(
        features_dir=args.features_dir,
        output_csv=args.output_csv,
        probs_dir=args.probs_dir,
        datasets=datasets,
        fractions=_parse_float_list(args.fractions, FRACTIONS),
        lrs=_parse_float_list(args.lrs, LR_SWEEP),
        n_bootstrap=args.n_bootstrap,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
