#!/usr/bin/env python
"""
Bootstrap confidence intervals and permutation tests for segmentation evaluation CSVs.

Usage examples:
  # Bootstrap CIs for a single CSV
  python permutation.py --csv evaluation.csv

  # Bootstrap CIs + permutation test against a reference CSV
  python permutation.py --csv evaluation.csv --ref_csv reference_evaluation.csv

  # Custom settings
  python permutation.py --csv evaluation.csv --ref_csv ref.csv \
      --n_bootstrap 10000 --n_permutation 10000 --ci 95 --seed 42 --output results.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load a CSV and return only the numeric metric columns, indexed by case ID.

    Handles CSVs that contain a summary row (e.g. "Overall (mean±std)") or
    columns with mixed numeric/string values by coercing to numeric and
    dropping rows that are entirely non-numeric.
    """
    df = pd.read_csv(csv_path)
    id_candidates = [c for c in df.columns if c.lower() in ("caseid", "case_id", "filename", "file")]
    if id_candidates:
        id_col = id_candidates[0]
    else:
        first_col = df.columns[0]
        id_col = first_col if df[first_col].dtype == object else None

    if id_col is not None:
        df = df.set_index(id_col)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def group_columns_by_metric(columns: pd.Index) -> dict[str, list[str]]:
    """
    Group column names by their metric-type prefix.
    E.g. ['Dice_tumor', 'Dice_mass', 'SD_tumor', 'SD_mass']
      -> {'Dice': ['Dice_tumor', 'Dice_mass'], 'SD': ['SD_tumor', 'SD_mass']}
    The prefix is the part before the first underscore.
    """
    groups: dict[str, list[str]] = {}
    for col in columns:
        prefix = col.split("_", 1)[0]
        groups.setdefault(prefix, []).append(col)
    return groups


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Bootstrap the mean of *values* (1-D array, may contain NaN).
    Returns dict with keys: mean, low_CI, up_CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"mean": np.nan, "low_CI": np.nan, "up_CI": np.nan}

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()

    alpha = (100.0 - ci) / 2.0
    return {
        "mean": float(np.nanmean(values)),
        "low_CI": float(np.percentile(boot_means, alpha)),
        "up_CI": float(np.percentile(boot_means, 100.0 - alpha)),
    }


def permutation_test_paired(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_permutation: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Two-sided paired permutation test on the mean.

    For each case i, under the null the assignment of (a_i, b_i) is random,
    so we flip each pair independently with probability 0.5.

    Returns dict with observed_diff, p_value.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    a = vals_a[mask]
    b = vals_b[mask]
    n = len(a)
    if n == 0:
        return {"observed_diff": np.nan, "p_value": np.nan, "n_cases": 0}

    observed_diff = a.mean() - b.mean()
    diffs = a - b

    count_extreme = 0
    for _ in range(n_permutation):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = (signs * diffs).mean()
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutation + 1)  # +1 correction avoids p=0
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_cases": int(n),
    }


def run_bootstrap(df: pd.DataFrame, n_bootstrap: int, ci: float, seed: int) -> pd.DataFrame:
    """
    Bootstrap CI for every metric column, plus a mean_<prefix> row for each
    metric-type group (e.g. mean_Dice, mean_SD).
    """
    rng = np.random.default_rng(seed)
    records = []
    groups = group_columns_by_metric(df.columns)

    for prefix, cols in groups.items():
        for col in cols:
            res = bootstrap_ci(df[col].values, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            res["metric"] = col
            records.append(res)

        mean_per_case = df[cols].mean(axis=1).values
        res = bootstrap_ci(mean_per_case, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        res["metric"] = f"mean_{prefix}"
        records.append(res)

    out = pd.DataFrame(records)[["metric", "mean", "low_CI", "up_CI"]]
    return out


def run_permutation_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_permutation: int,
    seed: int,
) -> pd.DataFrame:
    """
    Paired permutation test (two-sided) for each mean metric group
    (e.g. mean_Dice, mean_SD) between two CSVs.
    """
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError("No overlapping case IDs between the two CSVs.")
    common_cols = df_a.columns.intersection(df_b.columns)
    if len(common_cols) == 0:
        raise ValueError("No overlapping metric columns between the two CSVs.")

    df_a = df_a.loc[common, common_cols]
    df_b = df_b.loc[common, common_cols]
    rng = np.random.default_rng(seed)
    groups = group_columns_by_metric(common_cols)

    records = []
    for prefix, cols in groups.items():
        mean_a = df_a[cols].mean(axis=1).values
        mean_b = df_b[cols].mean(axis=1).values
        res = permutation_test_paired(mean_a, mean_b, n_permutation=n_permutation, rng=rng)
        res["metric"] = f"mean_{prefix}"
        res["mean_A"] = float(np.nanmean(mean_a))
        res["mean_B"] = float(np.nanmean(mean_b))
        records.append(res)

    out = pd.DataFrame(records)[
        ["metric", "mean_A", "mean_B", "observed_diff", "p_value", "n_cases"]
    ]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs and paired permutation tests for segmentation evaluation CSVs."
    )
    parser.add_argument("--csv", required=True, help="Path to the evaluation CSV.")
    parser.add_argument(
        "--ref_csv",
        default=None,
        help="Path to a reference CSV for permutation test. Must share case IDs with --csv.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--n_permutation", type=int, default=10_000)
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence level (default 95).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional path to save bootstrap results CSV.")
    args = parser.parse_args()

    df = load_metrics(args.csv)
    print(f"Loaded {args.csv}: {len(df)} cases, {len(df.columns)} metric columns")
    print(f"  Columns: {list(df.columns)}\n")

    boot_df = run_bootstrap(df, n_bootstrap=args.n_bootstrap, ci=args.ci, seed=args.seed)
    print(f"=== Bootstrap ({args.n_bootstrap} resamples, {args.ci}% CI) ===")
    print(boot_df.to_string(index=False, float_format="%.6f"))
    print()

    if args.output:
        boot_df.to_csv(args.output, index=False)
        print(f"Bootstrap results saved to {args.output}\n")

    if args.ref_csv is not None:
        df_ref = load_metrics(args.ref_csv)
        print(f"Loaded reference {args.ref_csv}: {len(df_ref)} cases, {len(df_ref.columns)} metric columns")
        print(f"  Columns: {list(df_ref.columns)}\n")

        perm_df = run_permutation_test(df, df_ref, n_permutation=args.n_permutation, seed=args.seed)
        print(f"=== Permutation Test ({args.n_permutation} permutations, two-sided) ===")
        print("  A = --csv,  B = --ref_csv")
        print(perm_df.to_string(index=False, float_format="%.6f"))
        print()

        if args.output:
            perm_path = str(Path(args.output).with_suffix("")) + "_permtest.csv"
            perm_df.to_csv(perm_path, index=False)
            print(f"Permutation test results saved to {perm_path}")


if __name__ == "__main__":
    main()
