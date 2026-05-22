#!/usr/bin/env python
"""
Sensitivity analysis for segmentation evaluation CSVs.

This script does three things for each metric group (for example Dice, SD):
1. Reports the actual mean, bootstrap CI, and paired permutation-test p-value
   between --csv and --ref_csv.
2. Computes the smallest *hypothetical uniform uplift* added to --csv that
   would make the paired permutation test fall below alpha, using a fixed seed
   for reproducibility.
3. Optionally evaluates a user-specified hypothetical uniform uplift and
   reports the resulting mean, CI, and p-value.

The uplifted summaries are explicitly hypothetical thresholds, not observed
results. They answer "how much would the evaluated system need to move, under
this simple shift model, before the same test becomes significant?"

Examples
--------
python try.py --csv evaluation.csv --ref_csv baseline.csv
python try.py --csv evaluation.csv --ref_csv baseline.csv --metric Dice
python try.py --csv evaluation.csv --ref_csv baseline.csv --uplift 0.01
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from permutation import bootstrap_ci, group_columns_by_metric, load_metrics, permutation_test_paired


def get_common_data(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Align two dataframes on shared case IDs and metric columns."""
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError("No overlapping case IDs between the two CSVs.")

    common_cols = df_a.columns.intersection(df_b.columns)
    if len(common_cols) == 0:
        raise ValueError("No overlapping metric columns between the two CSVs.")

    df_a = df_a.loc[common, common_cols]
    df_b = df_b.loc[common, common_cols]
    groups = group_columns_by_metric(common_cols)
    return df_a, df_b, groups


def find_min_uniform_shift(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    alpha: float,
    n_permutation: int,
    seed: int,
    max_shift: float,
    tol: float,
    max_iter: int = 40,
) -> tuple[float | None, dict | None]:
    """
    Find the smallest non-negative constant delta such that the paired
    permutation test on (vals_a + delta, vals_b) has p < alpha.

    Returns (delta, test_result). If significance is not reached within
    max_shift, returns (None, None).
    """
    actual = permutation_test_paired(
        vals_a,
        vals_b,
        n_permutation=n_permutation,
        rng=np.random.default_rng(seed),
    )
    if np.isnan(actual["p_value"]):
        return None, None
    if actual["p_value"] < alpha:
        return 0.0, actual

    diffs = vals_a - vals_b
    scale = np.nanstd(diffs)
    if not np.isfinite(scale) or scale <= 0:
        scale = abs(np.nanmean(diffs))
    if not np.isfinite(scale) or scale <= 0:
        scale = 0.01

    low = 0.0
    high = max(scale, 1e-6)
    high_result = None

    while high <= max_shift + tol:
        high_result = permutation_test_paired(
            vals_a + high,
            vals_b,
            n_permutation=n_permutation,
            rng=np.random.default_rng(seed),
        )
        if high_result["p_value"] < alpha:
            break
        low = high
        high *= 2.0

    if high_result is None or high_result["p_value"] >= alpha:
        return None, None

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        mid_result = permutation_test_paired(
            vals_a + mid,
            vals_b,
            n_permutation=n_permutation,
            rng=np.random.default_rng(seed),
        )
        if mid_result["p_value"] < alpha:
            high = mid
            high_result = mid_result
        else:
            low = mid
        if high - low <= tol:
            break

    return high, high_result


def build_records(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    groups: dict[str, list[str]],
    n_bootstrap: int,
    n_permutation: int,
    ci: float,
    alpha: float,
    seed: int,
    max_shift: float,
    tol: float,
    uplift: float | None,
    metric_filter: str | None,
) -> pd.DataFrame:
    """Compute actual and hypothetical summaries for each metric group."""
    records = []

    prefixes = list(groups.keys())
    if metric_filter is not None:
        metric_filter = metric_filter.removeprefix("mean_")
        prefixes = [p for p in prefixes if p == metric_filter]
        if not prefixes:
            raise ValueError(f"Metric group '{metric_filter}' not found. Available: {sorted(groups)}")

    for idx, prefix in enumerate(prefixes):
        cols = groups[prefix]
        mean_a = df_a[cols].mean(axis=1).values
        mean_b = df_b[cols].mean(axis=1).values

        actual_test = permutation_test_paired(
            mean_a,
            mean_b,
            n_permutation=n_permutation,
            rng=np.random.default_rng(seed + idx * 100 + 1),
        )
        actual_ci_a = bootstrap_ci(
            mean_a,
            n_bootstrap=n_bootstrap,
            ci=ci,
            rng=np.random.default_rng(seed + idx * 100 + 2),
        )
        actual_ci_b = bootstrap_ci(
            mean_b,
            n_bootstrap=n_bootstrap,
            ci=ci,
            rng=np.random.default_rng(seed + idx * 100 + 3),
        )

        delta, shifted_test = find_min_uniform_shift(
            mean_a,
            mean_b,
            alpha=alpha,
            n_permutation=n_permutation,
            seed=seed + idx * 100 + 4,
            max_shift=max_shift,
            tol=tol,
        )

        if uplift is None:
            user_uplift_diff = np.nan
            user_uplift_p = np.nan
            user_uplift_mean = np.nan
            user_uplift_low = np.nan
            user_uplift_high = np.nan
        else:
            uplift_test = permutation_test_paired(
                mean_a + uplift,
                mean_b,
                n_permutation=n_permutation,
                rng=np.random.default_rng(seed + idx * 100 + 5),
            )
            user_uplift_diff = uplift_test["observed_diff"]
            user_uplift_p = uplift_test["p_value"]
            user_uplift_mean = actual_ci_a["mean"] + uplift
            user_uplift_low = actual_ci_a["low_CI"] + uplift
            user_uplift_high = actual_ci_a["up_CI"] + uplift

        if delta is None:
            threshold_mean = np.nan
            threshold_low = np.nan
            threshold_high = np.nan
            threshold_p = np.nan
            threshold_diff = np.nan
            reached = False
        else:
            threshold_mean = actual_ci_a["mean"] + delta
            threshold_low = actual_ci_a["low_CI"] + delta
            threshold_high = actual_ci_a["up_CI"] + delta
            threshold_p = shifted_test["p_value"]
            threshold_diff = shifted_test["observed_diff"]
            reached = True

        records.append(
            {
                "metric": f"mean_{prefix}",
                "n_cases": actual_test["n_cases"],
                "actual_mean_A": actual_ci_a["mean"],
                "actual_low_CI_A": actual_ci_a["low_CI"],
                "actual_up_CI_A": actual_ci_a["up_CI"],
                "actual_mean_B": actual_ci_b["mean"],
                "actual_low_CI_B": actual_ci_b["low_CI"],
                "actual_up_CI_B": actual_ci_b["up_CI"],
                "actual_diff_A_minus_B": actual_test["observed_diff"],
                "actual_p_value": actual_test["p_value"],
                "actual_significant": bool(actual_test["p_value"] < alpha),
                "min_uniform_uplift_for_p_lt_alpha": delta,
                "threshold_hypothetical_mean_A": threshold_mean,
                "threshold_hypothetical_low_CI_A": threshold_low,
                "threshold_hypothetical_up_CI_A": threshold_high,
                "threshold_hypothetical_diff_A_minus_B": threshold_diff,
                "threshold_hypothetical_p_value": threshold_p,
                "threshold_hypothetical_found": reached,
                "specified_uniform_uplift": uplift,
                "specified_hypothetical_mean_A": user_uplift_mean,
                "specified_hypothetical_low_CI_A": user_uplift_low,
                "specified_hypothetical_up_CI_A": user_uplift_high,
                "specified_hypothetical_diff_A_minus_B": user_uplift_diff,
                "specified_hypothetical_p_value": user_uplift_p,
            }
        )

    return pd.DataFrame(records)


def fmt_float(value: float, digits: int = 6) -> str:
    """Format floats consistently for terminal output."""
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_bool(value: bool) -> str:
    """Render booleans as readable text."""
    return "yes" if value else "no"


def print_readable_summary(out: pd.DataFrame, ci: float, alpha: float, uplift: float | None) -> None:
    """Print grouped, human-readable results for the terminal."""
    print("=== Actual Results and Hypothetical Sensitivity Analysis ===")
    print("A = --csv, B = --ref_csv")
    print(f"Bootstrap CI: {ci:.1f}%")
    print(f"Permutation alpha: {alpha:.3f}")
    if uplift is not None:
        print(f"Specified hypothetical uplift: {uplift:.6f}")
    print()

    for _, row in out.iterrows():
        print(row["metric"])
        print(
            "  actual A: "
            f"mean={fmt_float(row['actual_mean_A'])}, "
            f"CI=[{fmt_float(row['actual_low_CI_A'])}, {fmt_float(row['actual_up_CI_A'])}]"
        )
        print(
            "  actual B: "
            f"mean={fmt_float(row['actual_mean_B'])}, "
            f"CI=[{fmt_float(row['actual_low_CI_B'])}, {fmt_float(row['actual_up_CI_B'])}]"
        )
        print(
            "  actual test: "
            f"diff={fmt_float(row['actual_diff_A_minus_B'])}, "
            f"p={fmt_float(row['actual_p_value'])}, "
            f"significant={fmt_bool(bool(row['actual_significant']))}, "
            f"n={int(row['n_cases'])}"
        )

        if bool(row["threshold_hypothetical_found"]):
            print(
                "  min uplift for p < alpha: "
                f"delta={fmt_float(row['min_uniform_uplift_for_p_lt_alpha'])}, "
                f"mean={fmt_float(row['threshold_hypothetical_mean_A'])}, "
                f"CI=[{fmt_float(row['threshold_hypothetical_low_CI_A'])}, "
                f"{fmt_float(row['threshold_hypothetical_up_CI_A'])}], "
                f"diff={fmt_float(row['threshold_hypothetical_diff_A_minus_B'])}, "
                f"p={fmt_float(row['threshold_hypothetical_p_value'])}"
            )
        else:
            print("  min uplift for p < alpha: not found within search range")

        if uplift is not None:
            print(
                "  specified uplift result: "
                f"delta={fmt_float(row['specified_uniform_uplift'])}, "
                f"mean={fmt_float(row['specified_hypothetical_mean_A'])}, "
                f"CI=[{fmt_float(row['specified_hypothetical_low_CI_A'])}, "
                f"{fmt_float(row['specified_hypothetical_up_CI_A'])}], "
                f"diff={fmt_float(row['specified_hypothetical_diff_A_minus_B'])}, "
                f"p={fmt_float(row['specified_hypothetical_p_value'])}"
            )

        print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report actual mean/CI/p-values from two evaluation CSVs and the minimum "
            "hypothetical uniform uplift needed for p < alpha."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to the evaluation CSV.")
    parser.add_argument("--ref_csv", required=True, help="Path to the baseline/reference CSV.")
    parser.add_argument(
        "--metric",
        default=None,
        help="Optional metric group prefix to analyze (for example Dice or mean_Dice).",
    )
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--n_permutation", type=int, default=10_000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uplift",
        type=float,
        default=None,
        help="Optional user-specified hypothetical uniform uplift to add to --csv.",
    )
    parser.add_argument(
        "--max_shift",
        type=float,
        default=1.0,
        help="Maximum hypothetical uplift to search before giving up (default: 1.0).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Binary-search tolerance for the hypothetical uplift threshold.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save results CSV.")
    args = parser.parse_args()

    df_a = load_metrics(args.csv)
    df_b = load_metrics(args.ref_csv)
    df_a, df_b, groups = get_common_data(df_a, df_b)

    out = build_records(
        df_a=df_a,
        df_b=df_b,
        groups=groups,
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        ci=args.ci,
        alpha=args.alpha,
        seed=args.seed,
        max_shift=args.max_shift,
        tol=args.tol,
        uplift=args.uplift,
        metric_filter=args.metric,
    )

    print_readable_summary(out, ci=args.ci, alpha=args.alpha, uplift=args.uplift)

    if args.output:
        output_path = Path(args.output)
        out.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
