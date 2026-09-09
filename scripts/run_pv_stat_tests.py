"""Run paired statistical tests for Online Lyra against each baseline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini--Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return adjusted

    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    scale = len(ranked) / np.arange(1, len(ranked) + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    restored = np.empty_like(ranked_adjusted)
    restored[order] = ranked_adjusted
    adjusted[finite] = restored
    return adjusted


def try_scipy_tests(online: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    try:
        from scipy import stats

        diff = online - other
        if np.allclose(diff, 0.0):
            return 1.0, 1.0
        wilcoxon_p = float(stats.wilcoxon(online, other, alternative="less").pvalue)
        ttest_p = float(stats.ttest_rel(online, other, alternative="less").pvalue)
        return wilcoxon_p, ttest_p
    except Exception:
        return float("nan"), float("nan")


def sign_test_pvalue(online: np.ndarray, other: np.ndarray) -> float:
    diff = online - other
    wins = int(np.sum(diff < 0))
    losses = int(np.sum(diff > 0))
    n = wins + losses
    if n == 0:
        return 1.0
    # One-sided binomial p-value: P(X >= wins), p=0.5.
    p = 0.0
    for k in range(wins, n + 1):
        p += math.comb(n, k) * (0.5 ** n)
    return float(p)


def run_tests(condition_csv: Path, output_dir: Path, online_model: str = "online_lyra_b06_final") -> pd.DataFrame:
    df = pd.read_csv(condition_csv)
    metrics = ["nrmse", "nmae", "daytime_nrmse", "daytime_nmae"]
    keys = ["farm", "horizon"]
    online = df[df["model"].eq(online_model)][keys + metrics].copy()
    rows = []
    for model in sorted(m for m in df["model"].unique() if m != online_model):
        other = df[df["model"].eq(model)][keys + metrics + ["model_display"]].copy()
        merged = online.merge(other, on=keys, suffixes=("_online", "_baseline"))
        if merged.empty:
            continue
        display = str(merged["model_display"].iloc[0])
        for metric in metrics:
            o = merged[f"{metric}_online"].to_numpy(dtype=float)
            b = merged[f"{metric}_baseline"].to_numpy(dtype=float)
            wilcoxon_p, ttest_p = try_scipy_tests(o, b)
            rows.append(
                {
                    "baseline_model": model,
                    "baseline_display": display,
                    "metric": metric,
                    "n_pairs": len(merged),
                    "online_mean": float(np.mean(o)),
                    "baseline_mean": float(np.mean(b)),
                    "mean_delta_online_minus_baseline": float(np.mean(o - b)),
                    "relative_improvement_pct": float((np.mean(b) - np.mean(o)) / max(np.mean(b), 1e-12) * 100.0),
                    "online_wins": int(np.sum(o < b)),
                    "online_ties": int(np.sum(np.isclose(o, b))),
                    "online_losses": int(np.sum(o > b)),
                    "win_rate": float(np.mean(o < b)),
                    "wilcoxon_p_less": wilcoxon_p,
                    "paired_ttest_p_less": ttest_p,
                    "sign_test_p_less": sign_test_pvalue(o, b),
                }
            )
    out = pd.DataFrame(rows).sort_values(["metric", "baseline_mean", "baseline_display"])
    out["wilcoxon_fdr_bh"] = np.nan
    for metric, indices in out.groupby("metric").groups.items():
        out.loc[indices, "wilcoxon_fdr_bh"] = benjamini_hochberg(
            out.loc[indices, "wilcoxon_p_less"].to_numpy(dtype=float)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "paired_tests_online_lyra_vs_baselines.csv", index=False)

    site_df = (
        df.groupby(["model", "model_display", "farm"], as_index=False)[metrics]
        .mean()
    )
    site_online = site_df[site_df["model"].eq(online_model)][["farm", *metrics]].copy()
    site_rows = []
    for model in sorted(m for m in site_df["model"].unique() if m != online_model):
        other = site_df[site_df["model"].eq(model)][
            ["farm", *metrics, "model_display"]
        ].copy()
        merged = site_online.merge(other, on="farm", suffixes=("_online", "_baseline"))
        if merged.empty:
            continue
        display = str(merged["model_display"].iloc[0])
        for metric in metrics:
            o = merged[f"{metric}_online"].to_numpy(dtype=float)
            b = merged[f"{metric}_baseline"].to_numpy(dtype=float)
            wilcoxon_p, ttest_p = try_scipy_tests(o, b)
            site_rows.append(
                {
                    "baseline_model": model,
                    "baseline_display": display,
                    "metric": metric,
                    "n_sites": len(merged),
                    "online_mean": float(np.mean(o)),
                    "baseline_mean": float(np.mean(b)),
                    "mean_delta_online_minus_baseline": float(np.mean(o - b)),
                    "relative_improvement_pct": float(
                        (np.mean(b) - np.mean(o)) / max(np.mean(b), 1e-12) * 100.0
                    ),
                    "online_wins": int(np.sum(o < b)),
                    "online_ties": int(np.sum(np.isclose(o, b))),
                    "online_losses": int(np.sum(o > b)),
                    "win_rate": float(np.mean(o < b)),
                    "wilcoxon_p_less": wilcoxon_p,
                    "paired_ttest_p_less": ttest_p,
                    "sign_test_p_less": sign_test_pvalue(o, b),
                }
            )
    site_out = pd.DataFrame(site_rows).sort_values(
        ["metric", "baseline_mean", "baseline_display"]
    )
    site_out["wilcoxon_fdr_bh"] = np.nan
    for metric, indices in site_out.groupby("metric").groups.items():
        site_out.loc[indices, "wilcoxon_fdr_bh"] = benjamini_hochberg(
            site_out.loc[indices, "wilcoxon_p_less"].to_numpy(dtype=float)
        )
    site_out.to_csv(
        output_dir / "site_blocked_tests_online_lyra_vs_baselines.csv", index=False
    )

    development_mask = df["farm"].astype(str).str.contains(
        "site 1 ", case=False, regex=False
    )
    held_out = df.loc[~development_mask].copy()
    held_out_summary = (
        held_out.groupby(["model", "model_display"], as_index=False)[metrics]
        .mean()
        .sort_values(["nrmse", "nmae"])
    )
    held_out_summary.to_csv(
        output_dir / "held_out_sites_2_to_8_model_summary.csv", index=False
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired tests for PV forecasting results")
    parser.add_argument(
        "--condition_csv",
        type=Path,
        default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425/paper_tables_with_offline_lyra/main_condition_average.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_result_pack/stat_tests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_tests(args.condition_csv, args.output_dir)
    print(out.to_string(index=False))
    print(f"Saved statistical tests to {args.output_dir}")


if __name__ == "__main__":
    main()
