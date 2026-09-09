"""Summarize Online Lyra tuning runs and optionally compare to static baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def collect_lyra_rows(tuning_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(tuning_root.glob("*/site_*/online_pv_summary.csv")):
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        cfg = summary_path.parents[1].name
        site = summary_path.parent.name
        keep = df[df["model"].isin(["online_lyra_pv", "online_lyra_pv_gated", "trend_only", "persistence"])].copy()
        keep.insert(0, "config", cfg)
        keep.insert(1, "site", site)
        keep.insert(2, "summary_path", str(summary_path))
        rows.append(keep)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def collect_static_baselines(baseline_root: Path) -> pd.DataFrame:
    paths = sorted(baseline_root.glob("static_official_seed_*/pv_baseline_summary.csv"))
    if not paths:
        paths = sorted(baseline_root.glob("**/pv_baseline_summary.csv"))
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        seed = path.parent.name
        df = df.copy()
        df.insert(0, "seed_dir", seed)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_configs(lyra_rows: pd.DataFrame) -> pd.DataFrame:
    target = lyra_rows[lyra_rows["model"].eq("online_lyra_pv_gated")].copy()
    if target.empty:
        return pd.DataFrame()
    agg = (
        target.groupby("config", as_index=False)
        .agg(
            rows=("nrmse", "size"),
            mean_nrmse=("nrmse", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_daytime_nrmse=("daytime_nrmse", "mean"),
            mean_skill_vs_persistence=("skill_vs_persistence", "mean"),
            mean_gate_lyra_fraction=("gate_lyra_fraction", "mean"),
            mean_predict_latency_ms=("predict_latency_ms", "mean"),
            mean_update_latency_ms=("update_latency_ms", "mean"),
        )
        .sort_values(["mean_nrmse", "mean_nmae"])
    )
    return agg


def compare_to_static(lyra_rows: pd.DataFrame, static_rows: pd.DataFrame) -> pd.DataFrame:
    if lyra_rows.empty or static_rows.empty:
        return pd.DataFrame()

    static_mean = (
        static_rows.groupby(["farm", "horizon", "model"], as_index=False)
        .agg(
            static_mean_nrmse=("nrmse", "mean"),
            static_mean_nmae=("nmae", "mean"),
            static_mean_daytime_nrmse=("daytime_nrmse", "mean"),
        )
    )
    best_static = (
        static_mean.sort_values(["farm", "horizon", "static_mean_nrmse"])
        .groupby(["farm", "horizon"], as_index=False)
        .first()
        .rename(
            columns={
                "model": "best_static_model",
                "static_mean_nrmse": "best_static_nrmse",
                "static_mean_nmae": "best_static_nmae",
                "static_mean_daytime_nrmse": "best_static_daytime_nrmse",
            }
        )
    )

    lyra = lyra_rows[lyra_rows["model"].eq("online_lyra_pv_gated")].copy()
    merged = lyra.merge(best_static, on=["farm", "horizon"], how="left")
    merged["delta_nrmse_vs_best_static"] = merged["nrmse"] - merged["best_static_nrmse"]
    merged["ratio_nrmse_vs_best_static"] = merged["nrmse"] / merged["best_static_nrmse"]
    merged["beats_best_static"] = merged["delta_nrmse_vs_best_static"] < 0.0
    return merged.sort_values(["config", "farm", "horizon"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tuning_root", type=Path)
    parser.add_argument("--baseline_root", type=Path, default=None)
    args = parser.parse_args()

    args.tuning_root.mkdir(parents=True, exist_ok=True)
    lyra_rows = collect_lyra_rows(args.tuning_root)
    if lyra_rows.empty:
        print(f"No tuning summaries found under {args.tuning_root}")
        return

    lyra_path = args.tuning_root / "tuning_lyra_rows.csv"
    lyra_rows.to_csv(lyra_path, index=False)

    config_summary = summarize_configs(lyra_rows)
    config_path = args.tuning_root / "tuning_config_summary.csv"
    config_summary.to_csv(config_path, index=False)
    print("\nConfig summary:")
    print(config_summary.to_string(index=False))

    if args.baseline_root:
        static_rows = collect_static_baselines(args.baseline_root)
        if static_rows.empty:
            print(f"No static baseline summaries found under {args.baseline_root}")
            return
        vs_static = compare_to_static(lyra_rows, static_rows)
        vs_path = args.tuning_root / "tuning_vs_static.csv"
        vs_static.to_csv(vs_path, index=False)
        if not vs_static.empty:
            by_config = (
                vs_static.groupby("config", as_index=False)
                .agg(
                    rows=("nrmse", "size"),
                    mean_delta_nrmse=("delta_nrmse_vs_best_static", "mean"),
                    mean_ratio_nrmse=("ratio_nrmse_vs_best_static", "mean"),
                    win_rate_vs_best_static=("beats_best_static", "mean"),
                )
                .sort_values(["mean_ratio_nrmse", "mean_delta_nrmse"])
            )
            by_config.to_csv(args.tuning_root / "tuning_vs_static_by_config.csv", index=False)
            print("\nVs best static baseline:")
            print(by_config.to_string(index=False))


if __name__ == "__main__":
    main()
