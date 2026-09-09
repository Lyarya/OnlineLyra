"""Build paper-facing Online Lyra core ablation tables.

The main full run already contains several ablation rows without retraining:
Online Lyra gated, Online Lyra without gate, trend-only, and persistence.
Offline Lyra is read from the offline control. The compact server ablation adds
the only missing trained variant: no spectral residual scaling.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = ["nrmse", "nmae", "daytime_nrmse", "daytime_nmae", "skill_vs_persistence"]


def site_id_from_text(text: str) -> int | None:
    match = re.search(r"site[_\s]+(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def collect_final(final_root: Path, seed: int, sites: set[int], horizons: set[int]) -> pd.DataFrame:
    rows = []
    model_map = {
        "online_lyra_pv_gated": "Full Online Lyra",
        "online_lyra_pv": "w/o gate",
        "trend_only": "w/o residual learner",
        "persistence": "Persistence",
    }
    for path in sorted(final_root.glob(f"b06_final/site_*_seed_{seed}/online_pv_summary.csv")):
        site = site_id_from_text(str(path))
        if site not in sites:
            continue
        df = pd.read_csv(path)
        df = df[df["horizon"].isin(horizons) & df["model"].isin(model_map)].copy()
        df["site_id"] = site
        df["ablation"] = df["model"].map(model_map)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_offline(offline_root: Path, seed: int, sites: set[int], horizons: set[int]) -> pd.DataFrame:
    path = offline_root / f"offline_lyra_seed_{seed}" / "offline_lyra_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["site_id"] = df["farm"].map(lambda x: site_id_from_text(str(x)))
    df = df[df["site_id"].isin(sites) & df["horizon"].isin(horizons)].copy()
    df["ablation"] = "w/o online adaptation"
    return df


def collect_compact(root: Path | None, seed: int, sites: set[int], horizons: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if root is None or not root.exists():
        return pd.DataFrame(), pd.DataFrame()
    ablation_rows = []
    cold_rows = []
    variant_map = {
        "ab_no_spectral_residual_scale": "w/o spectral residual scaling",
        "cold30_final": "cold start: 30 offline days",
        "cold60_final": "cold start: 60 offline days",
    }
    for path in sorted(root.glob(f"*/site_*_seed_{seed}/online_pv_summary.csv")):
        variant = path.parents[1].name
        if variant not in variant_map:
            continue
        site = site_id_from_text(str(path))
        if site not in sites:
            continue
        df = pd.read_csv(path)
        df = df[df["horizon"].isin(horizons) & df["model"].eq("online_lyra_pv_gated")].copy()
        df["site_id"] = site
        df["ablation"] = variant_map[variant]
        if variant.startswith("cold"):
            cold_rows.append(df)
        else:
            ablation_rows.append(df)
    ablation = pd.concat(ablation_rows, ignore_index=True) if ablation_rows else pd.DataFrame()
    cold = pd.concat(cold_rows, ignore_index=True) if cold_rows else pd.DataFrame()
    return ablation, cold


def aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(group_col, as_index=False)
        .agg(
            conditions=("horizon", "size"),
            **{metric: (metric, "mean") for metric in METRICS if metric in df.columns},
        )
        .sort_values("nrmse")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Online Lyra core ablation tables")
    parser.add_argument("--final_root", type=Path, default=Path("outputs/server_runs_20260716/final_online_lyra_b06_20260716_231353"))
    parser.add_argument("--offline_root", type=Path, default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425"))
    parser.add_argument("--compact_root", type=Path, default=None)
    parser.add_argument("--sites", type=int, nargs="+", default=[1, 2, 5, 8])
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 24, 96])
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_result_pack/core_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites = set(args.sites)
    horizons = set(args.horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final = collect_final(args.final_root, args.seed, sites, horizons)
    offline = collect_offline(args.offline_root, args.seed, sites, horizons)
    compact_ablation, cold = collect_compact(args.compact_root, args.seed, sites, horizons)

    ablation_long = pd.concat([final, offline, compact_ablation], ignore_index=True)
    ablation_long.to_csv(args.output_dir / "core_ablation_long.csv", index=False)
    aggregate(ablation_long, "ablation").to_csv(args.output_dir / "core_ablation_summary.csv", index=False)

    if not cold.empty:
        cold.to_csv(args.output_dir / "cold_start_long.csv", index=False)
        aggregate(cold, "ablation").to_csv(args.output_dir / "cold_start_summary.csv", index=False)
    print(f"Saved core ablation tables to {args.output_dir}")


if __name__ == "__main__":
    main()
