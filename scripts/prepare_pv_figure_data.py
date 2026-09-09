"""Prepare plot-ready data files for Online Lyra PV case-study figures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_prediction(root: Path, site: int, seed: int, horizon: int, kind: str) -> Path:
    if kind == "online":
        matches = sorted((root / "b06_final" / f"site_{site}_seed_{seed}" / "artifacts").glob(f"*_h{horizon}_predictions.npz"))
    elif kind == "offline":
        matches = sorted((root / f"offline_lyra_seed_{seed}" / "artifacts").glob(f"*site_{site}_*_h{horizon}_offline_lyra_final_predictions.npz"))
    else:
        raise ValueError(kind)
    if not matches:
        raise FileNotFoundError(f"No {kind} prediction file for site={site}, seed={seed}, horizon={horizon}")
    return matches[0]


def select_volatile_segment(y_true: np.ndarray, days: int) -> int:
    if len(y_true) <= days:
        return 0
    volatility = np.mean(np.abs(np.diff(y_true, axis=1)), axis=1)
    scores = np.convolve(volatility, np.ones(days), mode="valid")
    return int(np.argmax(scores))


def make_case_trace(
    online_root: Path,
    offline_root: Path,
    site: int,
    seed: int,
    horizon: int,
    days: int,
    output_dir: Path,
) -> None:
    online_path = find_prediction(online_root, site, seed, horizon, "online")
    offline_path = find_prediction(offline_root, site, seed, horizon, "offline")
    online = np.load(online_path)
    offline = np.load(offline_path)
    y_true = online["y_true"].astype(float)
    start = select_volatile_segment(y_true, days)
    stop = min(start + days, len(y_true))

    rows = []
    for local_day, window_idx in enumerate(range(start, stop)):
        gate = bool(online["gate_uses_lyra"][window_idx]) if "gate_uses_lyra" in online.files else True
        for lead in range(horizon):
            rows.append(
                {
                    "site": site,
                    "seed": seed,
                    "horizon": horizon,
                    "window_index": window_idx,
                    "local_day": local_day,
                    "lead": lead + 1,
                    "time_index": local_day * horizon + lead,
                    "y_true": float(online["y_true"][window_idx, lead]),
                    "online_lyra": float(online["online_lyra_pv_gated"][window_idx, lead]),
                    "online_lyra_raw": float(online["online_lyra_pv"][window_idx, lead]),
                    "persistence": float(online["persistence"][window_idx, lead]),
                    "offline_lyra": float(offline["offline_lyra_final"][window_idx, lead]),
                    "gate_uses_lyra": gate,
                    "capacity": float(online["capacity"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / f"case_trace_site{site}_h{horizon}_seed{seed}.csv", index=False)

    trace = pd.DataFrame(rows)
    lead_profile = (
        trace.assign(
            online_abs_error=lambda x: (x["online_lyra"] - x["y_true"]).abs(),
            offline_abs_error=lambda x: (x["offline_lyra"] - x["y_true"]).abs(),
            persistence_abs_error=lambda x: (x["persistence"] - x["y_true"]).abs(),
        )
        .groupby("lead", as_index=False)
        .agg(
            online_mae=("online_abs_error", "mean"),
            offline_mae=("offline_abs_error", "mean"),
            persistence_mae=("persistence_abs_error", "mean"),
        )
    )
    lead_profile.to_csv(output_dir / f"lead_error_profile_site{site}_h{horizon}_seed{seed}.csv", index=False)


def make_gate_profile(online_root: Path, output_dir: Path) -> None:
    rows = []
    for path in sorted((online_root / "b06_final").glob("site_*_seed_*/online_pv_summary.csv")):
        meta = re.search(r"site_(\d+)_seed_(\d+)", str(path))
        df = pd.read_csv(path)
        gated = df[df["model"].eq("online_lyra_pv_gated")].copy()
        if meta:
            gated["site_id"] = int(meta.group(1))
            gated["seed"] = int(meta.group(2))
        rows.append(gated)
    if rows:
        gate = pd.concat(rows, ignore_index=True)
        gate.to_csv(output_dir / "gate_profile_all_sites.csv", index=False)
        (
            gate.groupby(["site_id", "horizon"], as_index=False)
            .agg(
                gate_lyra_fraction=("gate_lyra_fraction", "mean"),
                nrmse=("nrmse", "mean"),
                nmae=("nmae", "mean"),
                update_latency_ms=("update_latency_ms", "mean"),
            )
            .to_csv(output_dir / "gate_profile_site_horizon_average.csv", index=False)
        )


def make_horizon_profiles(tables_dir: Path, output_dir: Path) -> None:
    for name in ["main_average_by_horizon.csv", "main_condition_average.csv", "main_metric_long_with_ranks.csv"]:
        src = tables_dir / name
        if src.exists():
            pd.read_csv(src).to_csv(output_dir / name, index=False)


def make_daily_error_data(
    online_root: Path,
    offline_root: Path,
    output_dir: Path,
    horizon: int = 96,
) -> None:
    rows = []
    for online_path in sorted((online_root / "b06_final").glob(f"site_*_seed_*/artifacts/*_h{horizon}_predictions.npz")):
        meta = re.search(r"site_(\d+)_seed_(\d+)", str(online_path))
        if not meta:
            continue
        site = int(meta.group(1))
        seed = int(meta.group(2))
        offline_path = find_prediction(offline_root, site, seed, horizon, "offline")
        online = np.load(online_path)
        offline = np.load(offline_path)
        capacity = float(online["capacity"])
        y_true = online["y_true"].astype(float)
        day_mask = y_true > capacity * 0.01
        model_arrays = {
            "Online Lyra": online["online_lyra_pv_gated"].astype(float),
            "Offline Lyra": offline["offline_lyra_final"].astype(float),
            "Persistence": online["persistence"].astype(float),
        }
        energy = y_true.mean(axis=1) / max(capacity, 1e-6)
        volatility = np.mean(np.abs(np.diff(y_true, axis=1)), axis=1) / max(capacity, 1e-6)
        for idx in range(len(y_true)):
            for model, pred in model_arrays.items():
                err = np.abs(pred[idx] - y_true[idx])
                rows.append(
                    {
                        "site": site,
                        "seed": seed,
                        "day_index": idx,
                        "model": model,
                        "mae": float(np.mean(err)),
                        "daytime_mae": float(np.mean(err[day_mask[idx]])) if np.any(day_mask[idx]) else float("nan"),
                        "nighttime_mae": float(np.mean(err[~day_mask[idx]])) if np.any(~day_mask[idx]) else float("nan"),
                        "energy_proxy": float(energy[idx]),
                        "volatility_proxy": float(volatility[idx]),
                        "gate_uses_lyra": bool(online["gate_uses_lyra"][idx]) if "gate_uses_lyra" in online.files else True,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / f"daily_error_data_h{horizon}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PV figure data")
    parser.add_argument("--online_root", type=Path, default=Path("outputs/server_runs_20260716/final_online_lyra_b06_20260716_231353"))
    parser.add_argument("--offline_root", type=Path, default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425"))
    parser.add_argument("--tables_dir", type=Path, default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425/paper_tables_with_offline_lyra"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_result_pack/figure_data"))
    parser.add_argument("--site", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--days", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_case_trace(args.online_root, args.offline_root, args.site, args.seed, args.horizon, args.days, args.output_dir)
    make_gate_profile(args.online_root, args.output_dir)
    make_horizon_profiles(args.tables_dir, args.output_dir)
    make_daily_error_data(args.online_root, args.offline_root, args.output_dir, horizon=96)
    print(f"Saved figure data to {args.output_dir}")


if __name__ == "__main__":
    main()
