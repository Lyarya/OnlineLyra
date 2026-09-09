#!/usr/bin/env python3
"""Re-score saved Online Lyra predictions after excluding gap-crossing windows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COLUMN = "Time(year-month-day h:m:s)"
POWER_COLUMN = "Power (MW)"
EXPECTED_MINUTES = 15.0
YEAR_POINTS = 365 * 96
OFFLINE_POINTS = 91 * 96
LOOKBACK = 96


def retained_gap_mask(path: Path) -> np.ndarray:
    raw = pd.read_excel(path, usecols=[TIME_COLUMN, POWER_COLUMN])
    timestamps = pd.to_datetime(raw[TIME_COLUMN], errors="coerce")
    power = pd.to_numeric(raw[POWER_COLUMN], errors="coerce")
    valid = timestamps.notna() & np.isfinite(power.to_numpy(dtype=float))
    retained = timestamps[valid].sort_values(kind="stable").iloc[:YEAR_POINTS]
    delta = retained.diff().dt.total_seconds().div(60.0).iloc[1:].to_numpy()
    return ~np.isclose(delta, EXPECTED_MINUTES, rtol=0.0, atol=1e-6)


def site_id(path: Path) -> int:
    match = re.search(r"site[_\s-]*(\d+)", path.name, flags=re.I)
    if not match:
        raise ValueError(f"Cannot infer site id from {path.name}")
    return int(match.group(1))


def prediction_keep_mask(gap_after: np.ndarray, horizon: int, n_rows: int) -> np.ndarray:
    starts = OFFLINE_POINTS - LOOKBACK + np.arange(n_rows) * horizon
    keep = np.ones(n_rows, dtype=bool)
    for row, start in enumerate(starts):
        stop = start + LOOKBACK + horizon
        keep[row] = not np.any(gap_after[start : stop - 1])
    return keep


def normalized_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float,
) -> tuple[float, float]:
    error = y_pred.astype(np.float64) - y_true.astype(np.float64)
    nrmse = float(np.sqrt(np.mean(np.square(error))) / capacity)
    nmae = float(np.mean(np.abs(error)) / capacity)
    return nrmse, nmae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("solar_stations"))
    parser.add_argument("--predictions_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    data_files = {
        site_id(path): path for path in sorted(args.data_dir.glob("*.xlsx"))
    }
    gap_masks = {site: retained_gap_mask(path) for site, path in data_files.items()}

    rows: list[dict[str, object]] = []
    for path in sorted(args.predictions_root.rglob("*_predictions.npz")):
        site = site_id(path)
        match_h = re.search(r"_h(\d+)(?:_baselines)?_predictions", path.name)
        match_seed = re.search(r"seed_(\d+)", str(path))
        if not match_h or not match_seed:
            continue
        horizon = int(match_h.group(1))
        seed = int(match_seed.group(1))

        with np.load(path) as artifact:
            y_true = artifact["y_true"]
            capacity = float(artifact["capacity"])
            keep = prediction_keep_mask(gap_masks[site], horizon, len(y_true))
            model_keys = [
                key
                for key in artifact.files
                if key not in {
                    "capacity",
                    "horizon",
                    "y_true",
                    "gate_uses_lyra",
                }
                and artifact[key].shape == y_true.shape
            ]
            for model_key in model_keys:
                y_pred = artifact[model_key]
                full_nrmse, full_nmae = normalized_metrics(y_true, y_pred, capacity)
                kept_nrmse, kept_nmae = normalized_metrics(
                    y_true[keep],
                    y_pred[keep],
                    capacity,
                )
                rows.append(
                    {
                        "site": site,
                        "horizon": horizon,
                        "seed": seed,
                        "model": model_key,
                        "total_windows": len(keep),
                        "excluded_windows": int(np.sum(~keep)),
                        "excluded_pct": 100.0 * np.mean(~keep),
                        "full_nrmse": full_nrmse,
                        "gap_excluded_nrmse": kept_nrmse,
                        "nrmse_change": kept_nrmse - full_nrmse,
                        "full_nmae": full_nmae,
                        "gap_excluded_nmae": kept_nmae,
                        "nmae_change": kept_nmae - full_nmae,
                    }
                )

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "gap_excluded_metric_sensitivity.csv", index=False)

    aggregate = (
        result.groupby("model", as_index=False)
        .agg(
            conditions=("nrmse_change", "size"),
            affected_conditions=("excluded_windows", lambda x: int(np.sum(x > 0))),
            mean_full_nrmse=("full_nrmse", "mean"),
            mean_gap_excluded_nrmse=("gap_excluded_nrmse", "mean"),
            mean_nrmse_change=("nrmse_change", "mean"),
            max_abs_nrmse_change=("nrmse_change", lambda x: float(np.max(np.abs(x)))),
            mean_full_nmae=("full_nmae", "mean"),
            mean_gap_excluded_nmae=("gap_excluded_nmae", "mean"),
            mean_nmae_change=("nmae_change", "mean"),
            max_abs_nmae_change=("nmae_change", lambda x: float(np.max(np.abs(x)))),
        )
    )
    aggregate.to_csv(
        args.output_dir / "gap_excluded_metric_sensitivity_aggregate.csv",
        index=False,
    )

    print(aggregate.to_string(index=False))
    print()
    affected = result[result["excluded_windows"] > 0]
    print(
        affected.sort_values(
            ["model", "site", "horizon", "seed"]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
