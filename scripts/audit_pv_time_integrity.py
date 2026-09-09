#!/usr/bin/env python3
"""Audit timestamp gaps introduced or exposed by PV target filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COLUMN = "Time(year-month-day h:m:s)"
POWER_COLUMN = "Power (MW)"
EXPECTED_MINUTES = 15.0
POINTS_PER_DAY = 96


def crosses_gap(gap_after: np.ndarray, start: int, stop: int) -> bool:
    """Return whether [start, stop) contains a discontinuity between samples."""
    if stop - start <= 1:
        return False
    return bool(np.any(gap_after[start : stop - 1]))


def count_protocol_windows(
    gap_after: np.ndarray,
    lookback: int,
    horizon: int,
    offline_points: int,
    stride: int,
    offline_stride: int,
) -> dict[str, int | float]:
    n_points = len(gap_after) + 1

    offline_starts = range(
        0,
        max(offline_points - lookback - horizon, -1) + 1,
        offline_stride,
    )
    online_starts = range(
        offline_points - lookback,
        max(n_points - lookback - horizon, offline_points - lookback - 1) + 1,
        stride,
    )

    offline_total = 0
    offline_crossing = 0
    for start in offline_starts:
        offline_total += 1
        offline_crossing += int(
            crosses_gap(gap_after, start, start + lookback + horizon)
        )

    online_total = 0
    online_crossing = 0
    for start in online_starts:
        online_total += 1
        online_crossing += int(
            crosses_gap(gap_after, start, start + lookback + horizon)
        )

    return {
        "offline_windows": offline_total,
        "offline_crossing": offline_crossing,
        "offline_crossing_pct": 100.0 * offline_crossing / max(offline_total, 1),
        "online_windows": online_total,
        "online_crossing": online_crossing,
        "online_crossing_pct": 100.0 * online_crossing / max(online_total, 1),
    }


def audit_file(
    path: Path,
    year_points: int,
    lookback: int,
    offline_days: int,
    horizons: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw = pd.read_excel(path, usecols=[TIME_COLUMN, POWER_COLUMN])
    timestamps = pd.to_datetime(raw[TIME_COLUMN], errors="coerce")
    power = pd.to_numeric(raw[POWER_COLUMN], errors="coerce")
    valid = timestamps.notna() & np.isfinite(power.to_numpy(dtype=float))

    retained = pd.DataFrame(
        {
            "timestamp": timestamps[valid],
            "power": power[valid],
            "source_row": np.flatnonzero(valid.to_numpy()),
        }
    ).sort_values("timestamp", kind="stable")
    if len(retained) < year_points:
        raise ValueError(f"{path.name}: only {len(retained)} valid timestamp-power rows")
    retained = retained.iloc[:year_points].reset_index(drop=True)

    delta_minutes = (
        retained["timestamp"].diff().dt.total_seconds().div(60.0).iloc[1:].to_numpy()
    )
    gap_after = ~np.isclose(delta_minutes, EXPECTED_MINUTES, rtol=0.0, atol=1e-6)
    positive_gaps = delta_minutes[delta_minutes > EXPECTED_MINUTES + 1e-6]
    compressed_or_duplicate = delta_minutes[delta_minutes < EXPECTED_MINUTES - 1e-6]
    missing_intervals = np.maximum(
        np.rint(positive_gaps / EXPECTED_MINUTES).astype(int) - 1,
        0,
    )

    original_time_valid = timestamps.notna()
    original_sorted = timestamps[original_time_valid].sort_values(kind="stable")
    original_delta = (
        original_sorted.diff().dt.total_seconds().div(60.0).iloc[1:].to_numpy()
    )
    source_gap_count = int(
        np.sum(original_delta > EXPECTED_MINUTES + 1e-6)
    )

    invalid_before_cutoff = int(
        np.sum(
            (~np.isfinite(power.to_numpy(dtype=float)))
            & timestamps.notna().to_numpy()
            & (timestamps <= retained["timestamp"].iloc[-1]).to_numpy()
        )
    )

    summary = {
        "site_file": path.name,
        "source_rows": int(len(raw)),
        "valid_timestamp_rows": int(timestamps.notna().sum()),
        "valid_power_rows": int(np.isfinite(power.to_numpy(dtype=float)).sum()),
        "retained_points": int(len(retained)),
        "retained_start": retained["timestamp"].iloc[0].isoformat(),
        "retained_end": retained["timestamp"].iloc[-1].isoformat(),
        "invalid_power_before_retained_end": invalid_before_cutoff,
        "invalid_power_pct_of_retained_span": (
            100.0 * invalid_before_cutoff / max(year_points + invalid_before_cutoff, 1)
        ),
        "source_timestamp_gap_count": source_gap_count,
        "retained_discontinuity_count": int(np.sum(gap_after)),
        "retained_positive_gap_count": int(len(positive_gaps)),
        "retained_short_or_duplicate_count": int(len(compressed_or_duplicate)),
        "estimated_missing_intervals": int(np.sum(missing_intervals)),
        "max_positive_gap_minutes": (
            float(np.max(positive_gaps)) if len(positive_gaps) else EXPECTED_MINUTES
        ),
    }

    rows: list[dict[str, object]] = []
    offline_points = offline_days * POINTS_PER_DAY
    for horizon in horizons:
        counts = count_protocol_windows(
            gap_after=gap_after,
            lookback=lookback,
            horizon=horizon,
            offline_points=offline_points,
            stride=horizon,
            offline_stride=1,
        )
        rows.append(
            {
                "site_file": path.name,
                "horizon": horizon,
                "horizon_hours": horizon / 4.0,
                **counts,
            }
        )
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("solar_stations"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--year_points", type=int, default=365 * POINTS_PER_DAY)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--offline_days", type=int, default=91)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[4, 12, 24, 48, 96],
    )
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    for path in sorted(args.data_dir.glob("*.xlsx")):
        summary, rows = audit_file(
            path=path,
            year_points=args.year_points,
            lookback=args.lookback,
            offline_days=args.offline_days,
            horizons=args.horizons,
        )
        summaries.append(summary)
        windows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    windows_df = pd.DataFrame(windows)
    summary_df.to_csv(args.output_dir / "site_time_integrity_summary.csv", index=False)
    windows_df.to_csv(args.output_dir / "protocol_window_gap_exposure.csv", index=False)

    aggregate = {
        "sites": len(summary_df),
        "total_invalid_power_before_retained_end": int(
            summary_df["invalid_power_before_retained_end"].sum()
        ),
        "max_site_invalid_power_pct": float(
            summary_df["invalid_power_pct_of_retained_span"].max()
        ),
        "max_gap_minutes": float(summary_df["max_positive_gap_minutes"].max()),
        "total_retained_discontinuities": int(
            summary_df["retained_discontinuity_count"].sum()
        ),
        "weighted_online_crossing_pct": float(
            100.0
            * windows_df["online_crossing"].sum()
            / max(windows_df["online_windows"].sum(), 1)
        ),
        "weighted_offline_crossing_pct": float(
            100.0
            * windows_df["offline_crossing"].sum()
            / max(windows_df["offline_windows"].sum(), 1)
        ),
    }
    (args.output_dir / "time_integrity_audit.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    print(summary_df.to_string(index=False))
    print()
    print(windows_df.to_string(index=False))
    print()
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
