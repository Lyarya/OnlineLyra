"""Validate chronological prediction logs and print regime-aware scores."""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.prequential_runner import validate_timestamps
from utils.metrics import MetricValue, daytime_mask, summarize


def read_predictions(
    path: Path,
    interval_minutes: Optional[float] = None,
) -> Tuple[List[datetime], List[float], List[float]]:
    """Read finite prediction pairs and validate a strictly regular time grid.

    Args:
        path: CSV with timestamp, target and prediction columns.
        interval_minutes: Optional expected sample interval.

    Returns:
        Validated timestamps, targets and predictions.
    """
    stamps: List[datetime] = []
    ys: List[float] = []
    ps: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "target", "prediction"}
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError("Need unique timestamp,target,prediction columns")
        for line, row in enumerate(reader, 2):
            try:
                stamp = row["timestamp"]
                if stamp.endswith("Z"):
                    stamp = stamp[:-1] + "+00:00"
                stamps.append(datetime.fromisoformat(stamp))
                ys.append(float(row["target"]))
                ps.append(float(row["prediction"]))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid CSV line {line}: {exc}") from exc
    validate_timestamps(stamps, interval_minutes)
    # Validate values without silently clipping supplied predictions.
    summarize(ys, ps, capacity=1.0)
    return stamps, ys, ps


def format_summary(results: Dict[str, Dict[str, MetricValue]]) -> str:
    """Format count, physical-unit errors and capacity-percent tail errors."""
    lines = [
        f"{'Subset':<12} {'N':>7} {'MSE':>12} {'MAE':>12} {'RMSE':>12} "
        f"{'P95 (%C)':>12} {'P99 (%C)':>12}",
        "-" * 87,
    ]
    for subset, scores in results.items():
        cells = [f"{subset:<12}", f"{scores['count']:>7}"]
        for key in ["mse", "mae", "rmse", "p95", "p99"]:
            value = scores[key]
            cells.append(
                f"{value:>12.6f}" if value is not None else f"{'N/A':>12}"
            )
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main() -> None:
    """Run the CSV auditor without claiming to validate model internals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--interval-minutes", type=float)
    parser.add_argument("--capacity", type=float, required=True)
    parser.add_argument(
        "--initialization-variance",
        type=float,
        help="Population variance from a separate, pre-deployment prefix",
    )
    args = parser.parse_args()
    try:
        _, ys, ps = read_predictions(args.csv_path, args.interval_minutes)
        mask = daytime_mask(ys, args.capacity)
        results = {
            "overall": summarize(
                ys, ps, args.capacity, args.initialization_variance
            ),
            "daytime": summarize(
                [y for y, keep in zip(ys, mask) if keep],
                [p for p, keep in zip(ps, mask) if keep],
                args.capacity,
                args.initialization_variance,
            ),
        }
        print(format_summary(results))
        if args.initialization_variance is not None:
            print(
                "NMSE (initialization variance):", results["overall"]["nmse"]
            )
        print(
            "Log checks passed; prediction-generation causality "
            "is not inferred."
        )
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
