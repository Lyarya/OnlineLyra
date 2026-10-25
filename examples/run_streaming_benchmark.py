"""Benchmark classical forecasters on a generated photovoltaic power stream."""

import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from exp.prequential_runner import BlockPrequentialEvaluator, EvaluationResult
from models.baselines import (
    DiurnalSmartPersistenceForecaster,
    LastValuePersistenceForecaster,
    RollingMeanForecaster,
)


def generate_stream(
    days: int = 14,
    capacity: float = 50.0,
    seed: int = 42,
) -> Tuple[List[datetime], List[float]]:
    """Generate 15-minute power with daily cycles and persistent cloud dips.

    Args:
        days: Number of complete generated days, at least four.
        capacity: Positive rated output.
        seed: Local random-generator seed.

    Returns:
        UTC timestamps and simulated power. Nighttime output is exactly zero.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days < 4:
        raise ValueError("Generate at least four complete days")
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Capacity must be finite and positive")
    rng = random.Random(seed)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    stamps: List[datetime] = []
    power: List[float] = []
    cloud = 0.0
    daily_strength = 0.9
    for index in range(days * 96):
        stamp = start + timedelta(minutes=15 * index)
        hour = stamp.hour + stamp.minute / 60
        if index % 96 == 0:
            daily_strength = rng.uniform(0.8, 1.0)
        cloud *= 0.8
        if 6 < hour < 18 and rng.random() < 0.08:
            cloud = rng.uniform(0.2, 0.75)
        daylight = (
            math.sin(math.pi * (hour - 6) / 12) if 6 < hour < 18 else 0.0
        )
        value = capacity * daily_strength * daylight * (1 - cloud)
        stamps.append(stamp)
        power.append(max(0.0, min(capacity, value)))
    return stamps, power


def run_benchmark(
    days: int,
    capacity: float,
    seed: int,
) -> Dict[Tuple[str, int], EvaluationResult]:
    """Run independent model instances across three short horizons."""
    stamps, power = generate_stream(days, capacity, seed)
    constructors = {
        "LastValuePersistence": LastValuePersistenceForecaster,
        "DiurnalSmartPersistence": DiurnalSmartPersistenceForecaster,
        "RollingMean": RollingMeanForecaster,
    }
    results: Dict[Tuple[str, int], EvaluationResult] = {}
    for name, constructor in constructors.items():
        for horizon in [4, 12, 24]:
            model = constructor(capacity=capacity)
            runner = BlockPrequentialEvaluator(
                model, lookback_steps=192, horizon=horizon
            )
            results[(name, horizon)] = runner.run(
                stamps, power, initialization_steps=288
            )
    return results


def export_predictions(path: Path, result: EvaluationResult) -> None:
    """Write one condition's deployment records without overwriting a file."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "target", "prediction"])
        for record in result.records:
            writer.writerow(
                [
                    record.timestamp.isoformat(),
                    record.target,
                    record.prediction,
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print all conditions and optionally export one prediction series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--capacity", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-csv", type=Path)
    parser.add_argument(
        "--export-model",
        default="LastValuePersistence",
        choices=[
            "LastValuePersistence",
            "DiurnalSmartPersistence",
            "RollingMean",
        ],
    )
    parser.add_argument(
        "--export-horizon", type=int, default=4, choices=[4, 12, 24]
    )
    args = parser.parse_args(argv)
    try:
        results = run_benchmark(args.days, args.capacity, args.seed)
        print(
            "Generated PV stream; simulation results, "
            "not measured station performance."
        )
        print(
            "Three-day initialization; two-day lookback; 15-minute sampling."
        )
        print(
            f"{'Baseline':<25} {'H':>3} {'N':>6} {'MSE':>10} {'MAE':>10} "
            f"{'Day MSE':>10} {'P95 %C':>9} {'P99 %C':>9}"
        )
        for (name, horizon), result in results.items():
            scores = result.summary()
            overall, daytime = scores["overall"], scores["daytime"]
            print(
                f"{name:<25} {horizon:>3} {overall['count']:>6} "
                f"{overall['mse']:>10.4f} {overall['mae']:>10.4f} "
                f"{daytime['mse']:>10.4f} {overall['p95']:>9.4f} "
                f"{overall['p99']:>9.4f}"
            )
        if args.export_csv:
            condition = (args.export_model, args.export_horizon)
            export_predictions(args.export_csv, results[condition])
            print(
                f"Exported {condition[0]}, H={condition[1]} "
                f"to {args.export_csv}"
            )
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
