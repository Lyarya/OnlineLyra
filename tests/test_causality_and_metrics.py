"""Tests for numerical definitions, public interfaces and causal scheduling."""

import math
import statistics
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

from examples.run_streaming_benchmark import (
    export_predictions,
    generate_stream,
)
from exp.prequential_runner import BlockPrequentialEvaluator
from models.base import StreamingForecaster
from models.baselines import (
    DiurnalSmartPersistenceForecaster,
    LastValuePersistenceForecaster,
    RollingMeanForecaster,
)
from scripts.evaluate_csv import read_predictions
from utils.metrics import (
    MetricValue,
    daytime_mask,
    high_percentile_ae,
    mae,
    mse,
    normalized_mse,
    rmse,
    summarize,
)


class RecordingForecaster(StreamingForecaster):
    """A test forecaster that records calls and counts completed updates."""

    def __init__(self, events: List[str]) -> None:
        super().__init__(capacity=50)
        self.events = events
        self.updates = 0

    def predict(
        self, lookback_window: Sequence[float], horizon: int
    ) -> List[float]:
        """Record a prediction without accessing future inputs."""
        self.events.append("predict")
        return [float(self.updates)] * horizon

    def update(
        self,
        lookback_window: Sequence[float],
        observed_target: Sequence[float],
    ) -> None:
        """Record one update after a complete block."""
        self.events.append("update")
        self.updates += 1


class MetricsTests(unittest.TestCase):
    """Verify scores against independently calculated examples."""

    def test_known_errors(self) -> None:
        """Errors 0, 1 and 2 have MSE 5/3 and MAE 1."""
        self.assertAlmostEqual(mse([0, 1, 2], [0, 2, 4]), 5 / 3)
        self.assertAlmostEqual(mae([0, 1, 2], [0, 2, 4]), 1)
        self.assertAlmostEqual(rmse([0, 1, 2], [0, 2, 4]), math.sqrt(5 / 3))
        self.assertAlmostEqual(normalized_mse([0, 1, 2], [0, 2, 4], 2), 5 / 6)

    def test_tail_interpolation_and_daytime_boundary(self) -> None:
        """Tail metrics interpolate and the daytime threshold is strict."""
        self.assertAlmostEqual(
            high_percentile_ae([0, 0], [0, 10], 20, 95), 47.5
        )
        self.assertAlmostEqual(
            high_percentile_ae([0, 0], [0, 10], 20, 99), 49.5
        )
        self.assertEqual(
            daytime_mask([0, 0.5, 0.5001], 50), [False, False, True]
        )

    def test_invalid_and_empty_inputs(self) -> None:
        """Reject malformed inputs and handle empty or constant subsets."""
        cases: List[Tuple[List[float], List[float]]] = [
            ([], []),
            ([1.0], [1.0, 2.0]),
            ([math.nan], [0.0]),
        ]
        for ys, ps in cases:
            with self.assertRaises(ValueError):
                mse(ys, ps)
        with self.assertRaises(ValueError):
            normalized_mse([1], [2], 0)
        with self.assertRaises(ValueError):
            high_percentile_ae([0], [1], 0)
        self.assertIsNone(summarize([], [], 50)["mse"])
        self.assertIsNone(summarize([1], [2], 50, 0)["nmse"])


class BaselineTests(unittest.TestCase):
    """Check baseline behavior without relying on the benchmark output."""

    def test_fixed_forecasts(self) -> None:
        """Check persistence, trailing mean and seasonal phase alignment."""
        self.assertEqual(
            LastValuePersistenceForecaster(50).predict([1, 3], 2), [3, 3]
        )
        self.assertEqual(
            RollingMeanForecaster(50, 2).predict([1, 3, 5], 2), [4, 4]
        )
        seasonal = DiurnalSmartPersistenceForecaster(50, 4)
        self.assertEqual(
            seasonal.predict([9, 1, 2, 3, 4], 6), [1, 2, 3, 4, 1, 2]
        )
        with self.assertRaises(ValueError):
            seasonal.predict([1, 2], 3)
        self.assertEqual(
            seasonal.clip_physical_bounds([-1, 5, 60]), [0, 5, 50]
        )


class CausalityTests(unittest.TestCase):
    """Test timestamp boundaries, call order and future-prefix independence."""

    def setUp(self) -> None:
        """Build a small regular stream."""
        self.times = [
            datetime(2020, 1, 1, tzinfo=timezone.utc)
            + timedelta(minutes=15 * i)
            for i in range(12)
        ]
        self.values = [float(i) for i in range(12)]

    def test_future_timestamp_in_history_is_rejected(self) -> None:
        """Reject future or overlapping timestamps in the lookback."""
        runner = BlockPrequentialEvaluator(
            LastValuePersistenceForecaster(50), 2, 2
        )
        origin = self.times[2] - timedelta(minutes=7.5)
        with self.assertRaises(ValueError):
            runner.validate_block(
                [self.times[0], self.times[2]], self.times[2:4], origin
            )
        with self.assertRaises(ValueError):
            runner.validate_block(self.times[:2], self.times[1:3], origin)

    def test_forecast_score_update_order(self) -> None:
        """Score before each update and forecast with prior state."""
        events: List[str] = []
        model = RecordingForecaster(events)
        runner = BlockPrequentialEvaluator(model, 2, 2)

        def score(
            targets: Sequence[float],
            predictions: Sequence[float],
            capacity: float,
            initialization_variance: Optional[float] = None,
        ) -> Dict[str, MetricValue]:
            events.append("score")
            return summarize(
                targets, predictions, capacity, initialization_variance
            )

        with patch("exp.prequential_runner.summarize", side_effect=score):
            result = runner.run(self.times, self.values, 4)
        self.assertEqual(events, ["predict", "score", "update"] * 4)
        self.assertEqual(
            [r.prediction for r in result.records], [0, 0, 1, 1, 2, 2, 3, 3]
        )
        self.assertEqual(
            result.initialization_variance,
            statistics.pvariance(self.values[:4]),
        )

    def test_future_values_do_not_change_issued_forecast(self) -> None:
        """Keep issued forecasts independent of an unobserved suffix."""
        changed = self.values[:4] + [49.0] * 8
        a = BlockPrequentialEvaluator(LastValuePersistenceForecaster(50), 2, 2)
        b = BlockPrequentialEvaluator(LastValuePersistenceForecaster(50), 2, 2)
        first = a.run(self.times, self.values, 4)
        second = b.run(self.times, changed, 4)
        self.assertEqual(
            [r.prediction for r in first.records[:2]],
            [r.prediction for r in second.records[:2]],
        )
        self.assertEqual(
            first.initialization_variance, second.initialization_variance
        )

    def test_irregular_grid_and_partial_tail(self) -> None:
        """Reject gaps and ignore incomplete trailing blocks."""
        runner = BlockPrequentialEvaluator(
            LastValuePersistenceForecaster(50), 2, 3
        )
        result = runner.run(self.times, self.values, 4)
        self.assertEqual(len(result.records), 6)
        self.assertEqual(result.ignored_tail_steps, 2)
        bad = self.times[:]
        bad[5] = bad[4]
        with self.assertRaises(ValueError):
            runner.run(bad, self.values, 4)

    def test_invalid_model_output(self) -> None:
        """Reject forecast blocks with invalid lengths or values."""
        model = LastValuePersistenceForecaster(50)
        runner = BlockPrequentialEvaluator(model, 2, 2)
        for output in [[1], [1, math.nan]]:
            with patch.object(model, "predict", return_value=output):
                with self.assertRaises(ValueError):
                    runner.run(self.times, self.values, 4)


class IntegrationTests(unittest.TestCase):
    """Exercise generated data and CSV round-trip behavior."""

    def test_synthetic_stream_and_csv_roundtrip(self) -> None:
        """Verify nighttime zeros and exact CSV round trips."""
        times, values = generate_stream(4, 50, 42)
        self.assertEqual((times, values), generate_stream(4, 50, 42))
        for stamp, value in zip(times, values):
            self.assertTrue(0 <= value <= 50)
            if stamp.hour < 6 or stamp.hour >= 18:
                self.assertEqual(value, 0)
        result = BlockPrequentialEvaluator(
            LastValuePersistenceForecaster(50), 96, 4
        ).run(times, values, 288)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            export_predictions(path, result)
            stamps, ys, ps = read_predictions(path, 15)
            self.assertEqual(stamps, [r.timestamp for r in result.records])
            self.assertEqual(ys, [r.target for r in result.records])
            self.assertEqual(ps, [r.prediction for r in result.records])
            with self.assertRaises(FileExistsError):
                export_predictions(path, result)


if __name__ == "__main__":
    unittest.main()
