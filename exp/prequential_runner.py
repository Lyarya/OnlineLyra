"""Blockwise forecast-observe-evaluate-update execution on a regular grid."""

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from models.base import StreamingForecaster, finite_values
from utils.metrics import MetricValue, daytime_mask, summarize


def validate_timestamps(
    timestamps: Sequence[datetime], interval_minutes: Optional[float] = None
) -> timedelta:
    """Validate ordering, timezone consistency and a constant sampling grid.

    Args:
        timestamps: At least two datetime values, all aware or all naive.
        interval_minutes: Optional expected positive interval.

    Returns:
        Observed sampling interval.

    Raises:
        ValueError: If timestamps are unordered, mixed-zone or irregular.
    """
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required")
    if not all(isinstance(t, datetime) for t in timestamps):
        raise ValueError("Timestamps must be datetime instances")
    aware = timestamps[0].utcoffset() is not None
    if any((t.utcoffset() is not None) != aware for t in timestamps):
        raise ValueError("Do not mix timezone-aware and naive timestamps")
    canonical = (
        [t.astimezone(timezone.utc) for t in timestamps]
        if aware
        else list(timestamps)
    )
    steps = [b - a for a, b in zip(canonical, canonical[1:])]
    if any(step <= timedelta(0) for step in steps):
        raise ValueError("Timestamps must be strictly increasing")
    if any(step != steps[0] for step in steps):
        raise ValueError("Sampling intervals must be constant")
    if interval_minutes is not None:
        if not math.isfinite(interval_minutes) or interval_minutes <= 0:
            raise ValueError("Expected interval must be finite and positive")
        if steps[0] != timedelta(minutes=interval_minutes):
            raise ValueError(
                "Sampling interval differs from the expected interval"
            )
    return steps[0]


@dataclass(frozen=True)
class ForecastRecord:
    """One completed prediction and its later observed target."""

    timestamp: datetime
    target: float
    prediction: float


@dataclass(frozen=True)
class BlockRecord:
    """Block boundaries and scores computed before the model update."""

    forecast_origin: datetime
    history_end: datetime
    target_start: datetime
    target_end: datetime
    metrics: Dict[str, MetricValue]


@dataclass
class EvaluationResult:
    """Deployment records and a variance computed only from initialization."""

    records: List[ForecastRecord]
    blocks: List[BlockRecord]
    initialization_variance: float
    ignored_tail_steps: int
    capacity: float

    def summary(self) -> Dict[str, Dict[str, MetricValue]]:
        """Summarize overall and active-generation deployment errors."""
        ys = [r.target for r in self.records]
        ps = [r.prediction for r in self.records]
        mask = daytime_mask(ys, self.capacity)
        return {
            "overall": summarize(
                ys, ps, self.capacity, self.initialization_variance
            ),
            "daytime": summarize(
                [y for y, use in zip(ys, mask) if use],
                [p for p, use in zip(ps, mask) if use],
                self.capacity,
                self.initialization_variance,
            ),
        }


class BlockPrequentialEvaluator:
    """Evaluate a model over nonoverlapping, completed forecast blocks.

    Args:
        model: Forecaster with user-controlled initialization, if needed.
        lookback_steps: Number of historical values passed to predict/update.
        horizon: Target block length.
        interval_minutes: Required regular sampling interval.

    The decision origin is the midpoint between the last history timestamp and
    the first target timestamp. Thus history strictly precedes the origin and
    all target timestamps strictly follow it. Timestamps denote available
    samples; telemetry delays and interval-end conventions are caller concerns.
    The model receives no target values until scoring completes. External
    model code remains responsible for not accessing other future data.
    """

    def __init__(
        self,
        model: StreamingForecaster,
        lookback_steps: int,
        horizon: int,
        interval_minutes: float = 15.0,
    ) -> None:
        for label, value in [
            ("Lookback", lookback_steps),
            ("Horizon", horizon),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        if not math.isfinite(interval_minutes) or interval_minutes <= 0:
            raise ValueError("Sampling interval must be finite and positive")
        self.model = model
        self.lookback_steps = lookback_steps
        self.horizon = horizon
        self.interval_minutes = interval_minutes

    def validate_block(
        self,
        history_timestamps: Sequence[datetime],
        target_timestamps: Sequence[datetime],
        forecast_origin: datetime,
    ) -> None:
        """Reject overlapping or future-contaminated lookback boundaries."""
        if len(history_timestamps) != self.lookback_steps:
            raise ValueError("Unexpected lookback length")
        if len(target_timestamps) != self.horizon:
            raise ValueError("Unexpected target block length")
        if forecast_origin.utcoffset() is not None:
            forecast_origin = forecast_origin.astimezone(timezone.utc)
            history_timestamps = [
                t.astimezone(timezone.utc) if t.utcoffset() is not None else t
                for t in history_timestamps
            ]
            target_timestamps = [
                t.astimezone(timezone.utc) if t.utcoffset() is not None else t
                for t in target_timestamps
            ]
        try:
            if any(t >= forecast_origin for t in history_timestamps):
                raise ValueError(
                    "Lookback contains a timestamp at or after origin"
                )
            if any(t <= forecast_origin for t in target_timestamps):
                raise ValueError(
                    "Targets must strictly follow the forecast origin"
                )
            validate_timestamps(
                list(history_timestamps) + list(target_timestamps),
                self.interval_minutes,
            )
        except TypeError as exc:
            raise ValueError(
                "Timestamp timezone conventions must agree"
            ) from exc

    def run(
        self,
        timestamps: Sequence[datetime],
        values: Sequence[float],
        initialization_steps: int,
    ) -> EvaluationResult:
        """Evaluate complete blocks after an initialization prefix.

        Args:
            timestamps: Ordered, regularly spaced sample timestamps.
            values: Observed power values in [0, capacity].
            initialization_steps: Prefix for history and scoring scale.

        Returns:
            Per-target records, pre-update block scores and subset summaries.

        Raises:
            ValueError: If inputs, timing or model outputs are invalid.
        """
        if len(timestamps) != len(values):
            raise ValueError("Timestamp/value lengths differ")
        step = validate_timestamps(timestamps, self.interval_minutes)
        timestamps = [
            t.astimezone(timezone.utc) if t.utcoffset() is not None else t
            for t in timestamps
        ]
        power = finite_values(values, "Power stream")
        if any(x < 0 or x > self.model.capacity for x in power):
            raise ValueError(
                "Power observations must lie within plant capacity"
            )
        if isinstance(initialization_steps, bool) or not isinstance(
            initialization_steps, int
        ):
            raise ValueError("Initialization length must be an integer")
        if (
            not self.lookback_steps
            <= initialization_steps
            <= len(power) - self.horizon
        ):
            raise ValueError(
                "Need a full lookback and at least one target block"
            )
        variance = statistics.pvariance(power[:initialization_steps])
        records: List[ForecastRecord] = []
        blocks: List[BlockRecord] = []
        for start in range(
            initialization_steps, len(power) - self.horizon + 1, self.horizon
        ):
            end = start + self.horizon
            history = tuple(power[start - self.lookback_steps : start])
            history_times = timestamps[start - self.lookback_steps : start]
            target_times = timestamps[start:end]
            origin = history_times[-1] + step / 2
            self.validate_block(history_times, target_times, origin)
            raw = self.model.predict(history, self.horizon)
            if len(raw) != self.horizon:
                raise ValueError(
                    "Model returned an unexpected forecast length"
                )
            prediction = self.model.clip_physical_bounds(raw)
            # Targets become visible to the scoring/update path only here.
            observed: Tuple[float, ...] = tuple(power[start:end])
            scores = summarize(
                observed, prediction, self.model.capacity, variance
            )
            blocks.append(
                BlockRecord(
                    origin,
                    history_times[-1],
                    target_times[0],
                    target_times[-1],
                    scores,
                )
            )
            records.extend(
                ForecastRecord(t, y, p)
                for t, y, p in zip(target_times, observed, prediction)
            )
            self.model.update(history, observed)
        return EvaluationResult(
            records,
            blocks,
            variance,
            (len(power) - initialization_steps) % self.horizon,
            self.model.capacity,
        )
