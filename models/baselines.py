"""Executable classical baselines with no trainable parameters."""

from typing import List, Sequence

from models.base import StreamingForecaster, finite_values


def _history(window: Sequence[float], horizon: int) -> List[float]:
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon <= 0
    ):
        raise ValueError("Horizon must be a positive integer")
    return finite_values(window, "Lookback")


class LastValuePersistenceForecaster(StreamingForecaster):
    """Repeat the most recently observed power value."""

    def predict(
        self, lookback_window: Sequence[float], horizon: int
    ) -> List[float]:
        """Return the last observed value for every future step."""
        values = _history(lookback_window, horizon)
        return self.clip_physical_bounds([values[-1]] * horizon)

    def update(
        self,
        lookback_window: Sequence[float],
        observed_target: Sequence[float],
    ) -> None:
        """Validate completed data; this baseline has no learned state."""
        finite_values(lookback_window, "Lookback")
        finite_values(observed_target, "Target")


class DiurnalSmartPersistenceForecaster(LastValuePersistenceForecaster):
    """Repeat the last complete daily cycle without irradiance adjustment.

    Args:
        capacity: Plant capacity.
        period_steps: Samples in a day; 96 for 15-minute observations.

    This is seasonal persistence. The class name does not imply an
    irradiance-normalized smart-persistence formulation.
    """

    def __init__(self, capacity: float, period_steps: int = 96) -> None:
        super().__init__(capacity)
        if isinstance(period_steps, bool) or not isinstance(period_steps, int):
            raise ValueError("Period must be a positive integer")
        if period_steps <= 0:
            raise ValueError("Period must be a positive integer")
        self.period_steps = period_steps

    def predict(
        self, lookback_window: Sequence[float], horizon: int
    ) -> List[float]:
        """Align each future step with its phase in the last observed cycle."""
        values = _history(lookback_window, horizon)
        if len(values) < self.period_steps:
            raise ValueError("A full observed cycle is required")
        cycle = values[-self.period_steps :]
        return self.clip_physical_bounds(
            [cycle[index % self.period_steps] for index in range(horizon)]
        )


class RollingMeanForecaster(LastValuePersistenceForecaster):
    """Repeat the simple mean of a trailing sub-window.

    Args:
        capacity: Plant capacity.
        window_steps: Maximum number of trailing observations to average.
    """

    def __init__(self, capacity: float, window_steps: int = 8) -> None:
        super().__init__(capacity)
        if isinstance(window_steps, bool) or not isinstance(window_steps, int):
            raise ValueError("Window size must be a positive integer")
        if window_steps <= 0:
            raise ValueError("Window size must be a positive integer")
        self.window_steps = window_steps

    def predict(
        self, lookback_window: Sequence[float], horizon: int
    ) -> List[float]:
        """Forecast a constant trailing-window arithmetic mean."""
        import statistics

        values = _history(lookback_window, horizon)[-self.window_steps :]
        return self.clip_physical_bounds([statistics.fmean(values)] * horizon)
