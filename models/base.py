"""Interfaces and physical validation for streaming forecasters."""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


def finite_values(values: Sequence[float], label: str) -> List[float]:
    """Convert a nonempty sequence to finite floats.

    Args:
        values: Numeric input sequence.
        label: Name used in validation messages.

    Returns:
        Validated numeric values.

    Raises:
        ValueError: If the sequence is empty or contains nonfinite values.
    """
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must be nonempty and finite")
    return result


class StreamingForecaster(ABC):
    """Forecasting interface whose updates consume completed observations.

    Args:
        capacity: Positive plant capacity in the same units as observations.
    """

    def __init__(self, capacity: float) -> None:
        self.capacity = float(capacity)
        if not math.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("Capacity must be finite and positive")

    @abstractmethod
    def predict(
        self, lookback_window: Sequence[float], horizon: int
    ) -> List[float]:
        """Issue a forecast using only the supplied observed history."""

    @abstractmethod
    def update(
        self,
        lookback_window: Sequence[float],
        observed_target: Sequence[float],
    ) -> None:
        """Consume a target block only after all of its values are observed."""

    def clip_physical_bounds(
        self, forecast: Sequence[float], capacity: Optional[float] = None
    ) -> List[float]:
        """Clip finite predictions to zero and the specified plant capacity.

        Args:
            forecast: Nonempty sequence of predictions.
            capacity: Optional capacity override; defaults to model capacity.

        Returns:
            Clipped predictions in the original order.
        """
        limit = self.capacity if capacity is None else float(capacity)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Capacity must be finite and positive")
        values = finite_values(forecast, "Forecast")
        return [min(limit, max(0.0, value)) for value in values]
