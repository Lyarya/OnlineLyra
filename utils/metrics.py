"""Numerical metrics in supplied units with explicit normalization rules."""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

MetricValue = Union[int, float, None]


def _pairs(
    targets: Sequence[float], predictions: Sequence[float]
) -> Tuple[List[float], List[float]]:
    ys, ps = list(targets), list(predictions)
    if not ys or len(ys) != len(ps):
        raise ValueError("Inputs must be nonempty and equally sized")
    ys, ps = [float(x) for x in ys], [float(x) for x in ps]
    if not all(math.isfinite(x) for x in ys + ps):
        raise ValueError("Inputs must be finite")
    if not all(math.isfinite(p - y) for y, p in zip(ys, ps)):
        raise ValueError("Errors exceed the supported numeric range")
    return ys, ps


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def mse(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return mean squared error in squared supplied units."""
    ys, ps = _pairs(targets, predictions)
    errors = [abs(p - y) for y, p in zip(ys, ps)]
    scale = max(errors)
    if scale == 0:
        return 0.0
    mean_scaled = math.fsum((e / scale) ** 2 for e in errors) / len(errors)
    try:
        result = (scale * math.sqrt(mean_scaled)) ** 2
    except OverflowError as exc:
        raise ValueError("MSE exceeds the supported numeric range") from exc
    if not math.isfinite(result):
        raise ValueError("MSE exceeds the supported numeric range")
    return result


def mae(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return mean absolute error in supplied units."""
    ys, ps = _pairs(targets, predictions)
    return math.fsum(abs(p - y) / len(ys) for y, p in zip(ys, ps))


def rmse(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Return root mean squared error in supplied units."""
    return math.sqrt(mse(targets, predictions))


def normalized_mse(
    targets: Sequence[float],
    predictions: Sequence[float],
    initialization_variance: float,
) -> float:
    """Divide MSE by a positive population variance fixed before deployment."""
    variance = _positive(initialization_variance, "Initialization variance")
    return mse(targets, predictions) / variance


def high_percentile_ae(
    targets: Sequence[float],
    predictions: Sequence[float],
    capacity: float,
    percentile: float = 95.0,
) -> float:
    """Return a linearly interpolated absolute-error percentile in percent C.

    Args:
        targets: Observed target values.
        predictions: Matched predictions.
        capacity: Positive plant capacity in the input units.
        percentile: Percentile in [0, 100]; interpolation uses (n - 1) * p.

    Returns:
        Percent of plant capacity, e.g. 2.0 means two percent of capacity.
    """
    limit = _positive(capacity, "Capacity")
    p = float(percentile)
    if not math.isfinite(p) or not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    ys, ps = _pairs(targets, predictions)
    errors = sorted(abs(y - pred) for y, pred in zip(ys, ps))
    index = (len(errors) - 1) * p / 100
    lower, upper = math.floor(index), math.ceil(index)
    fraction = index - lower
    value = errors[lower] * (1 - fraction) + errors[upper] * fraction
    return 100 * (value / limit)


def daytime_mask(targets: Sequence[float], capacity: float) -> List[bool]:
    """Select completed targets strictly above 1% of capacity."""
    limit = _positive(capacity, "Capacity")
    values = [float(x) for x in targets]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Targets must be finite")
    return [value > 0.01 * limit for value in values]


def summarize(
    targets: Sequence[float],
    predictions: Sequence[float],
    capacity: float,
    initialization_variance: Optional[float] = None,
) -> Dict[str, MetricValue]:
    """Return count, raw errors, capacity-percent tails and optional NMSE.

    An empty subset produces count zero and null metrics. A zero initialization
    variance produces null NMSE; negative or nonfinite variances are rejected.
    """
    _positive(capacity, "Capacity")
    if len(targets) != len(predictions):
        raise ValueError("Inputs must be equally sized")
    if initialization_variance is not None:
        if (
            not math.isfinite(initialization_variance)
            or initialization_variance < 0
        ):
            raise ValueError(
                "Initialization variance must be finite and nonnegative"
            )
    result: Dict[str, MetricValue] = {
        "count": len(targets),
        "mse": None,
        "mae": None,
        "rmse": None,
        "p95": None,
        "p99": None,
        "nmse": None,
    }
    if not targets:
        return result
    result.update(
        mse=mse(targets, predictions),
        mae=mae(targets, predictions),
        rmse=rmse(targets, predictions),
        p95=high_percentile_ae(targets, predictions, capacity, 95),
        p99=high_percentile_ae(targets, predictions, capacity, 99),
    )
    if initialization_variance is not None and initialization_variance > 0:
        result["nmse"] = normalized_mse(
            targets, predictions, initialization_variance
        )
    return result
