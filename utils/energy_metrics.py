"""
Energy-domain evaluation metrics for PV forecasting experiments.

These helpers are deliberately small and NumPy-only so they can be reused from
training scripts, notebooks, and paper table generation without pulling in the
PyTorch runtime.
"""

from __future__ import annotations

import numpy as np


EPSILON = 1e-8


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def normalized_rmse(y_true, y_pred, capacity: float | None = None) -> float:
    """RMSE normalized by PV installed capacity or observed range."""
    y_true = _as_float_array(y_true)
    denom = capacity
    if denom is None:
        denom = float(np.nanmax(y_true) - np.nanmin(y_true))
    return rmse(y_true, y_pred) / max(float(denom), EPSILON)


def normalized_mae(y_true, y_pred, capacity: float | None = None) -> float:
    """MAE normalized by PV installed capacity or observed range."""
    y_true = _as_float_array(y_true)
    denom = capacity
    if denom is None:
        denom = float(np.nanmax(y_true) - np.nanmin(y_true))
    return mae(y_true, y_pred) / max(float(denom), EPSILON)


def mape_nonzero(y_true, y_pred, min_power: float = EPSILON) -> float:
    """MAPE on non-zero generation periods only."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    mask = np.abs(y_true) > min_power
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))


def skill_score(y_true, y_pred, y_reference) -> float:
    """RMSE skill score relative to a reference forecast; higher is better."""
    model_rmse = rmse(y_true, y_pred)
    ref_rmse = rmse(y_true, y_reference)
    if ref_rmse <= EPSILON:
        return 0.0 if model_rmse <= EPSILON else float("nan")
    return 1.0 - model_rmse / ref_rmse


def horizon_rmse(y_true, y_pred) -> np.ndarray:
    """Per-horizon RMSE for arrays shaped (..., horizon[, channels])."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    horizon_axis = -1 if y_true.ndim == 1 else 1
    axes = tuple(i for i in range(y_true.ndim) if i != horizon_axis)
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=axes))


def horizon_divergence(y_true, y_pred) -> float:
    """Slope of per-horizon RMSE; lower values indicate more stable forecasts."""
    per_horizon = horizon_rmse(y_true, y_pred)
    if per_horizon.size < 2:
        return 0.0
    x = np.arange(per_horizon.size, dtype=np.float64)
    slope, _ = np.polyfit(x, per_horizon, deg=1)
    return float(slope)


def clip_pv_forecast(y_pred, capacity: float | None = None):
    """Apply PV physical bounds: non-negative and optionally <= capacity."""
    y_pred = _as_float_array(y_pred)
    upper = np.inf if capacity is None else float(capacity)
    return np.clip(y_pred, 0.0, upper)
