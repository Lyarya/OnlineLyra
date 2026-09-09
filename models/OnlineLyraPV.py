"""Online Lyra PV interfaces and standard online-learning components."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Lyra_Layers import LatentEvolutionProjector
from utils.release_scope import ComponentUnavailableError
from utils.math_ops import build_diag_avg_indices, build_future_diag_avg_indices


@dataclass
class OnlineAnalyticalState:
    trend_hist: torch.Tensor
    trend_forecast: torch.Tensor
    trend_forecast_work: torch.Tensor
    residual: torch.Tensor
    residual_norm: torch.Tensor
    residual_scale_work: torch.Tensor
    residual_scale: torch.Tensor
    instance_mean: torch.Tensor
    revin_mean: torch.Tensor
    revin_stdev: torch.Tensor
    scalar_a: torch.Tensor
    dominance_tau: torch.Tensor


class StreamingRank1PVForecaster(nn.Module):
    """Interface for the analytical PV forecaster described in the manuscript."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        hankel_L: int | None = None,
        ridge_lambda: float = 1e-3,
        n_power_iters: int = 2,
        capacity: float | None = None,
        normalization: str = "mean",
        residual_scale_mode: str = "spectral",
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.hankel_L = hankel_L if hankel_L is not None else lookback // 2
        self.K = lookback - self.hankel_L + 1
        self.ridge_lambda = ridge_lambda
        self.n_power_iters = n_power_iters
        self.capacity = capacity
        self.normalization = normalization
        self.residual_scale_mode = residual_scale_mode
        self.projector = LatentEvolutionProjector(
            horizon=horizon,
            ridge_lambda=ridge_lambda,
        )
        diag_idx, diag_count = build_diag_avg_indices(
            self.hankel_L, self.K, lookback, device=torch.device("cpu")
        )
        self.register_buffer("diag_idx", diag_idx)
        self.register_buffer("diag_count", diag_count)
        future_pos, future_idx, future_count = build_future_diag_avg_indices(
            self.hankel_L, horizon, device=torch.device("cpu")
        )
        self.register_buffer("future_diag_pos", future_pos)
        self.register_buffer("future_diag_idx", future_idx)
        self.register_buffer("future_diag_count", future_count)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> OnlineAnalyticalState:
        raise ComponentUnavailableError(
            "StreamingRank1PVForecaster is not included in this source distribution. "
            "See README.md."
        )


class ResidualReplayBuffer:
    def __init__(self, capacity: int = 1000):
        self.capacity = int(capacity)
        self._items: Deque[tuple[torch.Tensor, ...]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, state: OnlineAnalyticalState, y_true: torch.Tensor) -> None:
        fields = (
            state.residual_norm.detach().cpu(),
            state.instance_mean.detach().cpu(),
            state.trend_forecast.detach().cpu(),
            state.trend_forecast_work.detach().cpu(),
            state.residual_scale_work.detach().cpu(),
            state.residual_scale.detach().cpu(),
            state.revin_mean.detach().cpu(),
            state.revin_stdev.detach().cpu(),
            y_true.detach().cpu(),
        )
        for item in zip(*fields, strict=False):
            self._items.append(item)

    def sample(self, batch_size: int, device: torch.device):
        if batch_size <= 0 or not self._items:
            return None
        idx = torch.randint(0, len(self._items), (batch_size,))
        columns = zip(*(self._items[int(i)] for i in idx), strict=False)
        return tuple(torch.stack(values).to(device) for values in columns)


class ResidualMLP(nn.Module):
    """Standard GLU residual head used by the online wrapper."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_instance_mean: bool = True,
    ):
        super().__init__()
        self.use_instance_mean = use_instance_mean
        input_dim = lookback + (1 if use_instance_mean else 0)
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, horizon)

    def forward(
        self,
        residual_norm: torch.Tensor,
        instance_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_instance_mean:
            if instance_mean is None:
                raise ValueError("instance_mean is required when use_instance_mean=True.")
            residual_input = torch.cat([residual_norm, instance_mean], dim=-1)
        else:
            residual_input = residual_norm
        hidden = self.fc1(residual_input)
        hidden = F.glu(hidden, dim=-1)
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class OnlineLyraPV(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        capacity: float | None = None,
        hidden_dim: int = 64,
        replay_capacity: int = 1000,
        learning_rate: float = 1e-3,
        n_power_iters: int = 2,
        ridge_lambda: float = 1e-3,
        residual_dropout: float = 0.1,
        normalization: str = "mean",
        loss_mode: str = "forecast",
        use_instance_mean: bool = True,
        residual_scale_mode: str = "spectral",
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.loss_mode = loss_mode
        self.analytic = StreamingRank1PVForecaster(
            lookback=lookback,
            horizon=horizon,
            capacity=capacity,
            n_power_iters=n_power_iters,
            ridge_lambda=ridge_lambda,
            normalization=normalization,
            residual_scale_mode=residual_scale_mode,
        )
        self.residual = ResidualMLP(
            lookback=lookback,
            horizon=horizon,
            hidden_dim=hidden_dim,
            dropout=residual_dropout,
            use_instance_mean=use_instance_mean,
        )
        self.revin_gamma = nn.Parameter(torch.ones(1))
        self.revin_beta = nn.Parameter(torch.zeros(1))
        self.replay = ResidualReplayBuffer(capacity=replay_capacity)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.capacity = capacity
        self.to(self.device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        raise ComponentUnavailableError(
            "OnlineLyraPV.predict requires a component not included in this source distribution."
        )

    def online_update(self, x: torch.Tensor, y_true: torch.Tensor, steps: int = 3, replay_batch: int = 16) -> float:
        raise ComponentUnavailableError(
            "OnlineLyraPV.online_update requires a component not included in this source distribution."
        )

    def online_update_from_state(self, state, y_true, steps: int = 3, replay_batch: int = 16) -> float:
        raise ComponentUnavailableError(
            "OnlineLyraPV.online_update_from_state requires a component not included in this source distribution."
        )
