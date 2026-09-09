"""Offline Lyra interfaces and standard residual-learning components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Lyra_Layers import LatentEvolutionProjector
from utils.release_scope import ComponentUnavailableError
from utils.math_ops import build_diag_avg_indices, build_future_diag_avg_indices


class LyraDecomposer(nn.Module):
    FIELD_NAMES = (
        "residual_norm",
        "trend_forecast",
        "sigma_scale",
        "instance_mean",
        "revin_mean",
        "revin_stdev",
    )

    def __init__(
        self,
        lookback: int,
        horizon: int,
        hankel_L: int | None = None,
        cond_threshold: float = 1e6,
        n_power_iters: int = 2,
        ridge_lambda: float = 1.0,
        analytical_rank: int = 1,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.hankel_L = hankel_L if hankel_L is not None else lookback // 2
        self.K = lookback - self.hankel_L + 1
        self.n_power_iters = n_power_iters
        self.analytical_rank = analytical_rank
        self.projector = LatentEvolutionProjector(
            horizon=horizon,
            ridge_lambda=ridge_lambda,
        )
        idx, count = build_diag_avg_indices(
            self.hankel_L, self.K, lookback, device=torch.device("cpu")
        )
        self.register_buffer("diag_idx", idx)
        self.register_buffer("diag_count", count)
        future_pos, future_idx, future_count = build_future_diag_avg_indices(
            self.hankel_L, horizon, device=torch.device("cpu")
        )
        self.register_buffer("future_diag_pos", future_pos)
        self.register_buffer("future_diag_idx", future_idx)
        self.register_buffer("future_diag_count", future_count)

    def _bytes_per_seq_compute(self, dtype: torch.dtype = torch.float32) -> int:
        elem = torch.finfo(dtype).bits // 8
        hankel_elems = self.hankel_L * self.K
        return (hankel_elems * 4 + self.lookback * 3) * elem

    def _bytes_per_seq_output(self, dtype: torch.dtype = torch.float32) -> int:
        elem = torch.finfo(dtype).bits // 8
        return (self.lookback + self.horizon + 4) * elem

    @torch.no_grad()
    def _forward_impl(self, x: torch.Tensor):
        raise ComponentUnavailableError(
            "LyraDecomposer is not included in this source distribution. See README.md."
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, max_mem_gb: float = 8.0, cache_dir=None):
        raise ComponentUnavailableError(
            "LyraDecomposer is not included in this source distribution. See README.md."
        )

    @torch.no_grad()
    def decompose_to_disk(self, x, cache_dir, prefix: str = "lyra_cache", max_mem_gb: float = 6.0, device="cuda"):
        raise ComponentUnavailableError(
            "LyraDecomposer is not included in this source distribution. See README.md."
        )

    @staticmethod
    def load_cache(cache_dir: str | Path, prefix: str = "lyra_cache"):
        cache_dir = Path(cache_dir)
        result = {}
        for name in LyraDecomposer.FIELD_NAMES:
            fpath = cache_dir / f"{prefix}_{name}.npy"
            if not fpath.exists():
                raise FileNotFoundError(f"Cache not found: {fpath}")
            result[name] = np.load(str(fpath), mmap_mode="r")
        return result


class LyraCachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache: dict):
        self.cache = cache
        self.n_total = cache[LyraDecomposer.FIELD_NAMES[0]].shape[0]

    def __len__(self) -> int:
        return self.n_total

    def __getitem__(self, idx: int):
        return tuple(
            torch.from_numpy(np.asarray(self.cache[name][idx]).copy())
            for name in LyraDecomposer.FIELD_NAMES
        )


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        hidden_dim = getattr(configs, "d_model", 256)
        self.fc1 = nn.Linear(self.seq_len + 1, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, self.pred_len)
        self.dropout = nn.Dropout(getattr(configs, "dropout", 0.3))
        self.revin_gamma = nn.Parameter(torch.ones(1))
        self.revin_beta = nn.Parameter(torch.zeros(1))
        self.decomposer = LyraDecomposer(
            lookback=self.seq_len,
            horizon=self.pred_len,
            cond_threshold=getattr(configs, "cond_threshold", 1e6),
            n_power_iters=getattr(configs, "n_power_iters", 2),
            ridge_lambda=getattr(configs, "ridge_lambda", 1.0),
        )

    def forward_from_features(
        self,
        residual_norm,
        trend_forecast,
        sigma_scale,
        instance_mean,
        revin_mean,
        revin_stdev,
    ):
        mlp_input = torch.cat([residual_norm, instance_mean], dim=-1)
        h = self.fc1(mlp_input)
        h = F.glu(h, dim=-1)
        h = self.dropout(h)
        r_hat = self.fc2(h)
        y_hat = trend_forecast + r_hat * sigma_scale.unsqueeze(1) + instance_mean
        y_hat = (self.revin_gamma * y_hat + self.revin_beta) * revin_stdev + revin_mean
        return y_hat

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        raise ComponentUnavailableError(
            "Offline Lyra end-to-end inference is not included in this source distribution."
        )
