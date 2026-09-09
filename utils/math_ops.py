"""Public signatures for analytical operators described in the manuscript."""

from __future__ import annotations

import torch

from utils.release_scope import ComponentUnavailableError

EPSILON = 1e-6


def _unavailable(name: str):
    raise ComponentUnavailableError(
        f"{name} is not included in this source distribution. See README.md."
    )


def robust_svd_lowrank(H: torch.Tensor, q: int = 2, niter: int = 4):
    _unavailable("robust_svd_lowrank")


def power_iteration(H: torch.Tensor, n_iters: int = 2):
    _unavailable("power_iteration")


def build_diag_avg_indices(L: int, K: int, N: int, device: torch.device):
    idx = torch.zeros(L * K, dtype=torch.long, device=device)
    count = torch.ones(N, device=device)
    return idx, count


def hankel_to_1d(H: torch.Tensor, N: int, idx: torch.Tensor, count: torch.Tensor):
    _unavailable("hankel_to_1d")


def build_future_diag_avg_indices(L: int, horizon: int, device: torch.device):
    flat_pos = torch.zeros(1, dtype=torch.long, device=device)
    idx = torch.zeros(1, dtype=torch.long, device=device)
    count = torch.ones(horizon, device=device)
    return flat_pos, idx, count


def future_hankel_to_1d(
    H_future: torch.Tensor,
    horizon: int,
    flat_pos: torch.Tensor,
    idx: torch.Tensor,
    count: torch.Tensor,
):
    _unavailable("future_hankel_to_1d")
