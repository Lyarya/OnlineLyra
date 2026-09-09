"""Public interfaces for the analytical layers described in the manuscript."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.release_scope import ComponentUnavailableError
from utils.math_ops import EPSILON


def _unavailable(name: str):
    raise ComponentUnavailableError(
        f"{name} is not included in this source distribution. See README.md."
    )


class HankelOp:
    @staticmethod
    def hankelize(x: torch.Tensor, L: int | None = None) -> torch.Tensor:
        _unavailable("HankelOp.hankelize")


class LatentEvolutionProjector(nn.Module):
    def __init__(
        self,
        horizon: int,
        ridge_lambda: float = 1e-3,
        stability_eps: float = EPSILON,
    ):
        super().__init__()
        self.horizon = horizon
        self.ridge_lambda = ridge_lambda
        self.stability_eps = stability_eps

    def forward(self, V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        _unavailable("LatentEvolutionProjector.forward")


class SpectralResidualNormalizer:
    @staticmethod
    @torch.no_grad()
    def normalize(
        residual: torch.Tensor,
        hankel_L: int | None = None,
        n_power_iters: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _unavailable("SpectralResidualNormalizer.normalize")
