# ruff: noqa: E402
"""Smoke test for the interfaces included in this source distribution."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.OnlineLyraPV import OnlineLyraPV, ResidualMLP
from utils.release_scope import ComponentUnavailableError
from utils.energy_metrics import skill_score, normalized_rmse


def main() -> None:
    torch.manual_seed(2026)
    mlp = ResidualMLP(lookback=96, horizon=24, hidden_dim=32, use_instance_mean=False)
    y = mlp(torch.randn(2, 96))
    assert y.shape == (2, 24)

    model = OnlineLyraPV(lookback=96, horizon=24, capacity=1.0, hidden_dim=32)
    raised = False
    try:
        model.predict(torch.randn(1, 96))
    except ComponentUnavailableError:
        raised = True
    assert raised, "unavailable components should raise ComponentUnavailableError"

    print("source_distribution_ok=1")
    print(f"residual_mlp_out={tuple(y.shape)}")
    print(f"nrmse_helper={normalized_rmse([0.0, 1.0], [0.0, 1.0], capacity=1.0):.6f}")
    print(f"skill_helper={skill_score([0.0, 1.0], [0.0, 1.0], [0.0, 0.0]):.6f}")


if __name__ == "__main__":
    main()
