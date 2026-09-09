"""Run PV forecasting baselines under the Online Lyra protocol.

This script intentionally keeps the protocol identical to ``run_online_pv.py``:
one-year slice, first quarter for offline training, remaining quarters for
streaming-style evaluation windows. Baselines here are static/offline models;
they do not update during the online evaluation period.

The implemented baselines are meant to be reliable paper controls, not
renamed approximations of very recent official models. Recent models reported
by name are imported from their official repositories when public code is
available, while keeping this script's PV train/test protocol unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_online_pv import (  # noqa: E402
    clip_target_series,
    collect_files,
    infer_capacity_from_name,
    load_power_series,
    make_windows,
    masked_metric,
    safe_name,
    synthetic_pv_series,
)
from utils.energy_metrics import (  # noqa: E402
    horizon_divergence,
    mae,
    normalized_mae,
    normalized_rmse,
    rmse,
    skill_score,
)


DEFAULT_MODELS = [
    "persistence",
    "seasonal_naive",
    "nlinear",
    "dlinear",
    "patchtst",
    "lightgbm",
    "xgboost",
]


@dataclass
class BaselineSummary:
    farm: str
    file: str
    horizon: int
    horizon_minutes: int
    lookback: int
    stride: int
    capacity: float | None
    target_col: str
    model: str
    n_train_windows: int
    n_eval_windows: int
    mae: float
    rmse: float
    nmae: float
    nrmse: float
    daytime_points: int
    daytime_mae: float
    daytime_rmse: float
    daytime_nmae: float
    daytime_nrmse: float
    skill_vs_persistence: float
    horizon_divergence: float
    train_time_sec: float
    predict_latency_ms: float
    params: int | None
    checkpoint_size_kb: float | None
    notes: str


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = value
    return out


def load_official_module(
    repo_path: Path,
    model_file: Path,
    module_name: str,
    isolated_prefixes: tuple[str, ...] = ("layers", "utils", "models", "model"),
) -> ModuleType:
    """Import one official model file without leaking its common package names.

    Several forecasting repositories use top-level packages called ``layers`` or
    ``utils``. Isolating those imports lets us compare official implementations
    in a single process without silently mixing dependencies across repos.
    """
    repo_path = repo_path.expanduser().resolve()
    model_file = model_file if model_file.is_absolute() else repo_path / model_file
    if not model_file.exists():
        raise FileNotFoundError(f"Official model file not found at {model_file}")

    old_path = list(sys.path)
    old_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes)
    }
    sys.path.insert(0, str(repo_path))
    try:
        for name in old_modules:
            sys.modules.pop(name, None)
        for prefix in isolated_prefixes:
            pkg_dir = repo_path / prefix
            if pkg_dir.exists():
                package = ModuleType(prefix)
                package.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
                package.__package__ = prefix
                sys.modules[prefix] = package
        spec = importlib_util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {model_file}")
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path


def make_window_arrays(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    start: int,
    stop: int,
    stride: int,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in make_windows(series, lookback, horizon, start, stop, stride):
        xs.append(x)
        ys.append(y)
        if max_windows is not None and len(xs) >= max_windows:
            break
    if not xs:
        raise ValueError("No windows were generated.")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def normalize_train_eval(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(train_x))
    std = float(np.std(train_x))
    if std < 1e-6:
        std = 1.0
    return (
        (train_x - mean) / std,
        (train_y - mean) / std,
        (eval_x - mean) / std,
        mean,
        std,
    )


def persistence_predict(eval_x: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(eval_x[:, -1:], horizon, axis=1).astype(np.float32)


def seasonal_naive_predict(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    offline_points: int,
    stride: int,
    seasonal_period: int,
    n_eval: int,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    start0 = offline_points - lookback
    for step_idx in range(n_eval):
        idx = start0 + step_idx * stride
        future_start = idx + lookback
        pred = np.empty(horizon, dtype=np.float32)
        for h in range(horizon):
            src = future_start + h - seasonal_period
            if src >= 0:
                pred[h] = series[src]
            else:
                pred[h] = series[idx + lookback - 1]
        preds.append(pred)
    return np.stack(preds)


class NLinear(nn.Module):
    def __init__(self, lookback: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        last = x[:, -1:].detach()
        return self.linear(x - last) + last


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1].repeat(1, pad)
        end = x[:, -1:].repeat(1, self.kernel_size - 1 - pad)
        xp = torch.cat([front, x, end], dim=1).unsqueeze(1)
        return self.avg(xp).squeeze(1)


class DLinear(nn.Module):
    def __init__(self, lookback: int, horizon: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.decomp = MovingAverage(kernel_size)
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        return self.seasonal(seasonal) + self.trend(trend)


class PatchTSTSmall(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if lookback < patch_len:
            raise ValueError("lookback must be >= patch_len")
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patches = 1 + (lookback - patch_len) // patch_stride
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        z = self.patch_embed(patches) + self.pos
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


def build_torch_model(name: str, lookback: int, horizon: int, args: argparse.Namespace) -> nn.Module:
    if name == "nlinear":
        return NLinear(lookback, horizon)
    if name == "dlinear":
        return DLinear(lookback, horizon, kernel_size=args.dlinear_kernel)
    if name == "patchtst":
        return PatchTSTSmall(
            lookback,
            horizon,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    if name == "phaseformer":
        return OfficialPhaseFormerWrapper(
            repo_path=Path(args.phaseformer_repo),
            lookback=lookback,
            horizon=horizon,
            period_len=args.phaseformer_period_len,
            latent_dim=args.phaseformer_latent_dim,
            phase_encoder_hidden=args.phaseformer_encoder_hidden,
            predictor_hidden=args.phaseformer_predictor_hidden,
            phase_layers=args.phaseformer_layers,
            phase_num_routers=args.phaseformer_num_routers,
            phase_attn_heads=args.phaseformer_attn_heads,
            dropout=args.dropout,
        )
    if name == "mixlinear":
        return OfficialMixLinearWrapper(
            repo_path=Path(args.mixlinear_repo),
            lookback=lookback,
            horizon=horizon,
            period_len=args.mixlinear_period_len,
            lpf=args.mixlinear_lpf,
            alpha=args.mixlinear_alpha,
        )
    if name == "olivia_scratch":
        return OfficialOliviaScratchWrapper(
            repo_path=Path(args.olivia_repo),
            lookback=lookback,
            horizon=horizon,
            patch_len=args.olivia_patch_len,
            stride=args.olivia_stride,
            d_model=args.olivia_d_model,
            e_layers=args.olivia_e_layers,
            d_layers=args.olivia_d_layers,
            domain_len=args.olivia_domain_len,
        )
    if name in {"official_linear", "official_nlinear", "official_dlinear"}:
        return OfficialLTSFLinearWrapper(
            repo_path=Path(args.official_ltsf_repo),
            variant=name.removeprefix("official_"),
            lookback=lookback,
            horizon=horizon,
            individual=args.official_ltsf_individual,
        )
    if name == "official_patchtst":
        return OfficialPatchTSTWrapper(
            repo_path=Path(args.official_patchtst_repo),
            lookback=lookback,
            horizon=horizon,
            patch_len=args.official_patchtst_patch_len,
            patch_stride=args.official_patchtst_stride,
            d_model=args.official_patchtst_d_model,
            n_heads=args.official_patchtst_n_heads,
            n_layers=args.official_patchtst_layers,
            d_ff=args.official_patchtst_d_ff,
            dropout=args.dropout,
            fc_dropout=args.official_patchtst_fc_dropout,
            head_dropout=args.official_patchtst_head_dropout,
            revin=args.official_patchtst_revin,
            affine=args.official_patchtst_affine,
            subtract_last=args.official_patchtst_subtract_last,
            decomposition=args.official_patchtst_decomposition,
            kernel_size=args.official_patchtst_kernel,
        )
    if name == "itransformer":
        return OfficialITransformerWrapper(
            repo_path=Path(args.itransformer_repo),
            lookback=lookback,
            horizon=horizon,
            d_model=args.itransformer_d_model,
            n_heads=args.itransformer_n_heads,
            n_layers=args.itransformer_layers,
            d_ff=args.itransformer_d_ff,
            dropout=args.dropout,
            use_norm=args.itransformer_use_norm,
        )
    if name == "timekan":
        return OfficialTimeKANWrapper(
            repo_path=Path(args.timekan_repo),
            lookback=lookback,
            horizon=horizon,
            d_model=args.timekan_d_model,
            n_layers=args.timekan_layers,
            dropout=args.dropout,
            down_sampling_layers=args.timekan_down_sampling_layers,
            down_sampling_window=args.timekan_down_sampling_window,
            moving_avg=args.timekan_moving_avg,
            begin_order=args.timekan_begin_order,
            use_norm=args.timekan_use_norm,
        )
    raise ValueError(f"Unknown torch model: {name}")


class OfficialPhaseFormerWrapper(nn.Module):
    """Thin wrapper around the official PhaseFormer_TSL implementation.

    The official repository follows Time-Series-Library's dataset split. We
    import only the model definition and train it with this script's PV protocol.
    """

    def __init__(
        self,
        repo_path: Path,
        lookback: int,
        horizon: int,
        period_len: int,
        latent_dim: int,
        phase_encoder_hidden: int,
        predictor_hidden: int,
        phase_layers: int,
        phase_num_routers: int,
        phase_attn_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        repo_path = repo_path.expanduser().resolve()
        model_file = repo_path / "models" / "PhaseFormer.py"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Official PhaseFormer model not found at {model_file}. "
                "Pass --phaseformer_repo /path/to/PhaseFormer_TSL."
            )

        # The file imports `layers.*`; put the official repo first temporarily.
        old_path = list(sys.path)
        isolated_prefixes = ("layers", "utils")
        old_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes)
        }
        sys.path.insert(0, str(repo_path))
        try:
            for name in old_modules:
                sys.modules.pop(name, None)
            spec = importlib_util.spec_from_file_location("official_phaseformer_model", model_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {model_file}")
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for name in list(sys.modules):
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes):
                    sys.modules.pop(name, None)
            sys.modules.update(old_modules)
            sys.path[:] = old_path

        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=lookback,
            pred_len=horizon,
            enc_in=1,
            period_len=period_len,
            latent_dim=latent_dim,
            phase_encoder_hidden=phase_encoder_hidden,
            predictor_hidden=predictor_hidden,
            phase_attn_heads=phase_attn_heads,
            phase_attn_dropout=dropout,
            phase_attn_use_relpos=True,
            phase_attn_window=None,
            phase_attention_dim=None,
            phase_num_routers=phase_num_routers,
            phase_use_pos_embed=True,
            phase_pos_dropout=dropout,
            use_revin=True,
            revin_affine=False,
            revin_eps=1e-5,
            phase_layers=phase_layers,
            phase_encoder_use_mlp=False,
            phase_encoder_dropout=dropout,
            predictor_use_mlp=False,
            predictor_dropout=dropout,
        )
        self.model = module.Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.unsqueeze(-1))
        return y.squeeze(-1)


class OfficialMixLinearWrapper(nn.Module):
    """Thin wrapper around the official MixLinear implementation."""

    def __init__(
        self,
        repo_path: Path,
        lookback: int,
        horizon: int,
        period_len: int,
        lpf: int,
        alpha: float,
    ) -> None:
        super().__init__()
        repo_path = repo_path.expanduser().resolve()
        model_file = repo_path / "models" / "MixLinear.py"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Official MixLinear model not found at {model_file}. "
                "Pass --mixlinear_repo /path/to/MixLinear."
            )

        isolated_prefixes = ("layers", "utils")
        old_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes)
        }
        sys.path.insert(0, str(repo_path))
        try:
            for name in old_modules:
                sys.modules.pop(name, None)
            for prefix in isolated_prefixes:
                pkg_dir = repo_path / prefix
                if pkg_dir.exists():
                    package = ModuleType(prefix)
                    package.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
                    package.__package__ = prefix
                    sys.modules[prefix] = package
            spec = importlib_util.spec_from_file_location("official_mixlinear_model", model_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {model_file}")
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for name in list(sys.modules):
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes):
                    sys.modules.pop(name, None)
            sys.modules.update(old_modules)
            sys.path[:] = old_path

        configs = SimpleNamespace(
            seq_len=lookback,
            pred_len=horizon,
            enc_in=1,
            period_len=period_len,
            lpf=lpf,
            alpha=alpha,
        )
        self.model = module.Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The official implementation prints tensor shapes inside forward.
        # Suppress that debug output so long runs keep readable logs.
        with contextlib.redirect_stdout(io.StringIO()):
            y = self.model(x.unsqueeze(-1))
        return y.squeeze(-1)


class OfficialOliviaScratchWrapper(nn.Module):
    """Use Olivia's official architecture as a fully supervised scratch model.

    Olivia's released code is designed around pretraining/foundation-model
    protocols. In this comparison, the architecture is imported from the
    official repository, all parameters are unfrozen, and training uses only
    the same offline PV split as the other supervised baselines. Results are
    therefore labeled as scratch training rather than pretrained inference.
    """

    def __init__(
        self,
        repo_path: Path,
        lookback: int,
        horizon: int,
        patch_len: int,
        stride: int,
        d_model: int,
        e_layers: int,
        d_layers: int,
        domain_len: int,
    ) -> None:
        super().__init__()
        repo_path = repo_path.expanduser().resolve()
        model_file = repo_path / "models" / "Olivia.py"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Official Olivia model not found at {model_file}. "
                "Pass --olivia_repo /path/to/Olivia."
            )

        if patch_len > lookback:
            raise ValueError(f"Olivia patch_len={patch_len} must be <= lookback={lookback}")
        if d_model % 16 != 0:
            raise ValueError("Olivia's official model uses 16 heads; --olivia_d_model must be divisible by 16.")

        isolated_prefixes = ("layers", "utils")
        old_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes)
        }
        sys.path.insert(0, str(repo_path))
        try:
            for name in old_modules:
                sys.modules.pop(name, None)
            spec = importlib_util.spec_from_file_location("official_olivia_model", model_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {model_file}")
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            for name in list(sys.modules):
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in isolated_prefixes):
                    sys.modules.pop(name, None)
            sys.modules.update(old_modules)
            sys.path[:] = old_path

        configs = SimpleNamespace(
            head_type="prediction",
            c_in=1,
            seq_len=lookback,
            pred_len=horizon,
            label_len=0,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            e_layers=e_layers,
            d_layers=d_layers,
            domain_len=domain_len,
            horizon_lengths=[horizon],
            data="PV",
            setting="pv_scratch",
        )
        self.model = module.Model(configs)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.model(x.unsqueeze(-1), None, None, None)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.squeeze(-1)


class OfficialLTSFLinearWrapper(nn.Module):
    """Wrapper for the official LTSF-Linear Linear/NLinear/DLinear models."""

    MODEL_FILES = {
        "linear": "Linear.py",
        "nlinear": "NLinear.py",
        "dlinear": "DLinear.py",
    }

    def __init__(
        self,
        repo_path: Path,
        variant: str,
        lookback: int,
        horizon: int,
        individual: bool,
    ) -> None:
        super().__init__()
        if variant not in self.MODEL_FILES:
            raise ValueError(f"Unknown official LTSF-Linear variant: {variant}")
        repo_path = repo_path.expanduser().resolve()
        model_file = repo_path / "models" / self.MODEL_FILES[variant]
        module = load_official_module(
            repo_path,
            model_file,
            f"official_ltsf_{variant}_model",
            isolated_prefixes=("layers", "utils", "models"),
        )
        configs = SimpleNamespace(
            seq_len=lookback,
            pred_len=horizon,
            enc_in=1,
            individual=individual,
        )
        self.model = module.Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.unsqueeze(-1))
        return y.squeeze(-1)


class OfficialPatchTSTWrapper(nn.Module):
    """Wrapper for the supervised official PatchTST implementation."""

    def __init__(
        self,
        repo_path: Path,
        lookback: int,
        horizon: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        fc_dropout: float,
        head_dropout: float,
        revin: bool,
        affine: bool,
        subtract_last: bool,
        decomposition: bool,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if patch_len > lookback:
            raise ValueError(f"PatchTST patch_len={patch_len} must be <= lookback={lookback}")
        if d_model % n_heads != 0:
            raise ValueError("PatchTST d_model must be divisible by n_heads.")

        repo_path = repo_path.expanduser().resolve()
        supervised_root = repo_path / "PatchTST_supervised" if (repo_path / "PatchTST_supervised").exists() else repo_path
        model_file = supervised_root / "models" / "PatchTST.py"
        module = load_official_module(
            supervised_root,
            model_file,
            "official_patchtst_model",
            isolated_prefixes=("layers", "utils", "models"),
        )
        configs = SimpleNamespace(
            enc_in=1,
            seq_len=lookback,
            pred_len=horizon,
            e_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            individual=False,
            patch_len=patch_len,
            stride=patch_stride,
            padding_patch="end",
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            decomposition=decomposition,
            kernel_size=kernel_size,
        )
        self.model = module.Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.unsqueeze(-1))
        return y.squeeze(-1)


class OfficialITransformerWrapper(nn.Module):
    """Wrapper for the official iTransformer forecasting model."""

    def __init__(
        self,
        repo_path: Path,
        lookback: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        use_norm: bool,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("iTransformer d_model must be divisible by n_heads.")
        repo_path = repo_path.expanduser().resolve()
        model_file = repo_path / "model" / "iTransformer.py"
        old_reformer = sys.modules.get("reformer_pytorch")
        old_einops = sys.modules.get("einops")
        created_reformer_stub = False
        created_einops_stub = False
        try:
            __import__("reformer_pytorch")
        except Exception:
            reformer_stub = ModuleType("reformer_pytorch")

            class MissingLSHSelfAttention(nn.Module):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__()
                    raise ImportError("reformer-pytorch is required only for Reformer models, not iTransformer.")

            reformer_stub.LSHSelfAttention = MissingLSHSelfAttention  # type: ignore[attr-defined]
            sys.modules["reformer_pytorch"] = reformer_stub
            created_reformer_stub = True
        try:
            __import__("einops")
        except Exception:
            einops_stub = ModuleType("einops")

            def missing_rearrange(*args: Any, **kwargs: Any) -> None:
                raise ImportError("einops is required only for FlashAttention models, not iTransformer.")

            einops_stub.rearrange = missing_rearrange  # type: ignore[attr-defined]
            sys.modules["einops"] = einops_stub
            created_einops_stub = True
        try:
            module = load_official_module(
                repo_path,
                model_file,
                "official_itransformer_model",
                isolated_prefixes=("layers", "utils", "model"),
            )
        finally:
            if created_reformer_stub:
                if old_reformer is None:
                    sys.modules.pop("reformer_pytorch", None)
                else:
                    sys.modules["reformer_pytorch"] = old_reformer
            if created_einops_stub:
                if old_einops is None:
                    sys.modules.pop("einops", None)
                else:
                    sys.modules["einops"] = old_einops
        configs = SimpleNamespace(
            seq_len=lookback,
            pred_len=horizon,
            output_attention=False,
            use_norm=use_norm,
            d_model=d_model,
            embed="timeF",
            freq="t",
            dropout=dropout,
            class_strategy="projection",
            factor=1,
            n_heads=n_heads,
            e_layers=n_layers,
            d_ff=d_ff,
            activation="gelu",
        )
        self.model = module.Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.unsqueeze(-1), None, None, None)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.squeeze(-1)


class OfficialTimeKANWrapper(nn.Module):
    """Wrapper for the official TimeKAN long-term forecasting model."""

    def __init__(
        self,
        repo_path: Path,
        lookback: int,
        horizon: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        down_sampling_layers: int,
        down_sampling_window: int,
        moving_avg: int,
        begin_order: int,
        use_norm: int,
    ) -> None:
        super().__init__()
        scale = down_sampling_window ** down_sampling_layers
        if scale <= 0 or lookback < scale or lookback % scale != 0:
            raise ValueError(
                "TimeKAN requires lookback to be divisible by "
                f"down_sampling_window ** down_sampling_layers; got lookback={lookback}, scale={scale}."
            )
        repo_path = repo_path.expanduser().resolve()
        model_file = repo_path / "models" / "TimeKAN.py"
        module = load_official_module(
            repo_path,
            model_file,
            "official_timekan_model",
            isolated_prefixes=("layers", "utils", "models"),
        )
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=lookback,
            label_len=0,
            pred_len=horizon,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            channel_independence=1,
            e_layers=n_layers,
            moving_avg=moving_avg,
            enc_in=1,
            c_out=1,
            use_future_temporal_feature=False,
            embed="timeF",
            freq="t",
            dropout=dropout,
            d_model=d_model,
            begin_order=begin_order,
            use_norm=use_norm,
        )
        self.model = module.Model(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x.unsqueeze(-1), None, None, None)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.squeeze(-1)


def train_torch_baseline(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    args: argparse.Namespace,
    artifact_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_xn, train_yn, eval_xn, mean, std = normalize_train_eval(train_x, train_y, eval_x)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_torch_model(name, train_x.shape[1], train_y.shape[1], args).to(device)
    params = count_params(model)

    x_tensor = torch.from_numpy(train_xn)
    y_tensor = torch.from_numpy(train_yn)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    logs: list[dict[str, float | int | str]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = args.patience
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        logs.append({"model": name, "epoch": epoch, "train_mse": epoch_loss})
        if epoch_loss + 1e-8 < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    predict_times = []
    with torch.no_grad():
        for start in range(0, len(eval_xn), args.eval_batch_size):
            xb = torch.from_numpy(eval_xn[start:start + args.eval_batch_size]).to(device)
            t1 = time.perf_counter()
            pred = model(xb).detach().cpu().numpy()
            predict_times.append((time.perf_counter() - t1) / max(len(xb), 1))
            preds.append(pred)
    pred_np = np.concatenate(preds, axis=0) * std + mean

    checkpoint_size_kb = None
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": name,
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "params": params,
            "args": vars(args),
        }
        torch.save(checkpoint, artifact_path)
        checkpoint_size_kb = artifact_path.stat().st_size / 1024.0
        pd.DataFrame(logs).to_csv(artifact_path.with_suffix(".training_log.csv"), index=False)

    return pred_np.astype(np.float32), {
        "train_time_sec": train_time,
        "predict_latency_ms": float(np.mean(predict_times) * 1000.0),
        "params": params,
        "checkpoint_size_kb": checkpoint_size_kb,
        "notes": f"epochs={len(logs)}, best_train_mse={best_loss:.6g}",
    }


def train_tree_baseline(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("xgboost is not installed") from exc
        base = XGBRegressor(
            n_estimators=args.tree_estimators,
            max_depth=args.tree_max_depth,
            learning_rate=args.tree_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=args.num_workers,
            random_state=args.seed,
        )
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("lightgbm is not installed") from exc
        base = LGBMRegressor(
            n_estimators=args.tree_estimators,
            num_leaves=args.lightgbm_num_leaves,
            learning_rate=args.tree_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.seed,
            n_jobs=args.num_workers,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown tree baseline: {name}")

    horizon = train_y.shape[1]
    train_steps = np.tile(np.arange(horizon, dtype=np.float32) / max(horizon - 1, 1), len(train_x))
    train_features = np.repeat(train_x, horizon, axis=0)
    train_features = np.concatenate([train_features, train_steps[:, None]], axis=1)
    train_targets = train_y.reshape(-1)
    if args.tree_max_rows and len(train_targets) > args.tree_max_rows:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(train_targets), size=args.tree_max_rows, replace=False)
        train_features = train_features[keep]
        train_targets = train_targets[keep]

    t0 = time.perf_counter()
    base.fit(train_features, train_targets)
    train_time = time.perf_counter() - t0

    eval_steps = np.tile(np.arange(horizon, dtype=np.float32) / max(horizon - 1, 1), len(eval_x))
    eval_features = np.repeat(eval_x, horizon, axis=0)
    eval_features = np.concatenate([eval_features, eval_steps[:, None]], axis=1)
    t1 = time.perf_counter()
    pred = base.predict(eval_features).reshape(len(eval_x), horizon).astype(np.float32)
    predict_latency_ms = (time.perf_counter() - t1) / max(len(eval_x), 1) * 1000.0
    return pred, {
        "train_time_sec": train_time,
        "predict_latency_ms": predict_latency_ms,
        "params": None,
        "checkpoint_size_kb": None,
        "notes": f"n_estimators={args.tree_estimators}, direct_horizon_feature=True",
    }


def run_sarima_baseline(
    train_series: np.ndarray,
    eval_x: np.ndarray,
    horizon: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:
        raise RuntimeError("statsmodels is not installed") from exc

    order = tuple(args.sarima_order)
    seasonal_order = tuple(args.sarima_seasonal_order)
    fit_series = train_series[-args.sarima_fit_points:] if args.sarima_fit_points else train_series
    t0 = time.perf_counter()
    res = SARIMAX(
        fit_series.astype(np.float64),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=args.sarima_maxiter)
    train_time = time.perf_counter() - t0

    preds = []
    times = []
    # Re-filter each lookback window using fitted parameters. This is slower than
    # persistence but keeps the evaluation rolling and avoids future leakage.
    for x in eval_x:
        t1 = time.perf_counter()
        fitted = SARIMAX(
            x.astype(np.float64),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).filter(res.params)
        pred = fitted.forecast(horizon)
        times.append(time.perf_counter() - t1)
        preds.append(np.asarray(pred, dtype=np.float32))
    return np.stack(preds), {
        "train_time_sec": train_time,
        "predict_latency_ms": float(np.mean(times) * 1000.0),
        "params": int(len(res.params)),
        "checkpoint_size_kb": None,
        "notes": f"order={order}, seasonal_order={seasonal_order}, aic={res.aic:.3f}",
    }


def summarize_predictions(
    farm_name: str,
    file_name: str,
    target_col: str,
    capacity: float | None,
    horizon: int,
    lookback: int,
    stride: int,
    model_name: str,
    train_x: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    persistence_pred: np.ndarray,
    daytime_threshold_ratio: float,
    extra: dict[str, Any],
) -> BaselineSummary:
    cap = capacity if capacity is not None else float(np.nanmax(y_true))
    pred = np.clip(pred, 0.0, cap).astype(np.float32)
    daytime_mask = y_true > cap * daytime_threshold_ratio
    return BaselineSummary(
        farm=farm_name,
        file=file_name,
        horizon=horizon,
        horizon_minutes=horizon * 15,
        lookback=lookback,
        stride=stride,
        capacity=capacity,
        target_col=target_col,
        model=model_name,
        n_train_windows=len(train_x),
        n_eval_windows=len(y_true),
        mae=mae(y_true, pred),
        rmse=rmse(y_true, pred),
        nmae=normalized_mae(y_true, pred, capacity=cap),
        nrmse=normalized_rmse(y_true, pred, capacity=cap),
        daytime_points=int(np.sum(daytime_mask)),
        daytime_mae=masked_metric(mae, y_true, pred, daytime_mask),
        daytime_rmse=masked_metric(rmse, y_true, pred, daytime_mask),
        daytime_nmae=masked_metric(normalized_mae, y_true, pred, daytime_mask, cap),
        daytime_nrmse=masked_metric(normalized_rmse, y_true, pred, daytime_mask, cap),
        skill_vs_persistence=skill_score(y_true, pred, persistence_pred),
        horizon_divergence=horizon_divergence(y_true, pred),
        train_time_sec=float(extra.get("train_time_sec", 0.0)),
        predict_latency_ms=float(extra.get("predict_latency_ms", 0.0)),
        params=extra.get("params"),
        checkpoint_size_kb=extra.get("checkpoint_size_kb"),
        notes=str(extra.get("notes", "")),
    )


def run_one_horizon(
    series: np.ndarray,
    farm_name: str,
    file_name: str,
    target_col: str,
    capacity: float | None,
    horizon: int,
    args: argparse.Namespace,
    artifact_dir: Path | None,
) -> list[BaselineSummary]:
    stride = args.stride or horizon
    offline_points = args.offline_days * 96
    train_x, train_y = make_window_arrays(
        series,
        args.lookback,
        horizon,
        0,
        offline_points,
        args.train_stride,
        max_windows=args.max_train_windows,
    )
    eval_x, y_true = make_window_arrays(
        series,
        args.lookback,
        horizon,
        offline_points - args.lookback,
        len(series),
        stride,
        max_windows=args.max_eval_windows,
    )
    persistence_pred = persistence_predict(eval_x, horizon)

    summaries: list[BaselineSummary] = []
    predictions: dict[str, np.ndarray] = {
        "y_true": y_true,
        "persistence": persistence_pred,
    }

    for model_name in args.models:
        t0 = time.perf_counter()
        notes = ""
        try:
            if model_name == "persistence":
                pred = persistence_pred
                extra = {
                    "train_time_sec": 0.0,
                    "predict_latency_ms": 0.0,
                    "params": None,
                    "checkpoint_size_kb": None,
                    "notes": "last-value operational baseline",
                }
            elif model_name == "seasonal_naive":
                pred = seasonal_naive_predict(
                    series,
                    args.lookback,
                    horizon,
                    offline_points,
                    stride,
                    args.seasonal_period,
                    len(y_true),
                )
                extra = {
                    "train_time_sec": 0.0,
                    "predict_latency_ms": 0.0,
                    "params": None,
                    "checkpoint_size_kb": None,
                    "notes": f"seasonal_period={args.seasonal_period}",
                }
            elif model_name in {
                "nlinear",
                "dlinear",
                "patchtst",
                "phaseformer",
                "mixlinear",
                "olivia_scratch",
                "official_linear",
                "official_nlinear",
                "official_dlinear",
                "official_patchtst",
                "itransformer",
                "timekan",
            }:
                ckpt = None
                if artifact_dir is not None:
                    ckpt = artifact_dir / f"{safe_name(farm_name)}_h{horizon}_{model_name}.pt"
                pred, extra = train_torch_baseline(model_name, train_x, train_y, eval_x, args, ckpt)
            elif model_name in {"xgboost", "lightgbm"}:
                pred, extra = train_tree_baseline(model_name, train_x, train_y, eval_x, args)
            elif model_name in {"sarima", "arima"}:
                pred, extra = run_sarima_baseline(
                    series[:offline_points],
                    eval_x,
                    horizon,
                    args,
                )
            else:
                raise ValueError(f"Unknown baseline model: {model_name}")
            extra.setdefault("train_time_sec", time.perf_counter() - t0)
            predictions[model_name] = np.clip(
                pred,
                0.0,
                capacity if capacity is not None else float(np.nanmax(series)),
            ).astype(np.float32)
            summary = summarize_predictions(
                farm_name=farm_name,
                file_name=file_name,
                target_col=target_col,
                capacity=capacity,
                horizon=horizon,
                lookback=args.lookback,
                stride=stride,
                model_name=model_name,
                train_x=train_x,
                y_true=y_true,
                pred=predictions[model_name],
                persistence_pred=persistence_pred,
                daytime_threshold_ratio=args.daytime_threshold_ratio,
                extra=extra,
            )
            summaries.append(summary)
            print(
                f"{farm_name} H={horizon} {model_name}: "
                f"nRMSE={summary.nrmse:.4f}, day_nRMSE={summary.daytime_nrmse:.4f}, "
                f"skill={summary.skill_vs_persistence:.4f}, train={summary.train_time_sec:.1f}s",
                flush=True,
            )
        except Exception as exc:
            notes = f"failed: {type(exc).__name__}: {exc}"
            print(f"{farm_name} H={horizon} {model_name}: {notes}", flush=True)
            if args.strict:
                raise
            summaries.append(
                BaselineSummary(
                    farm=farm_name,
                    file=file_name,
                    horizon=horizon,
                    horizon_minutes=horizon * 15,
                    lookback=args.lookback,
                    stride=stride,
                    capacity=capacity,
                    target_col=target_col,
                    model=model_name,
                    n_train_windows=len(train_x),
                    n_eval_windows=len(y_true),
                    mae=float("nan"),
                    rmse=float("nan"),
                    nmae=float("nan"),
                    nrmse=float("nan"),
                    daytime_points=0,
                    daytime_mae=float("nan"),
                    daytime_rmse=float("nan"),
                    daytime_nmae=float("nan"),
                    daytime_nrmse=float("nan"),
                    skill_vs_persistence=float("nan"),
                    horizon_divergence=float("nan"),
                    train_time_sec=float("nan"),
                    predict_latency_ms=float("nan"),
                    params=None,
                    checkpoint_size_kb=None,
                    notes=notes,
                )
            )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{safe_name(farm_name)}_h{horizon}_baselines"
        np.savez_compressed(
            artifact_dir / f"{prefix}_predictions.npz",
            capacity=np.asarray(capacity if capacity is not None else np.nan, dtype=np.float32),
            horizon=np.asarray(horizon, dtype=np.int32),
            **predictions,
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PV static baseline experiments")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--glob", type=str, default="*.xlsx")
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48, 96])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--offline_days", type=int, default=91)
    parser.add_argument("--year_days", type=int, default=365)
    parser.add_argument("--clip_target", dest="clip_target", action="store_true", default=True)
    parser.add_argument("--no_clip_target", dest="clip_target", action="store_false")
    parser.add_argument("--daytime_threshold_ratio", type=float, default=0.01)
    parser.add_argument("--seasonal_period", type=int, default=96)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_eval_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/pv_baselines"))
    parser.add_argument("--save_artifacts", action="store_true")
    parser.add_argument("--artifact_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strict", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dlinear_kernel", type=int, default=25)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--phaseformer_repo", type=Path, default=ROOT / "external" / "PhaseFormer_TSL")
    parser.add_argument("--phaseformer_period_len", type=int, default=24)
    parser.add_argument("--phaseformer_latent_dim", type=int, default=32)
    parser.add_argument("--phaseformer_encoder_hidden", type=int, default=64)
    parser.add_argument("--phaseformer_predictor_hidden", type=int, default=128)
    parser.add_argument("--phaseformer_layers", type=int, default=2)
    parser.add_argument("--phaseformer_num_routers", type=int, default=1)
    parser.add_argument("--phaseformer_attn_heads", type=int, default=4)
    parser.add_argument("--mixlinear_repo", type=Path, default=ROOT / "external" / "MixLinear")
    parser.add_argument("--mixlinear_period_len", type=int, default=24)
    parser.add_argument("--mixlinear_lpf", type=int, default=2)
    parser.add_argument("--mixlinear_alpha", type=float, default=0.5)
    parser.add_argument("--olivia_repo", type=Path, default=ROOT / "external" / "Olivia")
    parser.add_argument("--olivia_patch_len", type=int, default=64)
    parser.add_argument("--olivia_stride", type=int, default=32)
    parser.add_argument("--olivia_d_model", type=int, default=64)
    parser.add_argument("--olivia_e_layers", type=int, default=1)
    parser.add_argument("--olivia_d_layers", type=int, default=1)
    parser.add_argument("--olivia_domain_len", type=int, default=128)
    parser.add_argument("--official_ltsf_repo", type=Path, default=ROOT / "external" / "LTSF-Linear")
    parser.add_argument("--official_ltsf_individual", action="store_true")
    parser.add_argument("--official_patchtst_repo", type=Path, default=ROOT / "external" / "PatchTST")
    parser.add_argument("--official_patchtst_patch_len", type=int, default=16)
    parser.add_argument("--official_patchtst_stride", type=int, default=8)
    parser.add_argument("--official_patchtst_d_model", type=int, default=64)
    parser.add_argument("--official_patchtst_n_heads", type=int, default=4)
    parser.add_argument("--official_patchtst_layers", type=int, default=2)
    parser.add_argument("--official_patchtst_d_ff", type=int, default=256)
    parser.add_argument("--official_patchtst_fc_dropout", type=float, default=0.1)
    parser.add_argument("--official_patchtst_head_dropout", type=float, default=0.1)
    parser.add_argument("--official_patchtst_revin", dest="official_patchtst_revin", action="store_true", default=True)
    parser.add_argument("--official_patchtst_no_revin", dest="official_patchtst_revin", action="store_false")
    parser.add_argument("--official_patchtst_affine", action="store_true")
    parser.add_argument("--official_patchtst_subtract_last", action="store_true")
    parser.add_argument("--official_patchtst_decomposition", action="store_true")
    parser.add_argument("--official_patchtst_kernel", type=int, default=25)
    parser.add_argument("--itransformer_repo", type=Path, default=ROOT / "external" / "iTransformer")
    parser.add_argument("--itransformer_d_model", type=int, default=64)
    parser.add_argument("--itransformer_n_heads", type=int, default=4)
    parser.add_argument("--itransformer_layers", type=int, default=2)
    parser.add_argument("--itransformer_d_ff", type=int, default=256)
    parser.add_argument("--itransformer_use_norm", dest="itransformer_use_norm", action="store_true", default=True)
    parser.add_argument("--itransformer_no_norm", dest="itransformer_use_norm", action="store_false")
    parser.add_argument("--timekan_repo", type=Path, default=ROOT / "external" / "TimeKAN")
    parser.add_argument("--timekan_d_model", type=int, default=64)
    parser.add_argument("--timekan_layers", type=int, default=2)
    parser.add_argument("--timekan_down_sampling_layers", type=int, default=2)
    parser.add_argument("--timekan_down_sampling_window", type=int, default=2)
    parser.add_argument("--timekan_moving_avg", type=int, default=25)
    parser.add_argument("--timekan_begin_order", type=int, default=1)
    parser.add_argument("--timekan_use_norm", type=int, default=1)

    parser.add_argument("--tree_estimators", type=int, default=200)
    parser.add_argument("--tree_max_depth", type=int, default=4)
    parser.add_argument("--tree_learning_rate", type=float, default=0.05)
    parser.add_argument("--lightgbm_num_leaves", type=int, default=31)
    parser.add_argument("--tree_max_rows", type=int, default=200000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--sarima_order", type=int, nargs=3, default=[1, 0, 1])
    parser.add_argument("--sarima_seasonal_order", type=int, nargs=4, default=[1, 0, 0, 96])
    parser.add_argument("--sarima_fit_points", type=int, default=7 * 96)
    parser.add_argument("--sarima_maxiter", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.artifact_dir or (args.output_dir / "artifacts")
    if not args.save_artifacts:
        artifact_dir = None

    points_per_day = 96
    year_points = args.year_days * points_per_day
    all_summaries: list[BaselineSummary] = []
    metadata: dict[str, Any] = {
        "protocol": "Static baselines trained on Q1, evaluated on Q2-Q4 rolling windows.",
        "resolution_minutes": 15,
        "args": jsonable_args(args) | {
            "file": str(args.file) if args.file else None,
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "output_dir": str(args.output_dir),
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
        },
        "files": [],
    }

    if args.synthetic:
        capacity = args.capacity or 50.0
        series = synthetic_pv_series(args.year_days, capacity=capacity)
        files_and_series = [("synthetic_pv", "synthetic", series, capacity, "synthetic_power")]
    else:
        files_and_series = []
        files = collect_files(args)
        if args.max_files is not None:
            files = files[:args.max_files]
        for path in files:
            capacity = args.capacity if args.capacity is not None else infer_capacity_from_name(path)
            series, target_col, time_col = load_power_series(
                path,
                target_col=args.target_col,
                time_col=args.time_col,
                sheet=args.sheet,
                year_points=year_points,
            )
            clip_diagnostics = None
            if args.clip_target:
                series, clip_diagnostics = clip_target_series(series, capacity)
            files_and_series.append((path.stem, str(path), series, capacity, target_col))
            file_metadata = {
                "file": str(path),
                "target_col": target_col,
                "time_col": time_col,
                "capacity": capacity,
                "points": len(series),
                "clip_target": args.clip_target,
            }
            if clip_diagnostics is not None:
                file_metadata.update(clip_diagnostics)
            metadata["files"].append(file_metadata)

    for farm_name, file_name, series, capacity, target_col in files_and_series:
        for horizon in args.horizons:
            summaries = run_one_horizon(
                series=series,
                farm_name=farm_name,
                file_name=file_name,
                target_col=target_col,
                capacity=capacity,
                horizon=horizon,
                args=args,
                artifact_dir=artifact_dir,
            )
            all_summaries.extend(summaries)
            pd.DataFrame([asdict(s) for s in all_summaries]).to_csv(
                args.output_dir / "pv_baseline_summary.csv",
                index=False,
            )

    summary_path = args.output_dir / "pv_baseline_summary.csv"
    metadata_path = args.output_dir / "pv_baseline_metadata.json"
    pd.DataFrame([asdict(s) for s in all_summaries]).to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
