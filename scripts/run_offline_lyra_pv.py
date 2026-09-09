"""Run the original offline Lyra model under the aligned PV protocol.

This script is the "Lyra without online adaptation" control for the paper:

* same one-year PV slice as the online experiments,
* Q1 offline train/validation only,
* Q2-Q4 rolling evaluation windows,
* same lookback and horizons as the main table,
* original Lyra decomposition + GLU residual learner.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.Lyra import Model as LyraModel  # noqa: E402
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


@dataclass
class OfflineLyraSummary:
    farm: str
    file: str
    horizon: int
    horizon_minutes: int
    lookback: int
    stride: int
    train_stride: int
    capacity: float | None
    target_col: str
    model: str
    n_train_windows: int
    n_val_windows: int
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
    params: int
    checkpoint_size_kb: float | None
    best_epoch: int
    best_val_loss: float
    notes: str


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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def persistence_predict(eval_x: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(eval_x[:, -1:], horizon, axis=1).astype(np.float32)


def build_lyra(args: argparse.Namespace, horizon: int, capacity: float | None, device: torch.device) -> LyraModel:
    configs = SimpleNamespace(
        seq_len=args.lookback,
        pred_len=horizon,
        enc_in=1,
        d_model=args.d_model,
        dropout=args.dropout,
        cond_threshold=args.cond_threshold,
        n_power_iters=args.n_power_iters,
        ridge_lambda=args.ridge_lambda,
    )
    model = LyraModel(configs).to(device)
    model.capacity = capacity
    return model


@torch.no_grad()
def decompose_features(
    model: LyraModel,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    pieces: list[list[torch.Tensor]] = [[] for _ in range(6)]
    model.eval()
    for start in range(0, len(x_np), batch_size):
        xb = torch.from_numpy(x_np[start:start + batch_size]).to(device)
        outputs = model.decomposer(xb)
        for bucket, tensor in zip(pieces, outputs, strict=False):
            bucket.append(tensor.detach().cpu())
    return tuple(torch.cat(bucket, dim=0) for bucket in pieces)


def feature_dataset(features: tuple[torch.Tensor, ...], y_np: np.ndarray) -> TensorDataset:
    y = torch.from_numpy(y_np.astype(np.float32))
    return TensorDataset(*features, y)


def forward_from_batch(model: LyraModel, batch: tuple[torch.Tensor, ...], device: torch.device) -> torch.Tensor:
    residual_norm, trend_forecast, sigma_scale, instance_mean, revin_mean, revin_stdev, _y = batch
    return model.forward_from_features(
        residual_norm.to(device),
        trend_forecast.to(device),
        sigma_scale.to(device),
        instance_mean.to(device),
        revin_mean.to(device),
        revin_stdev.to(device),
    )


def train_offline_lyra(
    train_x: np.ndarray,
    train_y: np.ndarray,
    horizon: int,
    capacity: float | None,
    args: argparse.Namespace,
    artifact_prefix: Path | None,
) -> tuple[LyraModel, dict[str, Any]]:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_lyra(args, horizon, capacity, device)
    params = count_params(model)

    n_total = len(train_x)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = max(1, n_total - n_val)
    train_feat = decompose_features(model, train_x[:n_train], device, args.decompose_batch_size)
    val_feat = decompose_features(model, train_x[n_train:], device, args.decompose_batch_size)
    train_ds = feature_dataset(train_feat, train_y[:n_train])
    val_ds = feature_dataset(val_feat, train_y[n_train:])

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    patience_left = args.patience
    logs: list[dict[str, float | int | bool]] = []
    text_lines: list[str] = []

    t0 = time.perf_counter()
    for epoch in range(1, args.train_epochs + 1):
        e0 = time.perf_counter()
        model.train()
        train_losses = []
        for batch in train_loader:
            yb = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward_from_batch(model, batch, device)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                yb = batch[-1].to(device)
                pred = forward_from_batch(model, batch, device)
                val_losses.append(float(loss_fn(pred, yb).detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        saved = val_loss + args.min_delta < best_loss
        if saved:
            previous = best_loss
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
            if np.isfinite(previous):
                text_lines.append(
                    f"Validation loss decreased ({previous:.6f} -> {val_loss:.6f}).  Saving model ..."
                )
            else:
                text_lines.append(f"Validation loss initialized ({val_loss:.6f}).  Saving model ...")
        else:
            patience_left -= 1

        epoch_time = time.perf_counter() - e0
        text_lines.append(
            f"Epoch {epoch}/{args.train_epochs} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Time: {epoch_time:.1f}s"
        )
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_sec": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
                "saved": saved,
                "patience_left": patience_left,
            }
        )
        if patience_left <= 0:
            text_lines.append(f"Early stopping at epoch {epoch}.")
            break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    checkpoint_size_kb = None
    if artifact_prefix is not None:
        artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(logs).to_csv(artifact_prefix.with_name(artifact_prefix.name + "_training_log.csv"), index=False)
        artifact_prefix.with_name(artifact_prefix.name + "_epoch_log.txt").write_text(
            "\n".join(text_lines) + "\n",
            encoding="utf-8",
        )
        ckpt_path = artifact_prefix.with_name(artifact_prefix.name + "_checkpoint.pt")
        torch.save(
            {
                "model": "offline_lyra_final",
                "model_state_dict": model.state_dict(),
                "lookback": args.lookback,
                "horizon": horizon,
                "capacity": capacity,
                "params": params,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "run_config": jsonable_args(args),
            },
            ckpt_path,
        )
        checkpoint_size_kb = ckpt_path.stat().st_size / 1024.0

    return model, {
        "train_time_sec": train_time,
        "params": params,
        "checkpoint_size_kb": checkpoint_size_kb,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "notes": f"offline Lyra final-like params; epochs={len(logs)}",
    }


@torch.no_grad()
def predict_offline_lyra(
    model: LyraModel,
    eval_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    features = decompose_features(model, eval_x, device, args.decompose_batch_size)
    ds = feature_dataset(features, np.zeros((len(eval_x), model.pred_len), dtype=np.float32))
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    preds = []
    times = []
    model.eval()
    for batch in loader:
        t0 = time.perf_counter()
        pred = forward_from_batch(model, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) / max(len(batch[-1]), 1))
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32), float(np.mean(times) * 1000.0)


def run_one_horizon(
    series: np.ndarray,
    farm_name: str,
    file_name: str,
    target_col: str,
    capacity: float | None,
    horizon: int,
    args: argparse.Namespace,
    artifact_dir: Path | None,
) -> OfflineLyraSummary:
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
    artifact_prefix = None
    if artifact_dir is not None:
        artifact_prefix = artifact_dir / f"{safe_name(farm_name)}_h{horizon}_offline_lyra_final"

    model, info = train_offline_lyra(train_x, train_y, horizon, capacity, args, artifact_prefix)
    pred, latency_ms = predict_offline_lyra(model, eval_x, args)

    cap = capacity if capacity is not None else float(np.nanmax(y_true))
    pred = np.clip(pred, 0.0, cap).astype(np.float32)
    persistence_pred = persistence_predict(eval_x, horizon)
    daytime_mask = y_true > cap * args.daytime_threshold_ratio

    if artifact_prefix is not None:
        np.savez_compressed(
            artifact_prefix.with_name(artifact_prefix.name + "_predictions.npz"),
            y_true=y_true,
            offline_lyra_final=pred,
            persistence=persistence_pred,
            capacity=np.asarray(cap, dtype=np.float32),
            horizon=np.asarray(horizon, dtype=np.int32),
        )

    n_val = max(1, int(round(len(train_x) * args.val_ratio)))
    n_train = max(1, len(train_x) - n_val)
    summary = OfflineLyraSummary(
        farm=farm_name,
        file=file_name,
        horizon=horizon,
        horizon_minutes=horizon * 15,
        lookback=args.lookback,
        stride=stride,
        train_stride=args.train_stride,
        capacity=capacity,
        target_col=target_col,
        model="offline_lyra_final",
        n_train_windows=n_train,
        n_val_windows=n_val,
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
        train_time_sec=float(info["train_time_sec"]),
        predict_latency_ms=latency_ms,
        params=int(info["params"]),
        checkpoint_size_kb=info["checkpoint_size_kb"],
        best_epoch=int(info["best_epoch"]),
        best_val_loss=float(info["best_val_loss"]),
        notes=str(info["notes"]),
    )
    print(
        f"{farm_name} H={horizon} offline_lyra_final: "
        f"nRMSE={summary.nrmse:.4f}, nMAE={summary.nmae:.4f}, "
        f"day_nRMSE={summary.daytime_nrmse:.4f}, best_epoch={summary.best_epoch}, "
        f"train={summary.train_time_sec:.1f}s",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Lyra PV aligned experiment")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--glob", type=str, default="*.xlsx")
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48, 96])
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--offline_days", type=int, default=91)
    parser.add_argument("--year_days", type=int, default=365)
    parser.add_argument("--clip_target", dest="clip_target", action="store_true", default=True)
    parser.add_argument("--no_clip_target", dest="clip_target", action="store_false")
    parser.add_argument("--daytime_threshold_ratio", type=float, default=0.01)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_eval_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/offline_lyra_pv"))
    parser.add_argument("--save_artifacts", action="store_true")
    parser.add_argument("--artifact_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--decompose_batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--min_delta", type=float, default=1e-8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Reported Offline Lyra architecture configuration.
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--n_power_iters", type=int, default=3)
    parser.add_argument("--ridge_lambda", type=float, default=5.0)
    parser.add_argument("--cond_threshold", type=float, default=1e6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.artifact_dir or (args.output_dir / "artifacts")
    if not args.save_artifacts:
        artifact_dir = None

    year_points = args.year_days * 96
    all_summaries: list[OfflineLyraSummary] = []
    metadata: dict[str, Any] = {
        "protocol": "Offline Lyra trained on Q1 train/validation, evaluated on Q2-Q4 rolling windows.",
        "resolution_minutes": 15,
        "model_role": "Lyra architecture without online adaptation.",
        "args": jsonable_args(args)
        | {
            "file": str(args.file) if args.file else None,
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "output_dir": str(args.output_dir),
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
        },
        "files": [],
    }

    if args.synthetic:
        capacity = args.capacity or 50.0
        files_and_series = [("synthetic_pv", "synthetic", synthetic_pv_series(args.year_days, capacity), capacity, "synthetic_power")]
    else:
        files_and_series = []
        files = collect_files(args)
        if args.max_files is not None:
            files = files[: args.max_files]
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
            summary = run_one_horizon(
                series=series,
                farm_name=farm_name,
                file_name=file_name,
                target_col=target_col,
                capacity=capacity,
                horizon=horizon,
                args=args,
                artifact_dir=artifact_dir,
            )
            all_summaries.append(summary)
            pd.DataFrame([asdict(s) for s in all_summaries]).to_csv(
                args.output_dir / "offline_lyra_summary.csv",
                index=False,
            )

    summary_path = args.output_dir / "offline_lyra_summary.csv"
    metadata_path = args.output_dir / "offline_lyra_metadata.json"
    pd.DataFrame([asdict(s) for s in all_summaries]).to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
