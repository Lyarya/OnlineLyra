# ruff: noqa: E402
"""Run Online Lyra PV on one or more PV farm spreadsheets.

The intended Applied Energy protocol is:
  * one farm = one dataset,
  * use a one-year slice of 15-minute data,
  * first quarter for offline initialization,
  * remaining three quarters for streaming online evaluation.
Targets are clipped to physical PV bounds by default, because the provided
station files contain small negative night-time values and occasional sensor
noise.

Example:
    python scripts/run_online_pv.py \
      --data_dir ./dataset/pv \
      --glob "Solar station site*.xlsx" \
      --target_col power \
      --horizons 4 12 24 48 96
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.OnlineLyraPV import OnlineLyraPV
from utils.energy_metrics import (
    horizon_divergence,
    mae,
    normalized_mae,
    normalized_rmse,
    rmse,
    skill_score,
)


TIME_CANDIDATES = (
    "date",
    "time",
    "datetime",
    "timestamp",
    "data_time",
    "采集时间",
    "时间",
    "日期",
)
POWER_KEYWORDS = (
    "power",
    "pv",
    "generation",
    "active",
    "output",
    "功率",
    "出力",
    "发电",
    "有功",
)


@dataclass
class RunSummary:
    farm: str
    file: str
    horizon: int
    horizon_minutes: int
    lookback: int
    stride: int
    offline_stride: int
    capacity: float | None
    target_col: str
    n_offline_windows: int
    n_online_windows: int
    model: str
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
    predict_latency_ms: float
    update_latency_ms: float
    gate_lyra_fraction: float | None


def infer_capacity_from_name(path: Path) -> float | None:
    match = re.search(r"capacity[-_\s]*([0-9]+(?:\.[0-9]+)?)\s*MW", path.name, re.I)
    if match:
        return float(match.group(1))
    return None


def read_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(axis=1, how="all")


def choose_time_col(df: pd.DataFrame, explicit: str | None) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"time_col={explicit!r} not found in columns.")
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for name in TIME_CANDIDATES:
        if name.lower() in lower:
            return lower[name.lower()]
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return col
    return None


def choose_target_col(df: pd.DataFrame, explicit: str | None, time_col: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"target_col={explicit!r} not found in columns.")
        return explicit

    numeric_cols = [
        c for c in df.columns
        if c != time_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8
    ]
    for col in numeric_cols:
        low = col.lower()
        if any(key.lower() in low for key in POWER_KEYWORDS):
            return col
    if not numeric_cols:
        raise ValueError("Could not infer a numeric PV power target column.")
    return numeric_cols[-1]


def load_power_series(
    path: Path,
    target_col: str | None,
    time_col: str | None,
    sheet: str | int | None,
    year_points: int,
) -> tuple[np.ndarray, str, str | None]:
    df = normalize_columns(read_table(path, sheet=sheet))
    selected_time = choose_time_col(df, time_col)
    selected_target = choose_target_col(df, target_col, selected_time)

    if selected_time is not None:
        df[selected_time] = pd.to_datetime(df[selected_time], errors="coerce")
        df = df.sort_values(selected_time)

    power = pd.to_numeric(df[selected_target], errors="coerce").to_numpy(dtype=np.float64)
    power = power[np.isfinite(power)]
    if len(power) < year_points:
        raise ValueError(
            f"{path.name} has {len(power)} valid target points, "
            f"but at least {year_points} are required."
        )
    power = power[:year_points].astype(np.float32)
    return power, selected_target, selected_time


def clip_target_series(series: np.ndarray, capacity: float | None) -> tuple[np.ndarray, dict[str, float | int]]:
    upper = np.inf if capacity is None else float(capacity)
    clipped = np.clip(series, 0.0, upper).astype(np.float32)
    above_capacity = int(np.sum(series > upper)) if np.isfinite(upper) else 0
    diagnostics = {
        "target_min_before_clip": float(np.nanmin(series)),
        "target_max_before_clip": float(np.nanmax(series)),
        "target_min_after_clip": float(np.nanmin(clipped)),
        "target_max_after_clip": float(np.nanmax(clipped)),
        "negative_target_points": int(np.sum(series < 0.0)),
        "above_capacity_target_points": above_capacity,
    }
    return clipped, diagnostics


def synthetic_pv_series(days: int, capacity: float = 50.0) -> np.ndarray:
    points = days * 96
    rng = np.random.default_rng(2026)
    t = np.arange(points, dtype=np.float64)
    daylight = np.maximum(0.0, np.sin(2.0 * np.pi * (t % 96) / 96.0))
    seasonal = 0.75 + 0.2 * np.sin(2.0 * np.pi * t / (96 * 365))
    cloud = 0.18 * np.sin(2.0 * np.pi * t / 37.0) + 0.12 * rng.standard_normal(points)
    ramps = np.zeros(points)
    ramps[96 * 40:96 * 45] -= 0.35
    pv = capacity * daylight * np.clip(seasonal + cloud + ramps, 0.0, 1.0)
    return pv.astype(np.float32)


def make_windows(series: np.ndarray, lookback: int, horizon: int, start: int, stop: int, stride: int):
    last_start = stop - lookback - horizon
    for idx in range(start, last_start + 1, stride):
        x = series[idx:idx + lookback]
        y = series[idx + lookback:idx + lookback + horizon]
        if len(x) == lookback and len(y) == horizon:
            yield x, y


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("_") or "item"


def masked_metric(metric_fn, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, *args) -> float:
    if not np.any(mask):
        return float("nan")
    return metric_fn(y_true[mask], y_pred[mask], *args)


def run_horizon(
    series: np.ndarray,
    farm_name: str,
    file_name: str,
    target_col: str,
    capacity: float | None,
    horizon: int,
    lookback: int,
    offline_points: int,
    stride: int,
    offline_stride: int,
    offline_epochs: int,
    online_steps: int,
    offline_batch_size: int,
    replay_batch: int,
    hidden_dim: int,
    replay_capacity: int,
    learning_rate: float,
    ridge_lambda: float,
    residual_dropout: float,
    normalization: str,
    loss_mode: str,
    use_instance_mean: bool,
    residual_scale_mode: str,
    device: str,
    max_online_steps: int | None,
    daytime_threshold_ratio: float,
    gate_window: int,
    gate_min_history: int,
    artifact_dir: Path | None = None,
    run_config: dict[str, Any] | None = None,
) -> list[RunSummary]:
    model = OnlineLyraPV(
        lookback=lookback,
        horizon=horizon,
        capacity=capacity,
        hidden_dim=hidden_dim,
        replay_capacity=replay_capacity,
        learning_rate=learning_rate,
        ridge_lambda=ridge_lambda,
        residual_dropout=residual_dropout,
        normalization=normalization,
        loss_mode=loss_mode,
        use_instance_mean=use_instance_mean,
        residual_scale_mode=residual_scale_mode,
        device=device,
    )

    offline_windows = list(
        make_windows(series, lookback, horizon, 0, offline_points, offline_stride)
    )
    loss_records: list[dict[str, Any]] = []
    offline_batch_size = max(int(offline_batch_size), 1)
    for epoch in range(max(offline_epochs, 1)):
        if offline_batch_size == 1:
            order = np.arange(len(offline_windows))
        else:
            order = np.random.permutation(len(offline_windows))
        for offline_idx, batch_start in enumerate(range(0, len(order), offline_batch_size)):
            batch_idx = order[batch_start:batch_start + offline_batch_size]
            x = torch.from_numpy(np.stack([offline_windows[int(i)][0] for i in batch_idx]))
            y = torch.from_numpy(np.stack([offline_windows[int(i)][1] for i in batch_idx]))
            loss = model.online_update(x, y, steps=online_steps, replay_batch=replay_batch)
            loss_records.append({
                "phase": "offline",
                "epoch": epoch,
                "window_index": offline_idx,
                "batch_size": int(len(batch_idx)),
                "horizon": horizon,
                "loss": loss,
                "use_lyra": np.nan,
                "lyra_mse": np.nan,
                "persistence_mse": np.nan,
                "lyra_recent_mse": np.nan,
                "persistence_recent_mse": np.nan,
                "predict_latency_ms": np.nan,
                "update_latency_ms": np.nan,
            })

    preds = []
    truths = []
    trend_preds = []
    persistence_preds = []
    gated_preds = []
    lyra_error_history = deque(maxlen=max(int(gate_window), 1))
    persistence_error_history = deque(maxlen=max(int(gate_window), 1))
    gate_uses_lyra = []
    predict_times = []
    update_times = []

    online_iter = make_windows(
        series,
        lookback,
        horizon,
        offline_points - lookback,
        len(series),
        stride,
    )
    for step_idx, (x_np, y_np) in enumerate(online_iter):
        if max_online_steps is not None and step_idx >= max_online_steps:
            break

        x = torch.from_numpy(x_np).unsqueeze(0)
        y = torch.from_numpy(y_np).unsqueeze(0)

        t0 = time.perf_counter()
        pred, state = model.predict(x)
        predict_times.append(time.perf_counter() - t0)

        pred_np = pred.detach().cpu().numpy()[0]
        persistence_np = np.full(horizon, x_np[-1], dtype=np.float32)
        lyra_recent_mse = float("nan")
        persistence_recent_mse = float("nan")
        if len(lyra_error_history) >= gate_min_history:
            lyra_recent_mse = float(np.mean(lyra_error_history))
            persistence_recent_mse = float(np.mean(persistence_error_history))
            use_lyra = lyra_recent_mse <= persistence_recent_mse
        else:
            use_lyra = True
        gated_np = pred_np if use_lyra else persistence_np
        lyra_mse = float(np.mean(np.square(pred_np - y_np)))
        persistence_mse = float(np.mean(np.square(persistence_np - y_np)))

        preds.append(pred_np)
        truths.append(y_np)
        trend_preds.append(state.trend_forecast.detach().cpu().numpy()[0])
        persistence_preds.append(persistence_np)
        gated_preds.append(gated_np)
        gate_uses_lyra.append(use_lyra)
        lyra_error_history.append(lyra_mse)
        persistence_error_history.append(persistence_mse)

        t0 = time.perf_counter()
        loss = model.online_update_from_state(state, y, steps=online_steps, replay_batch=replay_batch)
        update_elapsed = time.perf_counter() - t0
        update_times.append(update_elapsed)
        loss_records.append({
            "phase": "online",
            "epoch": 0,
            "window_index": step_idx,
            "batch_size": 1,
            "horizon": horizon,
            "loss": loss,
            "use_lyra": bool(use_lyra),
            "lyra_mse": lyra_mse,
            "persistence_mse": persistence_mse,
            "lyra_recent_mse": lyra_recent_mse,
            "persistence_recent_mse": persistence_recent_mse,
            "predict_latency_ms": predict_times[-1] * 1000.0,
            "update_latency_ms": update_elapsed * 1000.0,
        })

    if not preds:
        raise ValueError("No online windows were generated; adjust lookback/horizon/stride.")

    y_true = np.stack(truths)
    model_pred = np.stack(preds)
    trend_pred = np.stack(trend_preds)
    persistence_pred = np.stack(persistence_preds)
    gated_pred = np.stack(gated_preds)
    cap = capacity if capacity is not None else float(np.nanmax(series))
    daytime_threshold = cap * daytime_threshold_ratio
    daytime_mask = y_true > daytime_threshold

    def summarize(
        name: str,
        pred: np.ndarray,
        gate_lyra_fraction: float | None = None,
    ) -> RunSummary:
        return RunSummary(
            farm=farm_name,
            file=file_name,
            horizon=horizon,
            horizon_minutes=horizon * 15,
            lookback=lookback,
            stride=stride,
            offline_stride=offline_stride,
            capacity=capacity,
            target_col=target_col,
            n_offline_windows=len(offline_windows) * max(offline_epochs, 1),
            n_online_windows=len(y_true),
            model=name,
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
            predict_latency_ms=float(np.mean(predict_times) * 1000.0),
            update_latency_ms=float(np.mean(update_times) * 1000.0),
            gate_lyra_fraction=gate_lyra_fraction,
        )

    summaries = [
        summarize("persistence", persistence_pred),
        summarize("trend_only", trend_pred),
        summarize("online_lyra_pv", model_pred),
        summarize("online_lyra_pv_gated", gated_pred, float(np.mean(gate_uses_lyra))),
    ]

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{safe_name(farm_name)}_h{horizon}"
        np.savez_compressed(
            artifact_dir / f"{prefix}_predictions.npz",
            y_true=y_true,
            persistence=persistence_pred,
            trend_only=trend_pred,
            online_lyra_pv=model_pred,
            online_lyra_pv_gated=gated_pred,
            gate_uses_lyra=np.asarray(gate_uses_lyra, dtype=np.bool_),
            capacity=np.asarray(cap, dtype=np.float32),
            horizon=np.asarray(horizon, dtype=np.int32),
        )
        pd.DataFrame(loss_records).to_csv(
            artifact_dir / f"{prefix}_training_log.csv",
            index=False,
        )
        torch.save(
            {
                "farm": farm_name,
                "file": file_name,
                "target_col": target_col,
                "horizon": horizon,
                "lookback": lookback,
                "stride": stride,
                "capacity": capacity,
                "model_state_dict": model.state_dict(),
                "residual_state_dict": model.residual.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "summaries": [asdict(s) for s in summaries],
                "run_config": run_config or {},
            },
            artifact_dir / f"{prefix}_checkpoint.pt",
        )

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online Lyra PV streaming experiment")
    parser.add_argument("--file", type=Path, help="Single PV farm .xlsx/.xls/.csv file")
    parser.add_argument("--data_dir", type=Path, help="Directory containing PV farm files")
    parser.add_argument("--glob", type=str, default="*.xlsx", help="Glob used with --data_dir")
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument(
        "--clip_target",
        dest="clip_target",
        action="store_true",
        default=True,
        help="Clip target power to [0, capacity] before training/evaluation.",
    )
    parser.add_argument(
        "--no_clip_target",
        dest="clip_target",
        action="store_false",
        help="Use raw target power values without physical clipping.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24, 48, 96])
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--offline_stride",
        type=int,
        default=None,
        help=(
            "Stride for Q1 offline initialization windows. Defaults to --stride "
            "(or horizon when --stride is omitted) to preserve earlier runs; use "
            "1 for fair comparison with static baselines trained on dense Q1 windows."
        ),
    )
    parser.add_argument("--offline_days", type=int, default=91)
    parser.add_argument("--year_days", type=int, default=365)
    parser.add_argument("--offline_epochs", type=int, default=1)
    parser.add_argument("--offline_batch_size", type=int, default=1)
    parser.add_argument("--online_steps", type=int, default=2)
    parser.add_argument("--replay_batch", type=int, default=16)
    parser.add_argument("--daytime_threshold_ratio", type=float, default=0.01)
    parser.add_argument("--gate_window", type=int, default=24)
    parser.add_argument("--gate_min_history", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--ridge_lambda", type=float, default=1e-3)
    parser.add_argument("--residual_dropout", type=float, default=0.1)
    parser.add_argument("--normalization", type=str, default="mean", choices=["mean", "revin"])
    parser.add_argument("--loss_mode", type=str, default="forecast", choices=["residual", "forecast"])
    parser.add_argument(
        "--residual_scale_mode",
        type=str,
        default="spectral",
        choices=["spectral", "none"],
        help=(
            "Ablation switch for Hankel spectral residual normalization. "
            "'spectral' divides residuals by the leading Hankel singular value "
            "and restores the scale at prediction time; 'none' feeds raw residuals."
        ),
    )
    parser.add_argument(
        "--use_instance_mean",
        dest="use_instance_mean",
        action="store_true",
        default=True,
        help="Concatenate Lyra's per-window instance mean into the residual MLP input.",
    )
    parser.add_argument(
        "--no_instance_mean",
        dest="use_instance_mean",
        action="store_false",
        help="Use the original c06 residual MLP input: residual_norm only.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_online_steps", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/online_pv"))
    parser.add_argument(
        "--save_artifacts",
        action="store_true",
        help="Save per-farm/horizon training logs, predictions, and checkpoints.",
    )
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        default=None,
        help="Directory for detailed artifacts; defaults to output_dir/artifacts.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.synthetic:
        return []
    if args.file:
        return [args.file]
    if args.data_dir:
        files = sorted(args.data_dir.glob(args.glob))
        if not files:
            raise FileNotFoundError(f"No files matched {args.data_dir / args.glob}")
        return files
    raise ValueError("Provide --file, --data_dir, or --synthetic.")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points_per_day = 96
    year_points = args.year_days * points_per_day
    offline_points = args.offline_days * points_per_day
    all_summaries: list[RunSummary] = []
    metadata = {
        "protocol": "Q1 offline initialization, Q2-Q4 streaming online evaluation",
        "resolution_minutes": 15,
        "args": vars(args) | {
            "file": str(args.file) if args.file else None,
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "output_dir": str(args.output_dir),
            "artifact_dir": str(args.artifact_dir) if args.artifact_dir else None,
        },
    }
    artifact_dir = None
    if args.save_artifacts:
        artifact_dir = args.artifact_dir or (args.output_dir / "artifacts")

    if args.synthetic:
        capacity = args.capacity or 50.0
        series = synthetic_pv_series(args.year_days, capacity=capacity)
        files_and_series = [("synthetic_pv", "synthetic", series, capacity, "synthetic_power")]
    else:
        files_and_series = []
        for path in collect_files(args):
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
            metadata.setdefault("files", []).append(file_metadata)

    for farm_name, file_name, series, capacity, target_col in files_and_series:
        for horizon in args.horizons:
            stride = args.stride or horizon
            offline_stride = args.offline_stride or stride
            summaries = run_horizon(
                series=series,
                farm_name=farm_name,
                file_name=file_name,
                target_col=target_col,
                capacity=capacity,
                horizon=horizon,
                lookback=args.lookback,
                offline_points=offline_points,
                stride=stride,
                offline_stride=offline_stride,
                offline_epochs=args.offline_epochs,
                online_steps=args.online_steps,
                offline_batch_size=args.offline_batch_size,
                replay_batch=args.replay_batch,
                hidden_dim=args.hidden_dim,
                replay_capacity=args.replay_capacity,
                learning_rate=args.learning_rate,
                ridge_lambda=args.ridge_lambda,
                residual_dropout=args.residual_dropout,
                normalization=args.normalization,
                loss_mode=args.loss_mode,
                use_instance_mean=args.use_instance_mean,
                residual_scale_mode=args.residual_scale_mode,
                device=args.device,
                max_online_steps=args.max_online_steps,
                daytime_threshold_ratio=args.daytime_threshold_ratio,
                gate_window=args.gate_window,
                gate_min_history=args.gate_min_history,
                artifact_dir=artifact_dir,
                run_config=metadata["args"],
            )
            all_summaries.extend(summaries)
            best = [s for s in summaries if s.model == "online_lyra_pv"][0]
            gated = [s for s in summaries if s.model == "online_lyra_pv_gated"][0]
            print(
                f"{farm_name} H={horizon} ({horizon * 15}min): "
                f"nRMSE={best.nrmse:.4f}, gated_nRMSE={gated.nrmse:.4f}, "
                f"gated_skill={gated.skill_vs_persistence:.4f}, "
                f"gate_lyra={gated.gate_lyra_fraction:.2f}, "
                f"latency={best.predict_latency_ms:.2f}ms",
                flush=True,
            )

    df = pd.DataFrame([asdict(s) for s in all_summaries])
    summary_path = args.output_dir / "online_pv_summary.csv"
    metadata_path = args.output_dir / "online_pv_metadata.json"
    df.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
