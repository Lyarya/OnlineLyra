"""Benchmark PV model inference resources from saved checkpoints.

This is intended for the paper efficiency table. Run one model per process for
cleaner CUDA memory accounting, then concatenate the resulting CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.OnlineLyraPV import OnlineLyraPV  # noqa: E402
from scripts import run_pv_baselines as baselines  # noqa: E402
from scripts.run_online_pv import (  # noqa: E402
    clip_target_series,
    infer_capacity_from_name,
    load_power_series,
    make_windows,
    safe_name,
)


def load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def default_baseline_args(device: str) -> argparse.Namespace:
    old_argv = sys.argv[:]
    sys.argv = ["run_pv_baselines.py"]
    try:
        args = baselines.parse_args()
    finally:
        sys.argv = old_argv
    args.device = device
    return args


def load_eval_x(file: Path, horizon: int, lookback: int, year_days: int, offline_days: int) -> tuple[np.ndarray, float | None]:
    capacity = infer_capacity_from_name(file)
    series, _, _ = load_power_series(
        file,
        target_col="Power (MW)",
        time_col=None,
        sheet=None,
        year_points=year_days * 96,
    )
    series, _ = clip_target_series(series, capacity)
    xs = []
    for x, _ in make_windows(
        series,
        lookback=lookback,
        horizon=horizon,
        start=offline_days * 96 - lookback,
        stop=len(series),
        stride=horizon,
    ):
        xs.append(x)
    if not xs:
        raise RuntimeError("No evaluation windows were generated.")
    return np.stack(xs).astype(np.float32), capacity


def find_baseline_checkpoint(root: Path, seed: int, farm_name: str, horizon: int, model: str) -> Path | None:
    if model in {"persistence", "seasonal_naive"}:
        return None
    pattern = f"{safe_name(farm_name)}_h{horizon}_{model}.pt"
    path = root / f"static_official_seed_{seed}" / "artifacts" / pattern
    if path.exists():
        return path
    matches = sorted((root / f"static_official_seed_{seed}").glob(f"**/*_h{horizon}_{model}.pt"))
    return matches[0] if matches else None


def find_online_checkpoint(root: Path, site: int, seed: int, horizon: int) -> Path:
    matches = sorted((root / "b06_final" / f"site_{site}_seed_{seed}" / "artifacts").glob(f"*_h{horizon}_checkpoint.pt"))
    if not matches:
        raise FileNotFoundError(f"No Online Lyra checkpoint for site={site}, seed={seed}, horizon={horizon}")
    return matches[0]


def nvidia_smi_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


class NvidiaPoller:
    def __init__(self, interval_sec: float = 0.02) -> None:
        self.interval_sec = interval_sec
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.values.append(nvidia_smi_memory_mb())
            time.sleep(self.interval_sec)

    def __enter__(self) -> "NvidiaPoller":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        vals = [v for v in self.values if np.isfinite(v)]
        return max(vals) if vals else float("nan")


def build_online_model(ckpt: dict, device: torch.device) -> OnlineLyraPV:
    cfg = ckpt.get("run_config", {})
    model = OnlineLyraPV(
        lookback=int(ckpt["lookback"]),
        horizon=int(ckpt["horizon"]),
        capacity=ckpt.get("capacity"),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        replay_capacity=int(cfg.get("replay_capacity", 3000)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        ridge_lambda=float(cfg.get("ridge_lambda", 1e-3)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
        normalization=str(cfg.get("normalization", "mean")),
        loss_mode=str(cfg.get("loss_mode", "forecast")),
        use_instance_mean=bool(cfg.get("use_instance_mean", False)),
        residual_scale_mode=str(cfg.get("residual_scale_mode", "spectral")),
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_baseline_model(model_name: str, ckpt: dict | None, lookback: int, horizon: int, device: torch.device):
    if model_name in {"persistence", "seasonal_naive"}:
        return None, 0, None, None
    args = default_baseline_args(str(device))
    model = baselines.build_torch_model(model_name, lookback, horizon, args).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    params = baselines.count_params(model)
    return model, params, float(ckpt["mean"]), float(ckpt["std"])


def benchmark_forward(forward_fn, batch: torch.Tensor, warmup: int, repeats: int, device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    with NvidiaPoller() as poller:
        with torch.no_grad():
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = forward_fn(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1000.0)

    allocated = reserved = float("nan")
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_std_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "torch_peak_allocated_mb": allocated,
        "torch_peak_reserved_mb": reserved,
        "nvidia_smi_peak_mb": poller.peak_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--online_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--site", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--offline_days", type=int, default=91)
    parser.add_argument("--year_days", type=int, default=365)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    eval_x, capacity = load_eval_x(args.file, args.horizon, args.lookback, args.year_days, args.offline_days)
    batch_np = eval_x[: args.batch_size]
    if len(batch_np) < args.batch_size:
        reps = int(np.ceil(args.batch_size / len(batch_np)))
        batch_np = np.tile(batch_np, (reps, 1))[: args.batch_size]

    farm_name = args.file.stem
    ckpt_path: Path | None
    checkpoint_size_kb: float | None

    if args.model == "online_lyra_b06_final":
        ckpt_path = find_online_checkpoint(args.online_root, args.site, args.seed, args.horizon)
        ckpt = load_torch(ckpt_path)
        model = build_online_model(ckpt, device)
        params = baselines.count_params(model)
        checkpoint_size_kb = ckpt_path.stat().st_size / 1024.0
        batch = torch.from_numpy(batch_np).to(device)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            return model.predict(x)[0]

    else:
        ckpt_path = find_baseline_checkpoint(args.baseline_root, args.seed, farm_name, args.horizon, args.model)
        ckpt = load_torch(ckpt_path) if ckpt_path is not None else None
        model, params, mean, std = build_baseline_model(args.model, ckpt, args.lookback, args.horizon, device)
        checkpoint_size_kb = ckpt_path.stat().st_size / 1024.0 if ckpt_path is not None else None
        batch_work = torch.from_numpy(batch_np).to(device)
        if mean is not None and std is not None:
            batch_work = (batch_work - mean) / max(std, 1e-6)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            if args.model == "persistence":
                return x[:, -1:].repeat(1, args.horizon)
            if args.model == "seasonal_naive":
                return x[:, -args.horizon :]
            return model(x)

        batch = batch_work

    before_mb = nvidia_smi_memory_mb()
    result = benchmark_forward(forward_fn, batch, args.warmup, args.repeats, device)
    after_mb = nvidia_smi_memory_mb()

    row = {
        "model": args.model,
        "site": args.site,
        "seed": args.seed,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "device": str(device),
        "params": params,
        "checkpoint_size_kb": checkpoint_size_kb,
        "nvidia_smi_before_mb": before_mb,
        "nvidia_smi_after_mb": after_mb,
        "capacity": capacity,
        "checkpoint_path": str(ckpt_path) if ckpt_path else "",
        **result,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
