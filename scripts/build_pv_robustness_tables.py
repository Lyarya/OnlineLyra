"""Build robustness/generalization tables for the PV experiments.

This script separates two evidence levels:

1. Summary-level robustness for all models: site, capacity group, horizon, and
   daytime metrics from the main condition-average table.
2. Prediction-level proxy robustness for models with saved prediction files:
   clear/cloudy/volatile PV-output proxies and day/night errors.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "online_lyra_b06_final": "Online Lyra",
    "offline_lyra_final": "Offline Lyra",
    "online_lyra_pv_gated": "Online Lyra",
    "offline_lyra": "Offline Lyra",
    "seasonal_naive": "Seasonal Naive",
    "persistence": "Persistence",
    "trend_only": "Lyra trend only",
    "official_linear": "Linear",
    "official_nlinear": "NLinear",
    "official_dlinear": "DLinear",
    "official_patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timekan": "TimeKAN",
    "phaseformer": "PhaseFormer",
    "mixlinear": "MixLinear",
    "olivia_scratch": "Olivia",
}


def capacity_from_farm(farm: str) -> float:
    match = re.search(r"capacity-([0-9.]+)MW", farm)
    if not match:
        return float("nan")
    return float(match.group(1))


def site_id_from_farm(farm: str) -> int:
    match = re.search(r"site\s+(\d+)", farm, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def site_label(farm: str) -> str:
    site_id = site_id_from_farm(farm)
    capacity = capacity_from_farm(farm)
    if np.isfinite(capacity):
        cap_text = str(int(capacity)) if capacity.is_integer() else str(capacity)
        return f"Site {site_id} ({cap_text}MW)"
    return f"Site {site_id}" if site_id >= 0 else farm


def capacity_group(capacity: float) -> str:
    if not np.isfinite(capacity):
        return "unknown"
    if capacity <= 50:
        return "small <=50MW"
    if capacity <= 110:
        return "medium 50-110MW"
    return "large >=130MW"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def norm_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: float) -> dict[str, float]:
    cap = max(float(capacity), 1e-6)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nmae": mae(y_true, y_pred) / cap,
        "nrmse": rmse(y_true, y_pred) / cap,
    }


def summarize_condition_table(condition_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(condition_csv)
    df["capacity"] = df["farm"].map(capacity_from_farm)
    df["capacity_group"] = df["capacity"].map(capacity_group)
    df["site_id"] = df["farm"].map(site_id_from_farm)
    df["site"] = df["farm"].map(site_label)

    by_site = (
        df.groupby(["site_id", "site", "capacity", "model", "model_display"], as_index=False)
        .agg(
            nrmse=("nrmse", "mean"),
            nmae=("nmae", "mean"),
            daytime_nrmse=("daytime_nrmse", "mean"),
            daytime_nmae=("daytime_nmae", "mean"),
        )
        .sort_values(["site_id", "nrmse", "model_display"])
    )
    by_capacity = (
        df.groupby(["capacity_group", "model", "model_display"], as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            nrmse=("nrmse", "mean"),
            nmae=("nmae", "mean"),
            daytime_nrmse=("daytime_nrmse", "mean"),
            daytime_nmae=("daytime_nmae", "mean"),
        )
        .sort_values(["capacity_group", "nrmse", "model_display"])
    )
    by_horizon = (
        df.groupby(["horizon", "model", "model_display"], as_index=False)
        .agg(
            nrmse=("nrmse", "mean"),
            nmae=("nmae", "mean"),
            daytime_nrmse=("daytime_nrmse", "mean"),
            daytime_nmae=("daytime_nmae", "mean"),
        )
        .sort_values(["horizon", "nrmse", "model_display"])
    )
    daytime_gap = (
        df.groupby(["model", "model_display"], as_index=False)
        .agg(
            nrmse=("nrmse", "mean"),
            nmae=("nmae", "mean"),
            daytime_nrmse=("daytime_nrmse", "mean"),
            daytime_nmae=("daytime_nmae", "mean"),
        )
        .assign(
            daytime_minus_all_nrmse=lambda x: x["daytime_nrmse"] - x["nrmse"],
            daytime_minus_all_nmae=lambda x: x["daytime_nmae"] - x["nmae"],
        )
        .sort_values(["nrmse", "model_display"])
    )

    by_site.to_csv(output_dir / "robustness_by_site_all_models.csv", index=False)
    by_capacity.to_csv(output_dir / "robustness_by_capacity_all_models.csv", index=False)
    by_horizon.to_csv(output_dir / "robustness_by_horizon_all_models.csv", index=False)
    daytime_gap.to_csv(output_dir / "robustness_daytime_gap_all_models.csv", index=False)


def online_prediction_files(root: Path) -> list[tuple[str, int, int, int, Path]]:
    files = []
    for path in sorted((root / "b06_final").glob("site_*_seed_*/artifacts/*_predictions.npz")):
        site_match = re.search(r"site_(\d+)_seed_(\d+)", str(path))
        h_match = re.search(r"_h(\d+)_predictions\.npz$", path.name)
        if site_match and h_match:
            files.append(("online_lyra_b06_final", int(site_match.group(1)), int(site_match.group(2)), int(h_match.group(1)), path))
    return files


def offline_prediction_files(root: Path) -> list[tuple[str, int, int, int, Path]]:
    files = []
    for path in sorted(root.glob("offline_lyra_seed_*/artifacts/*_predictions.npz")):
        seed_match = re.search(r"seed_(\d+)", str(path))
        site_match = re.search(r"site_(\d+)_", path.name)
        h_match = re.search(r"_h(\d+)_offline_lyra_final_predictions\.npz$", path.name)
        if seed_match and site_match and h_match:
            files.append(("offline_lyra_final", int(site_match.group(1)), int(seed_match.group(1)), int(h_match.group(1)), path))
    return files


def static_baseline_prediction_files(root: Path) -> list[tuple[str, int, int, int, Path]]:
    files = []
    skip_keys = {"capacity", "horizon", "y_true"}
    for path in sorted(root.glob("static_official_seed_*/artifacts/*_baselines_predictions.npz")):
        seed_match = re.search(r"static_official_seed_(\d+)", str(path))
        site_match = re.search(r"site_(\d+)", path.name)
        h_match = re.search(r"_h(\d+)_baselines_predictions\.npz$", path.name)
        if not (seed_match and site_match and h_match):
            continue
        with np.load(path) as data:
            model_keys = [key for key in data.files if key not in skip_keys]
        for model in model_keys:
            files.append((model, int(site_match.group(1)), int(seed_match.group(1)), int(h_match.group(1)), path))
    return files


def per_window_records(model: str, site: int, seed: int, horizon: int, path: Path) -> list[dict[str, float | int | str]]:
    with np.load(path) as data:
        y_true = data["y_true"].astype(np.float64)
        capacity = float(data["capacity"]) if "capacity" in data.files else max(float(np.nanmax(y_true)), 1.0)
        if model == "online_lyra_b06_final":
            pred_key = "online_lyra_pv_gated"
        elif model == "offline_lyra_final":
            pred_key = "offline_lyra_final"
        else:
            pred_key = model
        if pred_key not in data.files:
            return []
        pred = data[pred_key].astype(np.float64)
        persistence = data["persistence"].astype(np.float64) if "persistence" in data.files else None
    energy = y_true.mean(axis=1) / max(capacity, 1e-6)
    volatility = np.mean(np.abs(np.diff(y_true, axis=1)), axis=1) / max(capacity, 1e-6) if y_true.shape[1] > 1 else np.zeros(len(y_true))
    e_low, e_high = np.quantile(energy, [1 / 3, 2 / 3])
    v_high = np.quantile(volatility, 2 / 3)

    records: list[dict[str, float | int | str]] = []
    categories = {
        "clear_proxy_high_generation": energy >= e_high,
        "cloudy_proxy_low_generation": energy <= e_low,
        "volatile_proxy_high_ramp": volatility >= v_high,
    }
    for name, mask in categories.items():
        if not np.any(mask):
            continue
        rec = {
            "group_type": "pv_weather_proxy",
            "group": name,
            "model": model,
            "model_display": MODEL_LABELS.get(model, model),
            "site_id": site,
            "seed": seed,
            "horizon": horizon,
            "windows": int(np.sum(mask)),
            "capacity": capacity,
            **norm_metrics(y_true[mask], pred[mask], capacity),
        }
        if persistence is not None:
            rec["skill_vs_persistence"] = 1.0 - rmse(y_true[mask], pred[mask]) / max(rmse(y_true[mask], persistence[mask]), 1e-12)
        records.append(rec)

    day_mask = y_true > capacity * 0.01
    night_mask = ~day_mask
    for name, mask in [("daytime", day_mask), ("nighttime_low_power", night_mask)]:
        if not np.any(mask):
            continue
        records.append(
            {
                "group_type": "day_night",
                "group": name,
                "model": model,
                "model_display": MODEL_LABELS.get(model, model),
                "site_id": site,
                "seed": seed,
                "horizon": horizon,
                "windows": int(y_true.shape[0]),
                "points": int(np.sum(mask)),
                "capacity": capacity,
                **norm_metrics(y_true[mask], pred[mask], capacity),
            }
        )
    return records


def summarize_predictions(
    online_root: Path | None,
    offline_root: Path | None,
    baseline_prediction_root: Path | None,
    output_dir: Path,
) -> None:
    files: list[tuple[str, int, int, int, Path]] = []
    if online_root:
        files.extend(online_prediction_files(online_root))
    if offline_root:
        files.extend(offline_prediction_files(offline_root))
    if baseline_prediction_root:
        files.extend(static_baseline_prediction_files(baseline_prediction_root))
    rows = []
    for model, site, seed, horizon, path in files:
        rows.extend(per_window_records(model, site, seed, horizon, path))
    if not rows:
        return
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "robustness_prediction_group_metrics_long.csv", index=False)
    agg = (
        raw.groupby(["group_type", "group", "model", "model_display", "horizon"], as_index=False)
        .agg(
            rows=("nrmse", "size"),
            nrmse=("nrmse", "mean"),
            nmae=("nmae", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            skill_vs_persistence=("skill_vs_persistence", "mean"),
        )
        .sort_values(["group_type", "group", "horizon", "nrmse", "model_display"])
    )
    agg.to_csv(output_dir / "robustness_prediction_group_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PV robustness/generalization tables")
    parser.add_argument(
        "--condition_csv",
        type=Path,
        default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425/paper_tables_with_offline_lyra/main_condition_average.csv"),
    )
    parser.add_argument(
        "--online_root",
        type=Path,
        default=Path("outputs/server_runs_20260716/final_online_lyra_b06_20260716_231353"),
    )
    parser.add_argument(
        "--offline_root",
        type=Path,
        default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425"),
    )
    parser.add_argument(
        "--baseline_prediction_root",
        type=Path,
        default=Path("outputs/server_runs_20260716/fair_protocol_20260716_123158"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_result_pack/robustness"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize_condition_table(args.condition_csv, args.output_dir)
    summarize_predictions(args.online_root, args.offline_root, args.baseline_prediction_root, args.output_dir)
    print(f"Saved robustness tables to {args.output_dir}")


if __name__ == "__main__":
    main()
