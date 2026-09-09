"""Build PV paper main tables from Online Lyra, baselines, and Offline Lyra.

The script intentionally keeps model collection explicit:

* Online Lyra final uses the gated b06 final row and is displayed as Online Lyra.
* Static baselines are read from pv_baseline_summary.csv files.
* Offline Lyra is optional and read from offline_lyra_summary.csv files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = ["nrmse", "nmae", "daytime_nrmse", "daytime_nmae"]
MODEL_LABELS = {
    "online_lyra_b06_final": "Online Lyra",
    "offline_lyra_final": "Offline Lyra",
    "official_dlinear": "DLinear",
    "official_nlinear": "NLinear",
    "official_linear": "Linear",
    "official_patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timekan": "TimeKAN",
    "phaseformer": "PhaseFormer",
    "mixlinear": "MixLinear",
    "olivia_scratch": "Olivia",
    "seasonal_naive": "Seasonal Naive",
    "persistence": "Persistence",
}
MODEL_ORDER = [
    "online_lyra_b06_final",
    "offline_lyra_final",
    "itransformer",
    "olivia_scratch",
    "official_dlinear",
    "official_nlinear",
    "phaseformer",
    "timekan",
    "official_patchtst",
    "official_linear",
    "mixlinear",
    "seasonal_naive",
    "persistence",
]


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_online(online_root: Path | None) -> pd.DataFrame:
    if online_root is None:
        return pd.DataFrame()
    paths = sorted((online_root / "b06_final").glob("site_*_seed_*/online_pv_summary.csv"))
    rows = read_csvs(paths)
    if rows.empty:
        return rows
    rows = rows[rows["model"].eq("online_lyra_pv_gated")].copy()
    rows["model"] = "online_lyra_b06_final"
    return rows


def collect_static(baseline_root: Path | None) -> pd.DataFrame:
    if baseline_root is None:
        return pd.DataFrame()
    paths = sorted(baseline_root.glob("static_official_seed_*/pv_baseline_summary.csv"))
    if not paths:
        paths = sorted(baseline_root.glob("**/pv_baseline_summary.csv"))
    return read_csvs(paths)


def collect_offline(offline_root: Path | None) -> pd.DataFrame:
    if offline_root is None:
        return pd.DataFrame()
    paths = sorted(offline_root.glob("offline_lyra_seed_*/offline_lyra_summary.csv"))
    if not paths and (offline_root / "offline_lyra_summary.csv").exists():
        paths = [offline_root / "offline_lyra_summary.csv"]
    rows = read_csvs(paths)
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["model"] = "offline_lyra_final"
    return rows


def site_label(farm: str) -> str:
    site_match = re.search(r"site\s+(\d+)", farm, flags=re.IGNORECASE)
    cap_match = re.search(r"capacity-([0-9.]+)MW", farm)
    site = f"Site {site_match.group(1)}" if site_match else farm
    if cap_match:
        cap_value = float(cap_match.group(1))
        cap = str(int(cap_value)) if cap_value.is_integer() else str(cap_value)
        return f"{site} ({cap}MW)"
    return site


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site"] = out["farm"].map(site_label)
    out["site_id"] = (
        out["site"].str.extract(r"Site\s+(\d+)", expand=False).astype(float).astype("Int64")
    )
    out["model_display"] = out["model"].map(MODEL_LABELS).fillna(out["model"])
    return out


def condition_means(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"farm", "horizon", "model", *METRICS}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    grouped = (
        rows.groupby(["farm", "horizon", "model"], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .pipe(add_display_columns)
    )
    return grouped


def make_long_ranks(avg: pd.DataFrame) -> pd.DataFrame:
    long = avg.melt(
        id_vars=["farm", "horizon", "model", "site", "site_id", "model_display"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    long["rank"] = long.groupby(["farm", "horizon", "metric"])["value"].rank(method="min", ascending=True)
    long["rank"] = long["rank"].astype(int)
    long["is_top1"] = long["rank"].eq(1)
    long["is_top3"] = long["rank"].le(3)
    long["is_lyra"] = long["model"].str.contains("lyra", case=False, na=False)
    return long.sort_values(["metric", "site_id", "horizon", "rank", "model_display"])


def model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def ordered_models(df: pd.DataFrame) -> list[str]:
    models = sorted(df["model"].dropna().unique(), key=model_sort_key)
    return models


def format_value(value: float, rank: int, is_online_lyra: bool) -> str:
    text = f"{value:.4f}"
    if rank == 1:
        text = f"**{text}**"
        if is_online_lyra:
            text += "*"
    elif rank <= 3:
        text = f"<u>{text}</u>"
    return text


def markdown_by_site(avg: pd.DataFrame, metric: str) -> str:
    models = ordered_models(avg)
    labels = {model: MODEL_LABELS.get(model, model) for model in models}
    lines = ["| Site | Horizon | " + " | ".join(labels[m] for m in models) + " |"]
    lines.append("|---|---:|" + "|".join(["---:"] * len(models)) + "|")
    for (_site_id, horizon_key), group in avg.groupby(["site_id", "horizon"], sort=True):
        site = group["site"].iloc[0]
        horizon = int(horizon_key)
        ranks = group.set_index("model")[metric].rank(method="min", ascending=True).astype(int)
        vals = group.set_index("model")[metric]
        cells = []
        for model in models:
            if model not in vals.index:
                cells.append("")
                continue
            cells.append(format_value(float(vals[model]), int(ranks[model]), model == "online_lyra_b06_final"))
        lines.append(f"| {site} | {horizon} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown_by_horizon(horizon_avg: pd.DataFrame, metric: str) -> str:
    models = ordered_models(horizon_avg)
    labels = {model: MODEL_LABELS.get(model, model) for model in models}
    lines = ["| Horizon | " + " | ".join(labels[m] for m in models) + " |"]
    lines.append("|---:|" + "|".join(["---:"] * len(models)) + "|")
    for horizon, group in horizon_avg.groupby("horizon", sort=True):
        ranks = group.set_index("model")[metric].rank(method="min", ascending=True).astype(int)
        vals = group.set_index("model")[metric]
        cells = []
        for model in models:
            if model not in vals.index:
                cells.append("")
                continue
            cells.append(format_value(float(vals[model]), int(ranks[model]), model == "online_lyra_b06_final"))
        lines.append(f"| {int(horizon)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_tables(rows: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_counts = rows.groupby("model")["nrmse"].size()
    avg = condition_means(rows)
    avg["model_order"] = avg["model"].map(model_sort_key)
    avg = avg.sort_values(["site_id", "horizon", "model_order", "model_display"]).drop(columns="model_order")
    avg.to_csv(output_dir / "main_condition_average.csv", index=False)

    long = make_long_ranks(avg)
    long.to_csv(output_dir / "main_metric_long_with_ranks.csv", index=False)

    horizon_avg = (
        avg.groupby(["model", "horizon"], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .pipe(add_display_columns_for_models)
    )
    horizon_avg["rank_nrmse"] = horizon_avg.groupby("horizon")["nrmse"].rank(method="min").astype(int)
    horizon_avg = horizon_avg.sort_values(["horizon", "rank_nrmse", "model_display"])
    horizon_avg.to_csv(output_dir / "main_average_by_horizon.csv", index=False)

    overall = (
        avg.groupby("model", as_index=False)[METRICS]
        .mean(numeric_only=True)
        .pipe(add_display_columns_for_models)
    )
    overall["overall_rank"] = overall["nrmse"].rank(method="min").astype(int)
    overall["rows"] = overall["model"].map(raw_counts).astype(int)
    overall = overall.sort_values(["overall_rank", "nmae", "model_display"])
    cols = ["overall_rank", "model", "rows", *METRICS, "model_display"]
    overall[cols].to_csv(output_dir / "main_overall_average.csv", index=False)

    rank_summary = (
        long[long["metric"].eq("nrmse")]
        .groupby("model", as_index=False)
        .agg(
            rows=("rank", "size"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            top1_rate=("is_top1", "mean"),
            top3_rate=("is_top3", "mean"),
            worst_rank=("rank", "max"),
        )
        .pipe(add_display_columns_for_models)
        .sort_values(["mean_rank", "model_display"])
    )
    rank_summary.to_csv(output_dir / "main_nrmse_rank_summary.csv", index=False)

    (output_dir / "main_nrmse_by_site.md").write_text(markdown_by_site(avg, "nrmse"), encoding="utf-8")
    (output_dir / "main_nmae_by_site.md").write_text(markdown_by_site(avg, "nmae"), encoding="utf-8")
    (output_dir / "main_nrmse_horizon_average.md").write_text(
        markdown_by_horizon(horizon_avg, "nrmse"),
        encoding="utf-8",
    )
    (output_dir / "main_nmae_horizon_average.md").write_text(
        markdown_by_horizon(horizon_avg, "nmae"),
        encoding="utf-8",
    )


def add_display_columns_for_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_display"] = out["model"].map(MODEL_LABELS).fillna(out["model"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PV main tables with optional Offline Lyra")
    parser.add_argument("--online_root", type=Path, default=None)
    parser.add_argument("--baseline_root", type=Path, default=None)
    parser.add_argument("--offline_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    frames = [
        collect_online(args.online_root),
        collect_static(args.baseline_root),
        collect_offline(args.offline_root),
    ]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if rows.empty:
        raise FileNotFoundError("No result rows found from the requested roots.")
    build_tables(rows, args.output_dir)
    print(f"Saved PV main tables: {args.output_dir}")


if __name__ == "__main__":
    main()
