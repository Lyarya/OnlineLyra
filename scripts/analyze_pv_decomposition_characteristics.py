#!/usr/bin/env python3
# ruff: noqa: E402
"""Characterize, without an accuracy-direction assumption, Online Lyra's decomposition."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_online_pv import (
    clip_target_series,
    infer_capacity_from_name,
    load_power_series,
)


LOOKBACK = 96
HANKEL_L = LOOKBACK // 2
OFFLINE_POINTS = 91 * 96
LAGS = (1, 4, 12, 24, 48)
ORIGINAL = "#7A8793"
RESIDUAL = "#0B4F8A"
ACCENT = "#D18A3A"
PANEL_SIZE_INCHES = (3.45, 2.35)
PANEL_MARGINS = {
    "left": 0.20,
    "right": 0.975,
    "bottom": 0.22,
    "top": 0.965,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def hankelize(x: np.ndarray, length: int = HANKEL_L) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, length).T


def diagonal_average(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    anti_diag = (
        np.arange(rows, dtype=np.int64)[:, None]
        + np.arange(cols, dtype=np.int64)[None, :]
    )
    values = np.bincount(anti_diag.ravel(), weights=matrix.ravel())
    counts = np.bincount(anti_diag.ravel())
    return values / counts


def decompose_rank1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean()
    trajectory = hankelize(centered)
    u, singular, vh = np.linalg.svd(trajectory, full_matrices=False)
    trend_matrix = singular[0] * np.outer(u[:, 0], vh[0])
    trend_centered = diagonal_average(trend_matrix)
    residual = centered - trend_centered
    return trend_centered + x.mean(), residual


def effective_rank(x: np.ndarray) -> float:
    singular = np.linalg.svd(hankelize(x), compute_uv=False)
    energy = np.square(singular)
    total = energy.sum()
    if total <= np.finfo(float).eps:
        return np.nan
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def spectral_entropy(x: np.ndarray) -> float:
    centered = x - x.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    power = power[1:]
    total = power.sum()
    if total <= np.finfo(float).eps:
        return np.nan
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(power)))


def autocorrelation(x: np.ndarray, lag: int) -> float:
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(float).eps or lag >= len(centered):
        return np.nan
    return float(np.dot(centered[:-lag], centered[lag:]) / denominator)


def site_id(path: Path) -> int:
    match = re.search(r"site\s+(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer site id from {path.name}")
    return int(match.group(1))


def analyze_window(
    x: np.ndarray,
    site: int,
    start: int,
    capacity: float,
) -> dict[str, float | int]:
    trend, residual = decompose_rank1(x)
    centered = x - x.mean()
    original_energy = float(np.dot(centered, centered))
    residual_energy = float(np.dot(residual, residual))
    result: dict[str, float | int] = {
        "site_id": site,
        "window_start": start,
        "capacity_mw": capacity,
        "observed_mean_mw": float(x.mean()),
        "observed_max_mw": float(x.max()),
        "energy_ratio_residual_to_centered": residual_energy / original_energy,
        "effective_rank_original": effective_rank(centered),
        "effective_rank_residual": effective_rank(residual),
        "spectral_entropy_original": spectral_entropy(centered),
        "spectral_entropy_residual": spectral_entropy(residual),
        "trend_total_variation": float(np.abs(np.diff(trend)).sum()),
        "observed_total_variation": float(np.abs(np.diff(x)).sum()),
    }
    result["trend_to_observed_total_variation"] = (
        result["trend_total_variation"] / result["observed_total_variation"]
        if result["observed_total_variation"] > 0
        else np.nan
    )
    for lag in LAGS:
        result[f"acf_original_lag{lag}"] = autocorrelation(centered, lag)
        result[f"acf_residual_lag{lag}"] = autocorrelation(residual, lag)
    result["integrated_abs_acf_original_1_24"] = float(
        np.nansum([abs(autocorrelation(centered, lag)) for lag in range(1, 25)])
    )
    result["integrated_abs_acf_residual_1_24"] = float(
        np.nansum([abs(autocorrelation(residual, lag)) for lag in range(1, 25)])
    )
    return result


def sample_site(
    path: Path,
    max_windows: int,
    seed: int,
) -> tuple[list[dict[str, float | int]], dict[str, int | float | str]]:
    capacity = infer_capacity_from_name(path)
    if capacity is None:
        raise ValueError(f"Cannot infer capacity from {path.name}")
    series, target_col, time_col = load_power_series(
        path,
        target_col="Power (MW)",
        time_col=None,
        sheet=None,
        year_points=365 * 96,
    )
    series, clipping = clip_target_series(series, capacity)
    first = max(0, OFFLINE_POINTS - LOOKBACK)
    candidates = np.arange(first, len(series) - LOOKBACK + 1, dtype=np.int64)
    rng = np.random.default_rng(seed + site_id(path))
    rng.shuffle(candidates)

    records: list[dict[str, float | int]] = []
    screened = 0
    for start in candidates:
        screened += 1
        x = np.asarray(series[start : start + LOOKBACK], dtype=np.float64)
        centered_energy = float(np.dot(x - x.mean(), x - x.mean()))
        if x.max() <= 0.01 * capacity or centered_energy <= 1e-10:
            continue
        records.append(analyze_window(x, site_id(path), int(start), float(capacity)))
        if len(records) >= max_windows:
            break
    audit = {
        "site_id": site_id(path),
        "file": path.name,
        "capacity_mw": float(capacity),
        "target_col": target_col,
        "time_col": time_col or "",
        "available_candidates": int(len(candidates)),
        "screened_candidates": screened,
        "retained_windows": len(records),
        "exclusion_rule": "max(window) <= 1% capacity or centered energy <= 1e-10",
        "clipped_below_zero": int(clipping["negative_target_points"]),
        "clipped_above_capacity": int(clipping["above_capacity_target_points"]),
    }
    return records, audit


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = [column for column in metrics.columns if column not in {
        "site_id", "window_start", "capacity_mw"
    }]
    site = (
        metrics.groupby("site_id")[value_cols]
        .agg(["median", "mean", "std", "count"])
    )
    site.columns = ["_".join(column) for column in site.columns]
    site = site.reset_index()

    rows = []
    for column in value_cols:
        values = metrics[column].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": column,
                "n_windows": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    return site, pd.DataFrame(rows)


def save_figure(fig: plt.Figure, base: Path) -> None:
    # Keep a fixed physical canvas and axes rectangle for every panel.
    # Tight bounding boxes would recrop each panel according to its labels and
    # therefore make nominally identical figures misalign in the manuscript.
    for ax in fig.axes:
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#000000")
            ax.spines[side].set_linewidth(0.55)
        ax.tick_params(
            direction="in", width=0.45, length=3.0,
            color="#000000", labelcolor="#000000",
        )
        ax.xaxis.label.set_color("#000000")
        ax.yaxis.label.set_color("#000000")
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_color("#000000")
        for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            if gridline.get_visible():
                gridline.set_color("#C9CED4")
                gridline.set_linewidth(0.36)
                gridline.set_alpha(0.28)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def new_panel() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=PANEL_SIZE_INCHES)
    fig.subplots_adjust(**PANEL_MARGINS)
    return fig, ax


def panel_energy(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = new_panel()
    groups = [group["energy_ratio_residual_to_centered"].to_numpy() for _, group in metrics.groupby("site_id")]
    sites = sorted(metrics["site_id"].unique())
    box = ax.boxplot(groups, positions=sites, widths=0.55, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set(facecolor=RESIDUAL, alpha=0.23, edgecolor=RESIDUAL, linewidth=0.8)
    for median in box["medians"]:
        median.set(color=RESIDUAL, linewidth=1.25)
    ax.axhline(1.0, color="#B7BEC5", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set(xlabel="PV site", ylabel=r"Residual energy ratio $\eta_r$", xticks=sites)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.45, alpha=0.7)
    save_figure(fig, output)


def paired_site_panel(
    metrics: pd.DataFrame,
    original_col: str,
    residual_col: str,
    ylabel: str,
    output: Path,
) -> None:
    paired = metrics.groupby("site_id")[[original_col, residual_col]].median()
    fig, ax = new_panel()
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row[original_col], row[residual_col]], color="#C4CBD1", linewidth=0.75, zorder=1)
    ax.scatter(np.zeros(len(paired)), paired[original_col], color=ORIGINAL, s=20, zorder=2, label="Centered input")
    ax.scatter(np.ones(len(paired)), paired[residual_col], color=RESIDUAL, s=20, zorder=2, label="Residual")
    ax.set(xlim=(-0.35, 1.35), xticks=[0, 1], xticklabels=["Input", "Residual"], ylabel=ylabel)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.45, alpha=0.7)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.88, edgecolor="#D0D4D8")
    save_figure(fig, output)


def panel_acf(metrics: pd.DataFrame, output: Path) -> None:
    lags = np.arange(1, 25)
    original = np.asarray([
        [autocorrelation_from_row(row, lag, "original") for lag in lags]
        for _, row in metrics.iterrows()
    ])
    residual = np.asarray([
        [autocorrelation_from_row(row, lag, "residual") for lag in lags]
        for _, row in metrics.iterrows()
    ])
    fig, ax = new_panel()
    for values, color, label in (
        (original, ORIGINAL, "Centered input"),
        (residual, RESIDUAL, "Residual"),
    ):
        median = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        ax.fill_between(lags, q25, q75, color=color, alpha=0.16, linewidth=0)
        ax.plot(lags, median, color=color, label=label)
    ax.axhline(0, color="#AEB5BC", linewidth=0.6)
    ax.set(xlabel="Lag (15-min steps)", ylabel="Autocorrelation", xlim=(1, 24))
    ax.grid(color="#D9DEE3", linewidth=0.45, alpha=0.7)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.88, edgecolor="#D0D4D8")
    save_figure(fig, output)


def autocorrelation_from_row(row: pd.Series, lag: int, family: str) -> float:
    # The compact window metrics store selected lags only; the full ACF is
    # reconstructed from source arrays retained transiently in main().
    return autocorrelation(row[f"_{family}_sequence"], lag)


def composite(paths: list[Path], output: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    del PdfPages  # PDF panels remain the submission source; this is a PNG preview.
    images = [plt.imread(path.with_suffix(".png")) for path in paths]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.95), constrained_layout=True)
    for label, ax, image in zip(("a", "b", "c", "d"), axes.flat, images):
        ax.imshow(image)
        ax.axis("off")
        ax.text(0.01, 0.99, label, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=ROOT / "solar_stations")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "paper_result_pack" / "component_characterization")
    parser.add_argument("--max_windows_per_site", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    records: list[dict[str, float | int]] = []
    audits: list[dict[str, int | float | str]] = []
    source_sequences: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(args.data_dir.glob("Solar station site *.xlsx"), key=site_id):
        site_records, audit = sample_site(path, args.max_windows_per_site, args.seed)
        records.extend(site_records)
        audits.append(audit)

        # Recreate only the retained windows for the ACF profile source.
        series, _, _ = load_power_series(
            path,
            target_col="Power (MW)",
            time_col=None,
            sheet=None,
            year_points=365 * 96,
        )
        capacity = infer_capacity_from_name(path)
        series, _ = clip_target_series(series, float(capacity))
        starts = [int(item["window_start"]) for item in site_records]
        for start in starts:
            x = np.asarray(series[start : start + LOOKBACK], dtype=np.float64)
            _, residual = decompose_rank1(x)
            source_sequences.append((x - x.mean(), residual))

    metrics = pd.DataFrame(records)
    metrics.to_csv(args.output_dir / "decomposition_window_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "decomposition_data_audit.csv", index=False)
    site, overall = summarize(metrics)
    site.to_csv(args.output_dir / "decomposition_site_summary.csv", index=False)
    overall.to_csv(args.output_dir / "decomposition_overall_summary.csv", index=False)

    # Object columns are used only for drawing the full ACF profile and are never exported.
    metrics_for_acf = metrics.copy()
    metrics_for_acf["_original_sequence"] = [item[0] for item in source_sequences]
    metrics_for_acf["_residual_sequence"] = [item[1] for item in source_sequences]

    panels = [
        args.output_dir / "fig_component_a_energy",
        args.output_dir / "fig_component_b_effective_rank",
        args.output_dir / "fig_component_c_spectral_entropy",
        args.output_dir / "fig_component_d_acf",
    ]
    panel_energy(metrics, panels[0])
    paired_site_panel(
        metrics,
        "effective_rank_original",
        "effective_rank_residual",
        "Effective Hankel rank",
        panels[1],
    )
    paired_site_panel(
        metrics,
        "spectral_entropy_original",
        "spectral_entropy_residual",
        "Normalized spectral entropy",
        panels[2],
    )
    panel_acf(metrics_for_acf, panels[3])
    composite(panels, args.output_dir / "fig_component_characterization_preview.png")

    selected = overall.set_index("metric")
    report = {
        "n_sites": int(metrics["site_id"].nunique()),
        "n_windows": int(len(metrics)),
        "median_residual_energy_ratio": float(selected.loc["energy_ratio_residual_to_centered", "median"]),
        "median_effective_rank_original": float(selected.loc["effective_rank_original", "median"]),
        "median_effective_rank_residual": float(selected.loc["effective_rank_residual", "median"]),
        "median_spectral_entropy_original": float(selected.loc["spectral_entropy_original", "median"]),
        "median_spectral_entropy_residual": float(selected.loc["spectral_entropy_residual", "median"]),
        "median_integrated_abs_acf_original": float(selected.loc["integrated_abs_acf_original_1_24", "median"]),
        "median_integrated_abs_acf_residual": float(selected.loc["integrated_abs_acf_residual_1_24", "median"]),
        "interpretation_policy": "Descriptive characterization only; no assumption that residuals are easier to forecast.",
    }
    (args.output_dir / "decomposition_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
