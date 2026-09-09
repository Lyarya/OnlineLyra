#!/usr/bin/env python3
"""Second-pass refinements for five Online Lyra manuscript figures.

The original ``figures_revised`` package is preserved. This script reuses the
original data-generation and audit functions, redirects them to a new output
root, and replaces only the rendering layer for Fig. 3, Fig. 5, Fig. 6,
Fig. S4, and Fig. S5.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import make_pv_paper_figures_revised as base  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec, patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


OUT = base.PACK / "figures_revised_v2"
DATA_OUT = OUT / "data"
OLD_OUT = base.PACK / "figures_revised"

# Redirect the original functions without changing their implementation.
base.OUT = OUT
base.DATA_OUT = DATA_OUT

COLORS = base.COLORS
GATE_ONLINE = "#154E80"
MODELS = ["Online Lyra", "iTransformer", "Smart Persistence"]
MODEL_STYLE = {
    "Online Lyra": {"color": COLORS["online"], "lw": 1.75, "ls": "-"},
    "iTransformer": {"color": COLORS["online_light"], "lw": 1.25, "ls": (0, (3.2, 1.6))},
    "Smart Persistence": {"color": "#8F8997", "lw": 1.20, "ls": (0, (5.0, 2.0))},
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / "svg" / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / "pdf" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "png" / f"{stem}.png", bbox_inches="tight", dpi=600)
    fig.savefig(OUT / "tiff" / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def fmt_latency(value: float) -> str:
    if value < 0.1:
        return f"{value:.2f} ms"
    return f"{value:.1f} ms"


def render_fig03_story() -> None:
    case = pd.read_csv(DATA_OUT / "fig03_case_trace_site2_h4_seed2028.csv")
    metadata = pd.read_csv(DATA_OUT / "fig03_event_metadata.csv").iloc[0]
    x = case["case_time_h"].to_numpy(dtype=float)
    y = case["y_true"].to_numpy(dtype=float)
    online = case["online_lyra"].to_numpy(dtype=float)
    persistence = case["persistence"].to_numpy(dtype=float)
    gate = case["gate_uses_lyra"].astype(bool).to_numpy()
    event_mask = case["event_selected"].astype(bool).to_numpy()
    event_idx = np.flatnonzero(event_mask)
    event_start = float(metadata["event_start_h"])
    event_end = float(metadata["event_end_h"])

    ramp = np.r_[0.0, np.abs(np.diff(y))]
    online_error = np.abs(y - online)
    persistence_error = np.abs(y - persistence)
    gain = persistence_error - online_error
    ramp_idx = int(event_idx[np.argmax(ramp[event_idx])])
    lag_candidates = event_idx[event_idx >= ramp_idx]
    lag_idx = int(lag_candidates[np.argmax(gain[lag_candidates])])
    observed_range = float(np.ptp(y[event_idx]))
    recovery_candidates = event_idx[
        (event_idx > lag_idx)
        & (online_error[event_idx] <= max(0.05 * observed_range, 1.0))
        & (online_error[event_idx] <= 0.35 * np.maximum(persistence_error[event_idx], 1e-9))
    ]
    if len(recovery_candidates):
        recovery_idx = int(recovery_candidates[0])
    else:
        after_lag = event_idx[event_idx > lag_idx]
        recovery_idx = int(after_lag[np.argmin(online_error[after_lag])])

    pd.DataFrame(
        [
            {
                "stage": "rapid ramp",
                "case_time_h": x[ramp_idx],
                "observed_mw": y[ramp_idx],
                "online_mw": online[ramp_idx],
                "persistence_mw": persistence[ramp_idx],
                "selection_rule": "largest absolute 15-min observed change within the audited event",
            },
            {
                "stage": "persistence lag",
                "case_time_h": x[lag_idx],
                "observed_mw": y[lag_idx],
                "online_mw": online[lag_idx],
                "persistence_mw": persistence[lag_idx],
                "selection_rule": "largest Online Lyra absolute-error gain over Persistence after ramp onset",
            },
            {
                "stage": "online recovery",
                "case_time_h": x[recovery_idx],
                "observed_mw": y[recovery_idx],
                "online_mw": online[recovery_idx],
                "persistence_mw": persistence[recovery_idx],
                "selection_rule": "first post-lag point with Online error <=5% event range and <=35% Persistence error",
            },
        ]
    ).to_csv(DATA_OUT / "fig03_story_stage_audit.csv", index=False)

    highlight_fc = "#E8D68A"
    boundary = "#B7AA70"
    guide = "#A5ADB7"
    gate_cmap = ListedColormap([COLORS["fallback"], GATE_ONLINE])

    fig = plt.figure(figsize=(7.25, 5.12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.45, 0.50, 0.23, 1.62], hspace=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(event_start, event_end, color=highlight_fc, alpha=0.08, lw=0, zorder=0)
    ax.axvline(event_start, color=boundary, lw=0.50, ls=(0, (2, 2)), alpha=0.55)
    ax.axvline(event_end, color=boundary, lw=0.50, ls=(0, (2, 2)), alpha=0.55)
    ax.plot(x, persistence, color=COLORS["persistence"], lw=1.0, ls="--", alpha=0.82, zorder=2)
    ax.plot(x, online, color=COLORS["online"], lw=1.65, alpha=0.96, zorder=3)
    observed_line, = ax.plot(x, y, color=COLORS["observed"], lw=1.02, alpha=0.98, zorder=4)
    observed_line.set_path_effects(
        [patheffects.withStroke(linewidth=1.65, foreground="white", alpha=0.40)]
    )
    ax.text(
        (event_start + event_end) / 2,
        1.012,
        "Critical ramp",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#6D674F",
        clip_on=False,
    )
    ax.set_xlim(0, 120)
    ax.set_ylabel("PV power (MW)")
    ax.tick_params(axis="x", labelbottom=False)
    base.quiet_grid(ax)
    base.panel_label_fixed(fig, ax, "a")

    axg = fig.add_subplot(gs[1, 0], sharex=ax)
    axg.imshow(
        gate[np.newaxis, :].astype(int),
        aspect="auto",
        interpolation="nearest",
        cmap=gate_cmap,
        vmin=0,
        vmax=1,
        extent=[0, 120, 0, 1],
        zorder=1,
    )
    axg.axvspan(event_start, event_end, color="white", alpha=0.14, lw=0, zorder=2)
    axg.axvline(event_start, color=boundary, lw=0.55, alpha=0.65, zorder=3)
    axg.axvline(event_end, color=boundary, lw=0.55, alpha=0.65, zorder=3)
    axg.annotate(
        "Lyra path active",
        xy=((event_start + event_end) / 2, 0.50),
        xytext=(event_start - 5.0, 0.50),
        ha="right",
        va="center",
        fontsize=6.2,
        color=COLORS["online"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.0),
        arrowprops=dict(arrowstyle="-", lw=0.55, color=guide, shrinkA=2, shrinkB=2),
        zorder=4,
    )
    axg.set_xlim(0, 120)
    axg.set_ylim(0, 1)
    axg.set_xticks(np.arange(0, 121, 20))
    axg.set_xlabel("")
    axg.set_ylabel("Gate", rotation=0, labelpad=20, va="center")
    axg.set_yticks([])
    for spine in axg.spines.values():
        spine.set_visible(False)
    base.panel_label_fixed(fig, axg, "b", dy_pt=5.0)

    axs = fig.add_subplot(gs[2, 0])
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    axs.axis("off")
    story = [
        (0.17, "1  Rapid ramp", "#4F5963"),
        (0.50, "2  Persistence lag", "#766E7C"),
        (0.83, "3  Online recovery", COLORS["online"]),
    ]
    for xpos, label, color in story:
        axs.text(
            xpos,
            0.48,
            label,
            ha="center",
            va="center",
            fontsize=6.7,
            color=color,
            fontweight="bold" if label.startswith("3") else "normal",
        )
    for left, right in [(0.29, 0.38), (0.62, 0.71)]:
        axs.annotate(
            "",
            xy=(right, 0.48),
            xytext=(left, 0.48),
            xycoords=axs.transAxes,
            arrowprops=dict(arrowstyle="->", color="#AAB1B9", lw=0.55),
        )

    axz = fig.add_subplot(gs[3, 0])
    event = case.loc[event_mask].copy()
    xe = event["case_time_h"].to_numpy(dtype=float)
    axz.axvspan(event_start, event_end, color=highlight_fc, alpha=0.035, lw=0, zorder=0)
    axz.plot(xe, event["persistence"], color=COLORS["persistence"], lw=1.25, ls="--", alpha=0.88, zorder=2)
    axz.plot(xe, event["online_lyra"], color=COLORS["online"], lw=2.0, alpha=0.97, zorder=3)
    zoom_obs, = axz.plot(xe, event["y_true"], color=COLORS["observed"], lw=1.25, alpha=0.98, zorder=4)
    zoom_obs.set_path_effects(
        [patheffects.withStroke(linewidth=1.95, foreground="white", alpha=0.40)]
    )
    zoom_values = event[["y_true", "online_lyra", "persistence"]].to_numpy(dtype=float)
    zlo = float(np.nanmin(zoom_values))
    zhi = float(np.nanmax(zoom_values))
    zpad = max((zhi - zlo) * 0.11, 1.0)
    axz.set_xlim(event_start, event_end)
    axz.set_ylim(max(0.0, zlo - zpad), zhi + zpad)
    axz.set_xlabel("Event time (h)")
    axz.set_ylabel("PV power (MW)")
    base.quiet_grid(axz)
    base.panel_label_fixed(fig, axz, "c")

    stage_specs = [
        ("1", ramp_idx, y[ramp_idx], "#4F5963"),
        ("2", lag_idx, persistence[lag_idx], "#766E7C"),
        ("3", recovery_idx, online[recovery_idx], COLORS["online"]),
    ]
    story_top = axz.get_ylim()[1] - 0.055 * (axz.get_ylim()[1] - axz.get_ylim()[0])
    for number, idx, target_y, color in stage_specs:
        axz.annotate(
            number,
            xy=(x[idx], target_y),
            xycoords="data",
            xytext=(x[idx], story_top),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=6.0,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.16", facecolor="white", edgecolor=color, lw=0.65),
            arrowprops=dict(arrowstyle="-", color=guide, lw=0.60, shrinkA=5, shrinkB=2),
            zorder=6,
        )

    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=1.25, label="Observed PV"),
        Line2D([0], [0], color=COLORS["online"], lw=2.0, label="Online Lyra"),
        Line2D([0], [0], color=COLORS["persistence"], lw=1.25, ls="--", label="Persistence"),
        Patch(facecolor=GATE_ONLINE, label="Lyra selected"),
        Patch(facecolor=COLORS["fallback"], label="Fallback selected"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.992),
        ncol=5,
        columnspacing=1.05,
        handlelength=1.95,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.89, hspace=0.14)
    save_pub(fig, "fig03_case_trace_gate")


def smooth_density(samples: np.ndarray, lo: float, hi: float, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, 521)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0])
    hist, _ = np.histogram(samples, bins=edges, density=True)
    radius = max(1, int(np.ceil(4.0 * bandwidth / width)))
    offsets = np.arange(-radius, radius + 1, dtype=float) * width
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(hist, kernel, mode="same")
    density /= np.trapezoid(density, centers)
    return centers, density


def render_fig05_error_and_agreement() -> None:
    site_metrics = pd.read_csv(DATA_OUT / "fig05_daytime_site_metrics.csv")
    metric_summary = pd.read_csv(DATA_OUT / "fig05_daytime_metric_summary.csv")
    ramp_summary = pd.read_csv(DATA_OUT / "fig05_ramp_quartile_summary.csv")

    styles = {
        "Online Lyra": {
            "color": COLORS["online"], "marker": "o", "ls": "-", "lw": 2.15, "zorder": 5,
        },
        "iTransformer": {
            "color": COLORS["online_light"], "marker": "s", "ls": (0, (3, 1.5)), "lw": 1.48, "zorder": 4,
        },
        "Smart Persistence": {
            "color": "#8F8997", "marker": "^", "ls": (0, (5, 2)), "lw": 1.38, "zorder": 3,
        },
    }
    metric_order = ["nMAE", "nRMSE", "P90 |error|"]
    ramp_labels = ["Q1", "Q2", "Q3", "Q4"]
    metric_spacing = 1.13
    summary_marker_size = 4.85
    raw_marker_area = 12.5
    error_lw = 1.08
    error_cap = 2.25

    fig = plt.figure(figsize=(7.25, 3.25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.14], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    x_metric = np.arange(len(metric_order), dtype=float) * metric_spacing
    offsets = {"Online Lyra": -0.18, "iTransformer": 0.0, "Smart Persistence": 0.18}
    jitter = np.linspace(-0.038, 0.038, 8)
    legend_handles = []
    for model in MODELS:
        style = styles[model]
        for metric_idx, metric in enumerate(metric_order):
            raw = (
                site_metrics.loc[
                    (site_metrics["model_display"] == model) & (site_metrics["metric"] == metric)
                ]
                .sort_values("site_id")["value"]
                .to_numpy(dtype=float)
            )
            summary = metric_summary.loc[
                (metric_summary["model_display"] == model) & (metric_summary["metric"] == metric)
            ].iloc[0]
            xpos = x_metric[metric_idx] + offsets[model]
            ax0.scatter(
                xpos + jitter[: len(raw)],
                raw,
                s=raw_marker_area,
                marker=style["marker"],
                color=style["color"],
                alpha=0.24,
                edgecolor="none",
                zorder=int(style["zorder"]) - 1,
            )
            ax0.errorbar(
                xpos,
                summary["mean"],
                yerr=summary["ci95"],
                fmt=style["marker"],
                ms=summary_marker_size,
                mfc=style["color"] if model == "Online Lyra" else "white",
                mec=style["color"],
                mew=0.90,
                ecolor=style["color"],
                elinewidth=error_lw,
                capsize=error_cap,
                capthick=1.02,
                zorder=int(style["zorder"]),
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                ls=style["ls"],
                lw=style["lw"],
                marker=style["marker"],
                ms=summary_marker_size,
                markerfacecolor=style["color"] if model == "Online Lyra" else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.90,
                label=model,
            )
        )
    ax0.set_xticks(x_metric)
    ax0.set_xticklabels(["nMAE", "nRMSE", "P90 $|e|$"])
    ax0.set_xlim(x_metric[0] - 0.48, x_metric[-1] + 0.48)
    metric_low = float(np.min(metric_summary["mean"] - metric_summary["ci95"]))
    metric_high = float(np.max(metric_summary["mean"] + metric_summary["ci95"]))
    ax0.set_ylim(max(0, metric_low - 0.025), metric_high + 0.025)
    ax0.set_ylabel("Normalized error metric")
    base.quiet_grid(ax0)
    base.panel_label_fixed(fig, ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    x_ramp = np.arange(1, 5, dtype=float)
    ax1.axvspan(3.5, 4.5, color="#E8D68A", alpha=0.08, lw=0, zorder=0)
    for model in MODELS:
        style = styles[model]
        sub = (
            ramp_summary.loc[ramp_summary["model_display"] == model]
            .set_index("ramp_quartile")
            .reindex(ramp_labels)
        )
        ax1.errorbar(
            x_ramp,
            sub["mean"].to_numpy(dtype=float),
            yerr=sub["ci95"].to_numpy(dtype=float),
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
            marker=style["marker"],
            ms=summary_marker_size,
            mfc=style["color"] if model == "Online Lyra" else "white",
            mec=style["color"],
            mew=0.90,
            ecolor=style["color"],
            elinewidth=error_lw,
            capsize=error_cap,
            capthick=1.02,
            label=model,
            zorder=style["zorder"],
        )
    prediction_df = pd.read_csv(DATA_OUT / "fig05_daytime_predictions_h96_seed2028.csv")
    ramp_cuts = np.quantile(prediction_df["absolute_pv_change_fraction_per_15min"], [0.25, 0.50, 0.75])
    cut_pct = ramp_cuts * 100.0
    ax1.set_xticks(x_ramp)
    ax1.set_xticklabels(
        [
            f"Q1\n<={cut_pct[0]:.2f}%",
            f"Q2\n{cut_pct[0]:.2f}-{cut_pct[1]:.2f}%",
            f"Q3\n{cut_pct[1]:.2f}-{cut_pct[2]:.2f}%",
            f"Q4\n>{cut_pct[2]:.2f}%",
        ]
    )
    ax1.set_xlim(0.72, 4.28)
    ramp_low = float(np.min(ramp_summary["mean"] - ramp_summary["ci95"]))
    ramp_high = float(np.max(ramp_summary["mean"] + ramp_summary["ci95"]))
    ax1.set_ylim(max(0, ramp_low - 0.010), ramp_high + 0.012)
    ax1.set_xlabel("Absolute PV change per 15 min (% capacity)")
    ax1.set_ylabel("nMAE")
    ax1.text(
        3.72,
        0.965,
        "High-variability regime",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.7,
        color="#6D674F",
    )
    base.quiet_grid(ax1)
    base.panel_label_fixed(fig, ax1, "b")

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=3,
        columnspacing=1.5,
        handlelength=2.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.82)

    pd.DataFrame(
        [
            {
                "panel": "a",
                "metric_spacing_multiplier": metric_spacing,
                "summary_marker_size_pt": summary_marker_size,
                "raw_marker_area_pt2": raw_marker_area,
                "errorbar_linewidth_pt": error_lw,
                "errorbar_capsize_pt": error_cap,
                "additional_statistics_box": False,
            },
            {
                "panel": "b",
                "q4_highlight_color": "#E8D68A",
                "q4_highlight_alpha": 0.08,
                "summary_marker_size_pt": summary_marker_size,
                "errorbar_linewidth_pt": error_lw,
                "errorbar_capsize_pt": error_cap,
                "online_linewidth_pt": styles["Online Lyra"]["lw"],
            },
        ]
    ).to_csv(DATA_OUT / "fig05_v2_layout_audit.csv", index=False)
    save_pub(fig, "fig05_daytime_error_agreement")


def render_fig06_direct_latency() -> None:
    res = pd.read_csv(DATA_OUT / "fig06_efficiency_resources.csv")
    res["params_k_plot"] = res["params_k"].clip(lower=0.01)
    accuracy_order = res.sort_values(["nrmse", "params_k_plot"])["model_display"].tolist()

    pareto_mask = []
    for _, row in res.iterrows():
        no_worse = (res["params_k_plot"] <= row["params_k_plot"]) & (res["nrmse"] <= row["nrmse"])
        strictly_better = (res["params_k_plot"] < row["params_k_plot"]) | (res["nrmse"] < row["nrmse"])
        pareto_mask.append(not bool((no_worse & strictly_better).any()))
    res["is_pareto"] = pareto_mask
    pareto = res.loc[res["is_pareto"]].sort_values(["params_k_plot", "nrmse"])

    fig = plt.figure(figsize=(7.25, 3.45))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.46)
    fig.subplots_adjust(left=0.085, right=0.925, bottom=0.17, top=0.93, wspace=0.46)
    ax0 = fig.add_subplot(gs[0, 0])
    bubble_area_scale = 205.0 / float(res["latency_mean_ms"].max())

    label_positions = {
        "Online Lyra": (15.6, 0.09438, "right"),
        "DLinear": (22.5, 0.09674, "left"),
        "NLinear": (7.1, 0.09838, "left"),
        "PhaseFormer": (27.0, 0.09868, "left"),
        "TimeKAN": (61.5, 0.09716, "left"),
        "PatchTST": (136.0, 0.09886, "left"),
        "iTransformer": (82.0, 0.09428, "right"),
        "Olivia": (151.0, 0.09622, "left"),
    }
    competitive = {"iTransformer", "Olivia"}
    audit_rows = []
    label_artists = {}
    for _, row in res.iterrows():
        model = row["model_display"]
        if model == "Online Lyra":
            color, edge, lw, alpha, zorder = COLORS["online"], "#08385E", 1.25, 0.98, 7
        elif model in competitive:
            color, edge, lw, alpha, zorder = "#86AEC5", "#5E8298", 0.75, 0.90, 4
        else:
            color, edge, lw, alpha, zorder = "#BDD0DE", "white", 0.60, 0.88, 3
        area = bubble_area_scale * float(row["latency_mean_ms"])
        ax0.scatter(
            row["params_k_plot"], row["nrmse"], s=area, color=color, edgecolor=edge,
            lw=lw, alpha=alpha, zorder=zorder,
        )
        label_x, label_y, ha = label_positions[model]
        label_artists[model] = ax0.annotate(
            f"{model}\n{fmt_latency(float(row['latency_mean_ms']))}",
            xy=(row["params_k_plot"], row["nrmse"]),
            xytext=(label_x, label_y),
            ha=ha,
            va="center",
            fontsize=6.25,
            linespacing=1.04,
            fontweight="bold" if model == "Online Lyra" else "normal",
            color=COLORS["online"] if model == "Online Lyra" else "#3F4852",
            arrowprops=dict(arrowstyle="-", color="#99A3AD", lw=0.48, shrinkA=2, shrinkB=3),
            zorder=9,
        )
        audit_rows.append(
            {
                "model_display": model,
                "cuda_latency_ms": float(row["latency_mean_ms"]),
                "bubble_area_pt2": area,
                "area_per_ms_pt2": bubble_area_scale,
                "label_x": label_x,
                "label_y": label_y,
                "is_pareto": bool(row["is_pareto"]),
            }
        )

    if len(pareto) >= 2:
        ax0.plot(
            pareto["params_k_plot"],
            pareto["nrmse"],
            color="#7C8791",
            lw=0.80,
            ls=(0, (4, 2)),
            zorder=1,
        )
        anchor = pareto.iloc[-2]
        ax0.annotate(
            "Pareto frontier",
            xy=(float(anchor["params_k_plot"]), 0.09575),
            xytext=(25.0, 0.09575),
            ha="left",
            va="center",
            fontsize=6.4,
            color="#66717B",
            arrowprops=dict(arrowstyle="-", color="#7C8791", lw=0.48, shrinkA=2, shrinkB=2),
            zorder=8,
        )
    ax0.set_xscale("log")
    ax0.set_xlim(6, 250)
    ax0.set_ylim(0.0939, 0.09925)
    ax0.set_xlabel("Trainable parameters (K, log scale)")
    ax0.set_ylabel("Mean nRMSE")
    base.quiet_grid(ax0)
    from matplotlib.ticker import LogLocator, NullFormatter

    ax0.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax0.xaxis.set_minor_formatter(NullFormatter())
    ax0.grid(which="major", axis="x", color="#DDE3E9", lw=0.60)
    ax0.grid(which="minor", axis="x", color="#EDF0F4", lw=0.45, ls=":")
    base.panel_label_fixed(fig, ax0, "a", dx_pt=-13.0, dy_pt=3.0)

    ax1 = fig.add_subplot(gs[0, 1])
    cols = [
        ("params_k", "Params"),
        ("checkpoint_size_mb", "Model\nsize"),
        ("latency_mean_ms", "CUDA\nlat."),
        ("cpu_latency_ms", "CPU\nlat."),
    ]
    heat = res.set_index("model_display")[[col for col, _ in cols]].reindex(accuracy_order)
    ratio = heat.divide(heat.loc["Online Lyra"], axis=1)
    ratio.loc["Online Lyra", :] = 1.0
    log2_ratio = np.log2(ratio)
    lim = max(2.0, float(np.nanmax(np.abs(log2_ratio.to_numpy()))))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = LinearSegmentedColormap.from_list(
        "ratio_diverging_v2", ["#3B78A8", "#FFFFFF", "#C56C43"], N=257
    )
    im = ax1.imshow(log2_ratio.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax1.set_xticks(np.arange(len(cols)))
    ax1.set_xticklabels([label for _, label in cols])
    ax1.set_yticks(np.arange(len(accuracy_order)))
    ax1.set_yticklabels(accuracy_order, fontsize=7.5)
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            value = float(ratio.iloc[i, j])
            rgba = cmap(norm(float(log2_ratio.iloc[i, j])))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.56 else "#20262D"
            label = f"{value:.2f}x" if value < 0.1 else (f"{value:.0f}x" if value >= 10 else f"{value:.1f}x")
            text = ax1.text(j, i, label, ha="center", va="center", fontsize=7.0, color=color)
            text.set_path_effects(
                [patheffects.withStroke(linewidth=1.0, foreground="black" if color == "white" else "white", alpha=0.42)]
            )
    cb = fig.colorbar(im, ax=ax1, fraction=0.052, pad=0.020)
    ticks = np.array([0.25, 0.5, 1, 2, 4, 8], dtype=float)
    ticks = ticks[(np.log2(ticks) >= -lim - 1e-9) & (np.log2(ticks) <= lim + 1e-9)]
    cb.set_ticks(np.log2(ticks))
    cb.set_ticklabels([f"{tick:g}x" for tick in ticks])
    cb.ax.axhline(0.0, color="#6F7780", lw=0.80, zorder=4)
    cb.set_label("Resource ratio")
    base.panel_label_fixed(fig, ax1, "b", dx_pt=-13.0, dy_pt=3.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {name: artist.get_window_extent(renderer=renderer) for name, artist in label_artists.items()}
    overlap_pairs = []
    names = list(boxes)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            if boxes[name_a].overlaps(boxes[name_b]):
                overlap_pairs.append(f"{name_a}|{name_b}")
    audit = pd.DataFrame(audit_rows)
    audit["bubble_area_to_latency_ratio"] = audit["bubble_area_pt2"] / audit["cuda_latency_ms"]
    audit["direct_latency_label"] = True
    audit["bubble_legend_used"] = False
    audit["label_overlap_pairs"] = ";".join(overlap_pairs)
    audit.to_csv(DATA_OUT / "fig06_direct_latency_label_audit.csv", index=False)
    if overlap_pairs:
        raise ValueError(f"Fig. 6 direct labels overlap: {overlap_pairs}")
    save_pub(fig, "fig06_efficiency")


def render_figS04_square_shared_scale() -> None:
    df = pd.read_csv(DATA_OUT / "fig05_daytime_predictions_h96_seed2028.csv")
    observed = df["observed"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, 46)
    counts_by_model = {}
    stat_rows = []
    for model in MODELS:
        predicted = df[model].to_numpy(dtype=float)
        counts, _, _ = np.histogram2d(observed, predicted, bins=[edges, edges])
        counts_by_model[model] = counts
        slope, intercept = np.polyfit(observed, predicted, 1)
        corr = float(np.corrcoef(observed, predicted)[0, 1])
        stat_rows.append(
            {
                "model": model,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": corr**2,
                "normalized_bias": float(np.mean(predicted - observed)),
                "n_points": int(len(observed)),
            }
        )
    stats = pd.DataFrame(stat_rows).set_index("model")
    cmap = LinearSegmentedColormap.from_list(
        "calibration_density_v2", ["#F7FAFC", "#DCEAF3", "#A9CCE0", "#5DA8C9", "#174A7C"]
    )
    cmap.set_bad("white")
    vmax = max(float(counts.max()) for counts in counts_by_model.values())
    norm = LogNorm(vmin=1.0, vmax=vmax)

    fig = plt.figure(figsize=(7.25, 2.75))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    mesh = None
    for ax, model, letter in zip(axes, MODELS, ["a", "b", "c"]):
        counts = counts_by_model[model].T
        masked = np.ma.masked_where(counts <= 0, counts)
        mesh = ax.pcolormesh(edges, edges, masked, cmap=cmap, norm=norm, shading="flat")
        ax.plot([0, 1], [0, 1], color="#4D4D4D", lw=0.78, ls="--", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
        ax.set_title(model, pad=5)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        row = stats.loc[model]
        ax.text(
            0.035,
            0.965,
            rf"slope = {row['slope']:.2f}" + "\n" + rf"$R^2$ = {row['r2']:.2f}" + "\n" + rf"bias = {row['normalized_bias']:+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            linespacing=1.16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
            zorder=4,
        )
        base.panel_label_fixed(fig, ax, letter, dx_pt=-12.0, dy_pt=3.0)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    if mesh is not None:
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label("Shared bin count")
    fig.supxlabel("Observed normalized PV", y=0.045)
    fig.supylabel("Predicted normalized PV", x=0.020)
    fig.subplots_adjust(left=0.085, right=0.955, bottom=0.20, top=0.86, wspace=0.18)

    fig.canvas.draw()
    geometry = []
    for model, ax in zip(MODELS, axes):
        box = ax.get_window_extent(renderer=fig.canvas.get_renderer())
        geometry.append(
            {
                "model": model,
                "panel_width_px": float(box.width),
                "panel_height_px": float(box.height),
                "width_height_ratio": float(box.width / box.height),
                "shared_vmin": 1.0,
                "shared_vmax": vmax,
                "shared_colorbar": True,
            }
        )
    geometry_df = pd.DataFrame(geometry)
    geometry_df.to_csv(DATA_OUT / "figS04_panel_geometry_audit.csv", index=False)
    if not np.allclose(geometry_df["width_height_ratio"], 1.0, atol=0.015):
        raise ValueError("Fig. S4 panels are not square within tolerance")
    save_pub(fig, "figS04_daytime_calibration_density")


def render_figS05_contrast() -> None:
    summary = pd.read_csv(DATA_OUT / "figS05_hankel_spectrum_summary.csv")
    units = pd.read_csv(DATA_OUT / "figS05_hankel_spectrum_window_units.csv")
    site = (
        units.groupby(["site_id", "rank"], as_index=False)
        .agg(
            normalized_singular_value=("normalized_singular_value", "median"),
            cumulative_squared_energy=("cumulative_squared_energy", "median"),
        )
    )
    max_rank = 20
    shown = summary.loc[summary["rank"] <= max_rank].copy()
    x = shown["rank"].to_numpy(dtype=float)
    median_color = "#07518A"
    site_color = "#9EADBC"
    band_color = "#E7F1F7"

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.82, wspace=0.27)
    axes[0].fill_between(
        x,
        shown["normalized_singular_value_q25"],
        shown["normalized_singular_value_q75"],
        color=band_color,
        alpha=0.86,
        lw=0,
        zorder=1,
    )
    axes[1].fill_between(
        x,
        shown["cumulative_energy_q25"] * 100.0,
        shown["cumulative_energy_q75"] * 100.0,
        color=band_color,
        alpha=0.86,
        lw=0,
        zorder=1,
    )

    for site_id, group in site.loc[site["rank"] <= max_rank].groupby("site_id"):
        group = group.sort_values("rank")
        axes[0].plot(group["rank"], group["normalized_singular_value"], color=site_color, lw=0.62, alpha=0.72, zorder=2)
        axes[1].plot(group["rank"], group["cumulative_squared_energy"] * 100.0, color=site_color, lw=0.62, alpha=0.72, zorder=2)

    axes[0].plot(x, shown["normalized_singular_value_median"], color=median_color, lw=1.95, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Normalized singular value $\sigma_i/\sigma_1$")
    axes[1].plot(x, shown["cumulative_energy_median"] * 100.0, color=median_color, lw=1.95, zorder=3)
    energy_floor = float(max(0.0, shown["cumulative_energy_q25"].min() * 100.0 - 4.0))
    axes[1].set_ylim(energy_floor, 100.5)
    axes[1].set_ylabel("Cumulative spectral energy (%)")

    for ax, letter in zip(axes, ["a", "b"]):
        ax.set_xlim(1, max_rank)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Singular-value index")
        base.quiet_grid(ax)
        base.panel_label_fixed(fig, ax, letter, dx_pt=-13.0, dy_pt=3.0)

    handles = [
        Line2D([0], [0], color=median_color, lw=1.95, label="Pooled median"),
        Patch(facecolor=band_color, edgecolor="#D8E6EF", lw=0.3, alpha=0.86, label="Interquartile range"),
        Line2D([0], [0], color=site_color, lw=0.75, alpha=0.75, label="Site medians (n=8)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        columnspacing=1.5,
        handlelength=2.0,
        handletextpad=0.55,
    )
    axes[1].legend(handles=handles, loc="best", ncol=1)
    pd.DataFrame(
        [
            {"element": "pooled median", "color": median_color, "alpha": 1.0, "linewidth_pt": 1.95},
            {"element": "interquartile range", "color": band_color, "alpha": 0.86, "linewidth_pt": 0.0},
            {"element": "site medians", "color": site_color, "alpha": 0.72, "linewidth_pt": 0.62},
        ]
    ).to_csv(DATA_OUT / "figS05_visual_hierarchy_audit.csv", index=False)
    save_pub(fig, "figS05_hankel_singular_spectrum")


UNCHANGED_FIGURES = [
    ("Fig. 1", "fig01_accuracy"),
    ("Fig. 2", "fig02_robustness"),
    ("Fig. S1", "figS01_fallback_grouped_dot"),
    ("Fig. S2", "figS02_core_ablation_cold_start"),
    ("Fig. S3", "figS03_paired_effect_size_significance"),
]


def sync_unchanged_figures() -> None:
    audit_rows = []
    for figure, stem in UNCHANGED_FIGURES:
        for format_name, extension in [("svg", "svg"), ("pdf", "pdf"), ("png", "png"), ("tiff", "tiff")]:
            source = OLD_OUT / format_name / f"{stem}.{extension}"
            destination = OUT / format_name / f"{stem}.{extension}"
            if not source.exists():
                raise FileNotFoundError(f"Missing unchanged source figure: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            audit_rows.append(
                {
                    "figure": figure,
                    "stem": stem,
                    "format": format_name,
                    "source_file": f"figures_revised/{format_name}/{source.name}",
                    "destination_file": f"figures_revised_v2/{format_name}/{destination.name}",
                    "copied": True,
                }
            )

    data_prefixes = ["fig01_", "fig02_", "fig04_", "figS01_", "figS02_", "figS03_"]
    for prefix in data_prefixes:
        for source in sorted((OLD_OUT / "data").glob(f"{prefix}*")):
            if source.is_file():
                shutil.copy2(source, DATA_OUT / source.name)
    pd.DataFrame(audit_rows).to_csv(DATA_OUT / "unchanged_figure_sync_manifest.csv", index=False)


def make_overview() -> None:
    from PIL import Image

    refined = [
        ("Fig. 3", "fig03_case_trace_gate"),
        ("Fig. 5", "fig05_daytime_error_agreement"),
        ("Fig. 6", "fig06_efficiency"),
        ("Fig. S4", "figS04_daytime_calibration_density"),
        ("Fig. S5", "figS05_hankel_singular_spectrum"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for ax, (label, stem) in zip(axes.ravel(), refined):
        with Image.open(OUT / "png" / f"{stem}.png") as source:
            source.thumbnail((1900, 1350), Image.Resampling.LANCZOS)
            preview = np.asarray(source.convert("RGB"))
        ax.imshow(preview, interpolation="lanczos")
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left", pad=5)
        ax.axis("off")
    for ax in axes.ravel()[len(refined):]:
        ax.axis("off")
    fig.savefig(OUT / "refined_five_overview_highres.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    full_set = [
        ("Fig. 1", "fig01_accuracy"),
        ("Fig. 2", "fig02_robustness"),
        ("Fig. 3", "fig03_case_trace_gate"),
        ("Fig. 5", "fig05_daytime_error_agreement"),
        ("Fig. 6", "fig06_efficiency"),
        ("Fig. S1", "figS01_fallback_grouped_dot"),
        ("Fig. S2", "figS02_core_ablation_cold_start"),
        ("Fig. S3", "figS03_paired_effect_size_significance"),
        ("Fig. S4", "figS04_daytime_calibration_density"),
        ("Fig. S5", "figS05_hankel_singular_spectrum"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15.0, 14.2), constrained_layout=True)
    for ax, (label, stem) in zip(axes.ravel(), full_set):
        with Image.open(OUT / "png" / f"{stem}.png") as source:
            source.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
            preview = np.asarray(source.convert("RGB"))
        ax.imshow(preview, interpolation="lanczos")
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left", pad=5)
        ax.axis("off")
    for ax in axes.ravel()[len(full_set):]:
        ax.axis("off")
    fig.savefig(OUT / "full_set_overview_highres.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure_overview_complete_highres.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_manifest() -> None:
    pd.DataFrame(
        [
            ("Fig. 1", "fig01_accuracy", "Accuracy across forecast horizons", "re-rendered by scripts/make_fig1.py"),
            ("Fig. 2", "fig02_robustness", "Site and capacity robustness", "preserved from figures_revised"),
            ("Fig. 3", "fig03_case_trace_gate", "Critical event -> gate -> recovery narrative", "re-rendered by scripts/make_fig3.py"),
            ("Fig. 5", "fig05_daytime_error_agreement", "Site-level normalized metrics and PV-variability-stratified nMAE", "refined original two-panel layout"),
            ("Fig. 6", "fig06_efficiency", "Direct CUDA-latency labels and audited Pareto frontier", "re-rendered"),
            ("Fig. S1", "figS01_fallback_grouped_dot", "Fallback behavior by site and horizon", "preserved from figures_revised"),
            ("Fig. S2", "figS02_core_ablation_cold_start", "Core ablation and cold-start sensitivity", "preserved from figures_revised"),
            ("Fig. S3", "figS03_paired_effect_size_significance", "Paired effect sizes and statistical significance", "preserved from figures_revised"),
            ("Fig. S4", "figS04_daytime_calibration_density", "Square panels under one shared density scale", "re-rendered"),
            ("Fig. S5", "figS05_hankel_singular_spectrum", "Stronger median/IQR/site visual hierarchy", "re-rendered"),
        ],
        columns=["figure", "stem", "revision", "status"],
    ).to_csv(OUT / "figure_manifest_v2.csv", index=False)
    notes = """# Figure refinement round 2

- The original `figures_revised` package was not overwritten.
- Fig. 3, Fig. 5, Fig. 6, Fig. S4, and Fig. S5 were re-rendered; the other six
  figures and their source tables were copied unchanged into this self-contained package.
- Original event selection, Smart Persistence construction, resource measurements,
  Pareto dominance logic, and Hankel/SVD preprocessing were reused before styling.
- Fig. 3 stage labels are derived by explicit rules recorded in source data.
- Fig. 5 restores the original two-panel evidence structure. Panel a reports
  eight site-level units with 95% CIs for nMAE, nRMSE, and P90 absolute error;
  Panel b reports the same models across four PV-change quartiles.
- Fig. 6 bubble area remains strictly proportional to measured CUDA latency; the
  former size legend is replaced by direct per-model latency labels.
- Fig. S4 uses a single shared logarithmic color scale and audited square panels.
"""
    (OUT / "notes" / "REFINEMENT_LOG.md").write_text(notes, encoding="utf-8")


def main() -> None:
    base.configure_style()
    base.ensure_dirs()

    # Original routines regenerate all source data and audit files in the new root.
    base.fig03_case_trace_revised()
    render_fig03_story()

    base.fig05_daytime_error_agreement_revised()
    render_fig05_error_and_agreement()

    base.fig06_efficiency_revised()
    render_fig06_direct_latency()

    base.figS04_daytime_calibration_density()
    render_figS04_square_shared_scale()

    base.figS05_hankel_singular_spectrum()
    render_figS05_contrast()

    sync_unchanged_figures()
    write_manifest()
    make_overview()
    print(f"Wrote refined figures to {OUT}")


if __name__ == "__main__":
    main()
