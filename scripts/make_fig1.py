#!/usr/bin/env python3
# ruff: noqa: E402
"""Fig. 1 redraw, version 3: refined focus + context, streamlined legend,
and adjusted panel c aesthetics per updated specification.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from scipy.interpolate import PchipInterpolator

PACK = ROOT / "outputs" / "paper_result_pack"
SRC = PACK / "figures_revised_v2" / "data"
OUT = PACK / "figures_revised_v2"

ACC_CSV = SRC / "fig01_accuracy_seed_ci.csv"

# palette
C_ONLINE = "#0B4F8A"
C_BEST = "#D8752D"
C_CONTEXT = "#AEB6C0"
C_BAND = "#DCE1E8"
C_INK = "#242424"
C_GRID = "#E8EBF0"

FOCUS_LW = 2.4
BEST_LW = 1.9
CONTEXT_LW = 0.65
CONTEXT_ALPHA = 0.55
PANEL_LABEL_SIZE = 10
TCRIT_N3 = 4.303  # 95% t critical, df=2 (n=3 seeds)

def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.55,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "lines.solid_capstyle": "round",
        }
    )

def quiet_grid(ax):
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
    ax.grid(axis="both", color="#C9CED4", lw=0.36, alpha=0.28)
    ax.set_axisbelow(True)

def panel_label(ax, letter):
    return None


def smooth_line(ax, x, y, **plot_kwargs):
    """Draw a shape-preserving guide that passes through every measured point."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 240)
    dense_y = PchipInterpolator(x, y)(dense_x)
    return ax.plot(
        dense_x,
        dense_y,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        **plot_kwargs,
    )

def strongest_baseline(acc):
    exclude = {"Online Lyra", "Offline Lyra"}
    r = (acc.loc[~acc["model_display"].isin(exclude)]
         .groupby("model_display")["nrmse_mean"].mean().sort_values())
    return r.index[0]

def dodged_errorbar(fig, ax, x, y, ci, color, offset_pt, zorder):
    if not np.isfinite(ci).any():
        return
    tr = ax.transData + ScaledTranslation(offset_pt / 72.0, 0.0, fig.dpi_scale_trans)
    ax.errorbar(x, y, yerr=ci, fmt="none", ecolor=color, elinewidth=0.85,
                capsize=1.6, capthick=0.85, alpha=0.9, zorder=zorder + 1, transform=tr)

def zoom_range(values, step, pad_frac=0.10):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    pad = max((hi - lo) * pad_frac, step * 0.5)
    return np.floor((lo - pad) / step) * step, np.ceil((hi + pad) / step) * step

def draw_metric_panel(fig, ax, acc, mean_col, ci_col, focus, others, letter, ylabel):
    ctx = acc[acc["model_display"].isin(others)]
    env = ctx.groupby("lead_time_h")[mean_col].agg(["min", "max"]).sort_index()
    ax.fill_between(env.index, env["min"], env["max"], color=C_BAND, alpha=0.7, lw=0, zorder=1)

    for m in others:
        sub = acc[acc["model_display"] == m].sort_values("lead_time_h")
        smooth_line(ax, sub["lead_time_h"], sub[mean_col], color=C_CONTEXT,
                    lw=CONTEXT_LW, alpha=CONTEXT_ALPHA, zorder=2, ls="-")

    for name, color, ls, marker, lw, ms, dodge, z in [
        ("Online Lyra", C_ONLINE, "-", "o", FOCUS_LW, 4.6, +2.4, 10),
        (focus, C_BEST, "--", "s", BEST_LW, 4.0, -2.4, 8),
    ]:
        sub = acc[acc["model_display"] == name].sort_values("lead_time_h")
        x = sub["lead_time_h"].to_numpy(float)
        y = sub[mean_col].to_numpy(float)
        ci = sub[ci_col].to_numpy(float)
        smooth_line(ax, x, y, color=color, ls=ls, lw=lw, alpha=0.97, zorder=z)
        tr = ax.transData + ScaledTranslation(dodge / 72.0, 0.0, fig.dpi_scale_trans)
        ax.plot(x, y, ls="none", marker=marker, ms=ms, mfc=color, mec="white",
                mew=0.6, alpha=0.9, zorder=z + 1, transform=tr)
        dodged_errorbar(fig, ax, x, y, ci, color, dodge, z)

    ylo, yhi = zoom_range(acc[mean_col].to_numpy(), 0.005)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.15, 25.30)
    ax.set_xticks([1, 3, 6, 12, 24])
    ax.set_ylabel(ylabel)
    quiet_grid(ax)
    panel_label(ax, letter)

def relative_improvement_vs_baseline(baseline: str) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        "rev", str(ROOT / "scripts" / "make_pv_paper_figures_revised.py"))
    rev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rev)
    rep = rev.load_replicate_results()
    unit = rep.groupby(["seed", "horizon", "lead_time_h", "model_display"],
                       as_index=False)[["nrmse"]].mean()

    piv = unit.pivot_table(index=["seed", "horizon", "lead_time_h"],
                           columns="model_display", values="nrmse")
    rel = ((piv[baseline] - piv["Online Lyra"]) / piv[baseline] * 100.0).reset_index(name="rel_improvement_pct")

    g = rel.groupby("lead_time_h")["rel_improvement_pct"].agg(["mean", "std", "count"]).reset_index()
    g["ci95"] = TCRIT_N3 * g["std"] / np.sqrt(g["count"])
    g = g.rename(columns={"count": "n"})
    return g

def draw_relative_improvement_panel(fig, ax, rel: pd.DataFrame, baseline: str):
    # Neutral dark-grey/black line with moderate thickness
    ax.axhline(0, color="#404040", lw=0.8, zorder=3)

    x = rel["lead_time_h"].to_numpy(float)
    y = rel["mean"].to_numpy(float)
    ci = rel["ci95"].to_numpy(float)
    sig = np.abs(y) > ci

    smooth_line(ax, x, y, color=C_ONLINE, ls="-", lw=FOCUS_LW, alpha=0.97, zorder=5)

    # Linewidth reduced by ~20%, capsize shortened by ~25%, drawn behind markers
    ax.errorbar(x, y, yerr=ci, fmt="none", ecolor=C_ONLINE, elinewidth=0.68,
                capsize=0.8, capthick=0.68, alpha=0.75, zorder=6)

    # Marker size increased slightly to maintain visual dominance over error bars
    if np.any(sig):
        ax.plot(x[sig], y[sig], ls="none", marker="o", ms=5.2,
                mfc=C_ONLINE, mec="white", mew=0.6, zorder=7)
    if np.any(~sig):
        ax.plot(x[~sig], y[~sig], ls="none", marker="o", ms=5.2,
                mfc="white", mec=C_ONLINE, mew=1.0, zorder=7)

    ax.set_xlim(0.15, 25.30)
    ax.set_xticks([1, 3, 6, 12, 24])
    all_y = np.concatenate([y + ci, y - ci]) if len(rel) else np.array([-1.0, 1.0])
    lim = max(np.nanmax(np.abs(all_y)), 1.0) if np.isfinite(all_y).any() else 1.0
    ax.set_ylim(-1.12 * lim, 1.12 * lim)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))

    # Refined label without redundant explanatory text
    ax.set_ylabel("Relative nRMSE improvement (%)")
    quiet_grid(ax)
    panel_label(ax, "c")

def build_caption_text(focus: str, others: list[str], n_ci: int) -> str:
    others_list = ", ".join(others[:-1]) + f", and {others[-1]}" if len(others) > 1 else others[0]
    return (
        f"**Figure 1. Forecast accuracy across horizons.** "
        f"(a) Mean nRMSE and (b) mean nMAE across forecast horizons for Online Lyra (blue) versus {focus} "
        f"(orange dashed). {focus} is selected as the representative strongest baseline according to the "
        f"predefined average-performance criterion used in the analysis. Grey curves denote the remaining "
        f"six baselines: {others_list}. "
        f"(c) Relative nRMSE improvement of Online Lyra over {focus} at each forecast horizon. "
        f"Positive values indicate lower nRMSE for Online Lyra than for {focus}. "
        f"Error bars are two-sided 95% t-confidence intervals across random seeds (n = {n_ci} seeds). "
        f"These intervals quantify sensitivity to random initialization and do not represent cross-site "
        f"population-level inference. Filled markers denote horizons where the 95% CI excludes zero; "
        f"hollow markers denote horizons where the CI includes zero."
    )

def run_layout_qc(fig, axes) -> list[str]:
    problems: list[str] = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax in axes:
        ax_bbox = ax.get_window_extent(renderer)
        tol_px = 3.0
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            xd, yd = line.get_xdata(), line.get_ydata()
            xd = np.asarray(xd, dtype=float)
            yd = np.asarray(yd, dtype=float)
            finite = np.isfinite(xd) & np.isfinite(yd)
            if not finite.any():
                continue
            pts = np.column_stack([xd[finite], yd[finite]])
            disp = line.get_transform().transform(pts)
            xpx, ypx = disp[:, 0], disp[:, 1]
            if (xpx < ax_bbox.x0 - tol_px).any() or (xpx > ax_bbox.x1 + tol_px).any():
                problems.append(f"{ax.get_label() or ax}: a curve/marker renders outside the axes horizontally.")
            if (ypx < ax_bbox.y0 - tol_px).any() or (ypx > ax_bbox.y1 + tol_px).any():
                problems.append(f"{ax.get_label() or ax}: a curve/marker renders outside the axes vertically.")

    for ax in axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        leg_bbox = leg.get_window_extent(renderer)
        ax_bbox = ax.get_window_extent(renderer)
        inter_w = max(0.0, min(leg_bbox.x1, ax_bbox.x1) - max(leg_bbox.x0, ax_bbox.x0))
        inter_h = max(0.0, min(leg_bbox.y1, ax_bbox.y1) - max(leg_bbox.y0, ax_bbox.y0))
        inter_area = inter_w * inter_h
        ax_area = ax_bbox.width * ax_bbox.height
        if ax_area > 0 and (inter_area / ax_area) > 0.22:
            problems.append(f"{ax.get_label() or ax}: legend covers >22% of the plotting area; likely overlaps data.")

    boxes = [ax.get_position() for ax in axes]
    tops = [b.y1 for b in boxes]
    bottoms = [b.y0 for b in boxes]
    heights = [b.height for b in boxes]
    if max(tops) - min(tops) > 1e-3:
        problems.append("Panel tops are not aligned across a/b/c.")
    if max(bottoms) - min(bottoms) > 1e-3:
        problems.append("Panel bottoms are not aligned across a/b/c.")
    if max(heights) - min(heights) > 1e-3:
        problems.append("Panel heights differ across a/b/c.")

    label_sizes = {ax.yaxis.label.get_fontsize() for ax in axes}
    tick_sizes = {lbl.get_fontsize() for ax in axes for lbl in ax.get_xticklabels() + ax.get_yticklabels()}
    if len(label_sizes) > 1:
        problems.append(f"Axis label font sizes differ across panels: {label_sizes}.")
    if len(tick_sizes) > 1:
        problems.append(f"Tick label font sizes differ across panels: {tick_sizes}.")

    for txt in fig.findobj(match=plt.Text):
        s = txt.get_text().strip()
        if not s:
            continue
        bb = txt.get_window_extent(renderer)
        fig_bbox = fig.bbox
        margin = 2.0
        if bb.x0 < fig_bbox.x0 - margin or bb.x1 > fig_bbox.x1 + margin:
            problems.append(f"Text element touches/exceeds figure border horizontally: '{s[:40]}'")
        if bb.y0 < fig_bbox.y0 - margin or bb.y1 > fig_bbox.y1 + margin:
            problems.append(f"Text element touches/exceeds figure border vertically: '{s[:40]}'")

    return problems

def main():
    configure_style()
    acc = pd.read_csv(ACC_CSV)
    focus = strongest_baseline(acc)
    all_models = list(acc["model_display"].unique())
    others = [m for m in all_models if m not in {"Online Lyra", focus}]

    rel = relative_improvement_vs_baseline(focus)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    rel.to_csv(OUT / "data" / "fig01_relative_improvement_vs_baseline.csv", index=False)

    fig = plt.figure(figsize=(7.25, 3.15))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.46, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    draw_metric_panel(fig, axes[0], acc, "nrmse_mean", "nrmse_ci95", focus, others, "a", "nRMSE")
    draw_metric_panel(fig, axes[1], acc, "nmae_mean", "nmae_ci95", focus, others, "b", "nMAE")
    draw_relative_improvement_panel(fig, axes[2], rel, focus)

    handles = [
        Line2D([0], [0], color=C_ONLINE, lw=FOCUS_LW, ls="-", marker="o", ms=4.6, mec="white", mew=0.6),
        Line2D([0], [0], color=C_BEST, lw=BEST_LW, ls="--", marker="s", ms=4.0, mec="white", mew=0.6),
        Line2D([0], [0], color=C_CONTEXT, lw=CONTEXT_LW + 0.15),
    ]

    labels = ["Online Lyra", focus, "Other baselines"]

    for ax in axes[:2]:
        legend = ax.legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            frameon=True,
            facecolor="white",
            framealpha=0.68,
            edgecolor="#D5DADF",
            fontsize=6.2,
            borderaxespad=0.62,
            borderpad=0.42,
            labelspacing=0.34,
            handlelength=1.45,
            handletextpad=0.48,
        )
        legend.get_frame().set_linewidth(0.45)

    relative_handles = [
        Line2D([0], [0], color=C_ONLINE, lw=FOCUS_LW, label=f"vs {focus}"),
        Line2D([0], [0], marker="o", ls="none", ms=4.6, mfc=C_ONLINE, mec="white", mew=0.6,
               label="95% CI excludes zero"),
        Line2D([0], [0], marker="o", ls="none", ms=4.6, mfc="white", mec=C_ONLINE, mew=1.0,
               label="95% CI includes zero"),
    ]
    relative_legend = axes[2].legend(
        handles=relative_handles,
        loc="best",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.68,
        edgecolor="#D5DADF",
        fontsize=5.8,
        borderaxespad=0.62,
        borderpad=0.38,
        labelspacing=0.30,
        handlelength=1.35,
        handletextpad=0.42,
    )
    relative_legend.get_frame().set_linewidth(0.45)

    fig.supxlabel("Forecast horizon (h)", x=0.5, y=0.005, fontsize=8)
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.24, top=0.90, wspace=0.46)

    problems = run_layout_qc(fig, axes)
    if problems:
        msg = "\n  - ".join(problems)
        plt.close(fig)
        raise RuntimeError(
            "Layout QC failed; figure was NOT exported. Fix the following and re-run:\n  - " + msg
        )

    for ext, kw in [("png", dict(dpi=600)), ("pdf", {}), ("svg", {}), ("tiff", dict(dpi=600))]:
        (OUT / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / ext / f"fig01_accuracy.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)

    n_ci = int(rel["n"].max()) if len(rel) else 3
    caption_text = build_caption_text(focus, others, n_ci)

    (OUT / "notes").mkdir(parents=True, exist_ok=True)
    (OUT / "notes" / "caption_fig01.md").write_text(caption_text + "\n", encoding="utf-8")

    print(f"strongest baseline = {focus}")
    print("layout QC: passed, no issues found")
    print(f"wrote fig1 redraw v3 to {OUT}")

if __name__ == "__main__":
    main()
