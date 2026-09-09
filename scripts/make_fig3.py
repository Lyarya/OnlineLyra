#!/usr/bin/env python3
"""
Refined rendering script for Figure 3: Critical event -> gate -> recovery narrative.

This is a self-contained, single-figure script. It reuses the original
data-generation function from ``make_pv_paper_figures_revised`` and replaces
only the rendering layer for Fig. 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Local path and data-loading dependencies (kept identical to the
# original architecture).
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import make_pv_paper_figures_revised as base  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec, patheffects  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Patch, Rectangle  # noqa: E402

# ---------------------------------------------------------
# Output configuration
# ---------------------------------------------------------
OUT = base.PACK / "figures_revised_v2"
DATA_OUT = OUT / "data"

COLORS = base.COLORS
GATE_ONLINE = "#154E80"

# Output stem: no spaces, safe for LaTeX / shell / cross-platform paths.
FIGURE_STEM = "fig03_case_trace_gate"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save_pub(fig: plt.Figure, stem: str, formats: tuple[str, ...] = ("svg", "pdf", "png", "tiff")) -> None:
    """Save a figure to each requested format under OUT/<format>/<stem>.<ext>."""
    for fmt in formats:
        (OUT / fmt).mkdir(parents=True, exist_ok=True)
    if "svg" in formats:
        fig.savefig(OUT / "svg" / f"{stem}.svg", bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(OUT / "pdf" / f"{stem}.pdf", bbox_inches="tight")
    if "png" in formats:
        fig.savefig(OUT / "png" / f"{stem}.png", bbox_inches="tight", dpi=600)
    if "tiff" in formats:
        fig.savefig(OUT / "tiff" / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_fig03revised_solar() -> None:
    # ---------------------------------------------------------
    # Load the original, real trace data and event metadata.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Stage selection: rapid ramp -> persistence lag -> online recovery.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Audit trail: record exactly how each stage point was selected.
    # ---------------------------------------------------------
    DATA_OUT.mkdir(parents=True, exist_ok=True)
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

    # ---------------------------------------------------------
    # Style configuration.
    # ---------------------------------------------------------
    highlight_fc = "#E8D68A"
    guide = "#A5ADB7"
    critical_red = "#CB2D2F"
    gate_cmap = ListedColormap([COLORS["fallback"], GATE_ONLINE])

    fig = plt.figure(figsize=(7.25, 5.12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.45, 0.50, 1.8], hspace=0.35)

    # ---------------------------------------------------------
    # Panel a: macro trend with critical-ramp highlight.
    # ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, persistence, color=COLORS["persistence"], lw=1.0, ls="--", alpha=0.82, zorder=2, label="Persistence")
    ax.plot(x, online, color=COLORS["online"], lw=1.65, alpha=0.96, zorder=3, label="Online Lyra")
    observed_line, = ax.plot(x, y, color=COLORS["observed"], lw=1.02, alpha=0.98, zorder=4, label="Observed PV")
    observed_line.set_path_effects([patheffects.withStroke(linewidth=1.65, foreground="white", alpha=0.40)])

    ax.set_xlim(0, 120)
    ax.set_ylabel("PV power (MW)")
    ax.tick_params(axis="x", labelbottom=False)

    ymin, ymax = ax.get_ylim()
    rect = Rectangle(
        (event_start, ymin + 0.02 * (ymax - ymin)),
        event_end - event_start,
        0.96 * (ymax - ymin),
        fill=True,
        facecolor=highlight_fc,
        alpha=0.15,
        edgecolor=critical_red,
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(
        event_start + 0.24 * (event_end - event_start),
        ymax - 0.245 * (ymax - ymin),
        "Critical ramp",
        transform=ax.transData,
        ha="left",
        va="top",
        fontsize=7.0,
        color=critical_red,
        fontweight="bold",
    )

    base.quiet_grid(ax)
    ax.legend(loc="best")
    base.panel_label_fixed(fig, ax, "a")

    # ---------------------------------------------------------
    # Panel b: gate state.
    # ---------------------------------------------------------
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

    axg.set_xlim(0, 120)
    axg.set_ylim(0, 1)
    axg.set_xticks(np.arange(0, 121, 20))
    axg.set_xlabel("Event time (h)", labelpad=5)
    axg.set_ylabel("Gate", rotation=0, labelpad=20, va="center")
    axg.set_yticks([])
    for spine in axg.spines.values():
        spine.set_visible(False)

    base.panel_label_fixed(fig, axg, "b", dy_pt=5.0)

    # ---------------------------------------------------------
    # Panel c: micro-scale zoom with stage annotations and error arrows.
    # ---------------------------------------------------------
    axz = fig.add_subplot(gs[2, 0])
    event = case.loc[event_mask].copy()
    xe = event["case_time_h"].to_numpy(dtype=float)

    axz.axvspan(event_start, event_end, color=critical_red, alpha=0.035, lw=0, zorder=0)
    axz.plot(xe, event["persistence"], color=COLORS["persistence"], lw=1.25, ls="--", alpha=0.88, zorder=2)
    axz.plot(xe, event["online_lyra"], color=COLORS["online"], lw=2.0, alpha=0.97, zorder=3)
    zoom_obs, = axz.plot(xe, event["y_true"], color=COLORS["observed"], lw=1.25, alpha=0.98, zorder=4)
    zoom_obs.set_path_effects([patheffects.withStroke(linewidth=1.95, foreground="white", alpha=0.40)])

    zoom_values = event[["y_true", "online_lyra", "persistence"]].to_numpy(dtype=float)
    zlo, zhi = float(np.nanmin(zoom_values)), float(np.nanmax(zoom_values))
    zpad = max((zhi - zlo) * 0.11, 1.0)
    axz.set_xlim(event_start, event_end)
    axz.set_ylim(max(0.0, zlo - zpad), zhi + zpad)
    axz.set_xlabel("Event time (h)")
    axz.set_ylabel("PV power (MW)")
    base.quiet_grid(axz)
    base.panel_label_fixed(fig, axz, "c")

    # Inline legend box. Use plain "1./2./3." text instead of circled
    # Unicode glyphs (U+2460 etc.), which many fonts do not include and
    # render as empty boxes ("tofu").
    legend_text = "1. Rapid ramp\n2. Persistence lag\n3. Online recovery"
    axz.text(
        0.03,
        0.95,
        legend_text,
        transform=axz.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        linespacing=1.5,
        bbox=dict(facecolor="white", edgecolor="#D1D5DB", alpha=0.9, boxstyle="round,pad=0.5"),
        zorder=10,
    )

    stage_specs = [
        ("1", ramp_idx, y[ramp_idx], "#4F5963", 15),
        ("2", lag_idx, persistence[lag_idx], "#766E7C", -18),
        ("3", recovery_idx, online[recovery_idx], COLORS["online"], 15),
    ]
    for number, idx, target_y, color, dy in stage_specs:
        # Draw the stage marker as a plain digit inside a small circular
        # bbox instead of relying on a circled-digit Unicode glyph.
        axz.annotate(
            number,
            xy=(x[idx], target_y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=color, edgecolor=color, linewidth=0.0),
            arrowprops=dict(arrowstyle="-", color=guide, lw=0.60),
            zorder=6,
        )


    # ---------------------------------------------------------
    # Bidirectional error arrows at the persistence-lag point.
    # e_t (persistence error) and e_t' (online error) are labeled on
    # opposite sides of the arrow, with a minimum vertical separation
    # so the two labels never collide even when the two error
    # magnitudes are numerically close.
    # ---------------------------------------------------------
    err_x = x[lag_idx]
    y_per, y_obs, y_onl = persistence[lag_idx], y[lag_idx], online[lag_idx]

    axz.annotate(
        "",
        xy=(err_x, y_obs),
        xytext=(err_x, y_per),
        arrowprops=dict(arrowstyle="<->", color="#D47C25", lw=1.2),
    )
    axz.annotate(
        "",
        xy=(err_x, y_obs),
        xytext=(err_x, y_onl),
        arrowprops=dict(arrowstyle="<->", color="#728D5D", lw=1.2),
    )

    axis_span = axz.get_ylim()[1] - axz.get_ylim()[0]
    min_gap = 0.045 * axis_span

    label_y_et = (y_obs + y_per) / 2.0
    label_y_etp = (y_obs + y_onl) / 2.0
    if abs(label_y_et - label_y_etp) < min_gap:
        mid = (label_y_et + label_y_etp) / 2.0
        label_y_et = mid + min_gap / 2.0
        label_y_etp = mid - min_gap / 2.0

    x_offset_et = 0.020 * (axz.get_xlim()[1] - axz.get_xlim()[0])
    x_offset_etp = 0.055 * (axz.get_xlim()[1] - axz.get_xlim()[0])

    axz.text(
        err_x - x_offset_et,
        label_y_et,
        r"$e_t$",
        ha="right",
        va="center",
        fontsize=9,
        color="#D47C25",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
        zorder=7,
    )
    axz.text(
        err_x + x_offset_etp,
        label_y_etp,
        r"$e_t'$",
        ha="left",
        va="center",
        fontsize=9,
        color="#728D5D",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
        zorder=7,
    )

    # Cross-panel connector lines (Panel a -> Panel c).
    con1 = ConnectionPatch(
        xyA=(event_start, ymin),
        xyB=(event_start, axz.get_ylim()[1]),
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=axz,
        color=critical_red,
        ls=":",
        lw=1.2,
        alpha=0.6,
    )
    con2 = ConnectionPatch(
        xyA=(event_end, ymin),
        xyB=(event_end, axz.get_ylim()[1]),
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=axz,
        color=critical_red,
        ls=":",
        lw=1.2,
        alpha=0.6,
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    # ---------------------------------------------------------
    # Legend and save.
    # ---------------------------------------------------------
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
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.90)

    save_pub(fig, FIGURE_STEM)


def main() -> None:
    base.configure_style()
    base.ensure_dirs()

    # Ensure Fig. 3's upstream data has been generated.
    base.fig03_case_trace_revised()

    render_fig03revised_solar()
    print(f"Successfully wrote '{FIGURE_STEM}' (SVG, PDF, PNG, TIFF) and stage audit CSV to {OUT}")


if __name__ == "__main__":
    main()
