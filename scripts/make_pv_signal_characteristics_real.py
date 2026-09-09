#!/usr/bin/env python3
# ruff: noqa: E402
"""Render the PV signal-characteristics figure from measured plant data.

The observed curve is a complete 24 h record from the source workbook. The
analytical trend is not simulated: it is reconstructed from the leading
singular triplet of the mean-centered 48 x 49 Hankel trajectory matrix, using
the same rank-1 decomposition principle as Online Lyra.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "tmp" / "cache"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = (
    PROJECT_ROOT
    / "solar_stations"
    / "Solar station site 2 (Nominal capacity-130MW).xlsx"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_result_pack"
    / "figures_revised_final"
    / "real_data_pv_characteristics"
)
STEM = "fig01_pv_signal_characteristics_real"

TIME_COLUMN = "Time(year-month-day h:m:s)"
POWER_COLUMN = "Power (MW)"
GHI_COLUMN = "Global horicontal irradiance (W/m2)"
CAPACITY_MW = 130.0
SAMPLES_PER_DAY = 96
HANKEL_ROWS = 48

def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.55,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )


def load_measurements() -> pd.DataFrame:
    data = pd.read_excel(
        SOURCE,
        sheet_name="sheet1",
        usecols=[TIME_COLUMN, POWER_COLUMN, GHI_COLUMN],
    ).rename(
        columns={
            TIME_COLUMN: "timestamp",
            POWER_COLUMN: "power_mw",
            GHI_COLUMN: "ghi_wm2",
        }
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["power_mw"] = pd.to_numeric(data["power_mw"], errors="coerce")
    data["ghi_wm2"] = pd.to_numeric(data["ghi_wm2"], errors="coerce")
    if data.isna().any().any():
        raise ValueError("Source columns contain missing or invalid values.")
    data = data.sort_values("timestamp").reset_index(drop=True)
    deltas = data["timestamp"].diff().dropna().dt.total_seconds().div(60)
    if not np.allclose(deltas.to_numpy(), 15.0):
        raise ValueError("Expected a continuous 15-minute measurement series.")
    data["date"] = data["timestamp"].dt.date
    data["hour"] = (
        data["timestamp"].dt.hour
        + data["timestamp"].dt.minute / 60.0
        + data["timestamp"].dt.second / 3600.0
    )
    return data


def rank1_hankel_trend(series: np.ndarray) -> tuple[np.ndarray, float]:
    """Return rank-1 Hankel reconstruction and leading energy fraction."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or values.size != SAMPLES_PER_DAY:
        raise ValueError("A complete 96-sample day is required.")
    centered = values - values.mean()
    hankel = np.lib.stride_tricks.sliding_window_view(
        centered, HANKEL_ROWS
    ).T
    u, singular_values, vh = np.linalg.svd(hankel, full_matrices=False)
    rank1 = singular_values[0] * np.outer(u[:, 0], vh[0])

    reconstructed = np.zeros(values.size, dtype=float)
    counts = np.zeros(values.size, dtype=float)
    for row in range(rank1.shape[0]):
        width = rank1.shape[1]
        reconstructed[row : row + width] += rank1[row]
        counts[row : row + width] += 1.0
    reconstructed = reconstructed / counts + values.mean()
    energy = float(
        singular_values[0] ** 2 / np.square(singular_values).sum()
    )
    return reconstructed, energy


def candidate_table(data: pd.DataFrame) -> pd.DataFrame:
    """Score physically valid complete days for the motivation figure."""
    rows: list[dict[str, object]] = []
    for date, group in data.groupby("date", sort=True):
        if len(group) != SAMPLES_PER_DAY:
            continue
        power = group["power_mw"].to_numpy(dtype=float)
        ghi = group["ghi_wm2"].to_numpy(dtype=float)
        active = ghi > 20.0
        peak = float(power.max())

        # Exclude incomplete daylight days and physically implausible records.
        if active.sum() < 24 or peak < 35.0 or peak > 1.05 * CAPACITY_MW:
            continue

        trend, energy = rank1_hankel_trend(power)
        residual = power - trend
        correlation = float(np.corrcoef(power[active], trend[active])[0, 1])
        residual_rms = float(np.sqrt(np.mean(np.square(residual[active]))) / peak)
        ramp_q95 = float(np.quantile(np.abs(np.diff(power)), 0.95) / peak)
        trend_roughness = float(
            np.sqrt(np.mean(np.square(np.diff(trend, n=2))))
            / max(float(trend.max()), 1.0)
        )
        negative_fraction = float(np.mean(trend < 0.0))
        start_offset_fraction = float(abs(trend[0]) / peak)

        # Favor an interpretable smooth envelope with a visible, localized
        # residual and a physically plausible midnight boundary. Capping terms
        # prevents extreme anomalies from winning.
        score = (
            1.45 * correlation
            + 3.00 * min(residual_rms, 0.30)
            + 1.20 * min(ramp_q95, 0.25)
            + 0.50 * energy
            - 2.00 * trend_roughness
            - 0.80 * negative_fraction
            - 6.00 * start_offset_fraction
        )
        rows.append(
            {
                "date": str(date),
                "score": score,
                "peak_mw": peak,
                "active_samples": int(active.sum()),
                "observed_trend_correlation": correlation,
                "active_residual_rms_fraction": residual_rms,
                "ramp_q95_fraction": ramp_q95,
                "rank1_hankel_energy_fraction": energy,
                "trend_roughness": trend_roughness,
                "negative_trend_fraction": negative_fraction,
                "midnight_start_offset_fraction": start_offset_fraction,
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["score", "date"], ascending=[False, True]
    )
    if candidates.empty:
        raise ValueError("No complete physically plausible day passed screening.")
    return candidates.reset_index(drop=True)


def select_residual_window(
    hour: np.ndarray,
    residual: np.ndarray,
    ghi: np.ndarray,
    observed: np.ndarray,
    width_samples: int = 21,
) -> tuple[int, int]:
    """Select a volatile daylight window with nearly level observed endpoints."""
    best: tuple[float, int, int] | None = None
    for start in range(0, len(hour) - width_samples + 1):
        stop = start + width_samples
        if np.mean(ghi[start:stop] > 20.0) < 0.75:
            continue
        local = residual[start:stop]
        endpoint_gap = float(abs(observed[start] - observed[stop - 1]))
        if endpoint_gap > 1.5:
            continue
        total_variation = float(np.sum(np.abs(np.diff(local))))
        difference_spread = float(np.std(np.diff(local)))
        score = total_variation + 5.0 * difference_spread - 2.5 * endpoint_gap
        if best is None or score > best[0]:
            best = (score, start, stop)
    if best is None:
        raise ValueError("No valid daylight residual window was found.")
    return best[1], best[2]


def render(
    day: pd.DataFrame,
    trend: np.ndarray,
    selected_date: str,
    inset_start: int,
    inset_stop: int,
) -> plt.Figure:
    configure_style()
    hour = day["hour"].to_numpy(dtype=float)
    observed = day["power_mw"].to_numpy(dtype=float)
    # The same physical output projection used by the forecasting pipeline
    # prevents negative PV power after inverse reconstruction.
    display_trend = np.clip(trend, 0.0, CAPACITY_MW)
    residual = observed - display_trend

    color_obs_line = "#6B8E6A"
    color_obs_fill = "#D3E0D1"
    color_trend_line = "#A9A9A9"
    color_trend_fill = "#D5D7D5"

    # Match the manuscript reference canvas exactly at 200 dpi:
    # 7.175 x 3.680 in -> 1435 x 736 px.
    fig, ax = plt.subplots(figsize=(7.175, 3.680))
    ax.fill_between(
        hour, display_trend, 0,
        color=color_trend_fill, alpha=0.68, zorder=1,
    )
    ax.fill_between(
        hour, observed, 0,
        color=color_obs_fill, alpha=0.72, zorder=2,
    )
    ax.plot(
        hour, display_trend,
        color=color_trend_line, linestyle=(0, (4, 2.5)),
        linewidth=1.45, label="Analytical trend", zorder=3,
    )
    ax.plot(
        hour, observed,
        color=color_obs_line, linewidth=1.55,
        label="Observed", zorder=4,
    )

    ymax = max(float(observed.max()), float(display_trend.max())) * 1.10
    ax.set_xlim(0, 24)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("Time (hour of day)")
    ax.set_ylabel("PV power (MW)")
    ax.set_title(
        f"Measured day: {pd.Timestamp(selected_date).strftime('%d %B %Y')}",
        fontsize=7.0,
        color="#555B60",
        pad=4.0,
    )
    ax.yaxis.grid(
        True, color="#C9CED4", linewidth=0.36, alpha=0.28, zorder=0,
    )
    ax.xaxis.grid(False)
    ax.tick_params(top=False, right=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#000000")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(direction="in", width=0.45, length=3.0, color="#000000")

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.012, 0.055),
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        edgecolor="#D0D3D5",
        fontsize=6.2,
        handlelength=2.5,
        borderpad=0.45,
        labelspacing=0.35,
    )

    inset_hours = hour[inset_start:inset_stop]
    inset_residual = residual[inset_start:inset_stop]
    # The 00:00--08:00 region is empty in the main trajectory, so the inset
    # can carry a real residual trace without covering either principal line.
    axins = ax.inset_axes([0.055, 0.54, 0.29, 0.34])
    axins.axhline(0, color="#B6BABD", linewidth=0.65, zorder=1)
    axins.plot(
        inset_hours, inset_residual,
        color=color_obs_line, linewidth=1.15, zorder=2,
    )
    axins.fill_between(
        inset_hours, inset_residual, 0,
        color=color_obs_fill, alpha=0.45, zorder=1,
    )
    axins.set_xlim(inset_hours[0], inset_hours[-1])
    residual_pad = max(float(np.ptp(inset_residual)) * 0.15, 0.5)
    axins.set_ylim(
        float(inset_residual.min()) - residual_pad,
        float(inset_residual.max()) + residual_pad,
    )
    axins.set_xticks([15, 17, 19])
    residual_low = 5.0 * np.floor(inset_residual.min() / 5.0)
    residual_high = 5.0 * np.ceil(inset_residual.max() / 5.0)
    axins.set_yticks(np.linspace(residual_low, residual_high, 3))
    axins.tick_params(
        axis="both",
        labelsize=5.2,
        length=2.2,
        width=0.55,
        direction="out",
    )
    axins.set_xlabel("Time (h)", fontsize=5.5, labelpad=1.2)
    axins.set_ylabel("Residual (MW)", fontsize=5.5, labelpad=1.4)
    axins.set_title("Residual variations", fontsize=6.2, pad=2.0)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#000000")

    # The outer padding is tuned so the visible left and right whitespace are
    # balanced after accounting for the y-axis label and tick labels.
    fig.subplots_adjust(left=0.105, right=0.9325, bottom=0.135, top=0.942)
    return fig


def save_outputs(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{STEM}.pdf")
    fig.savefig(OUTPUT_DIR / f"{STEM}.svg")
    fig.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=200)
    fig.savefig(OUTPUT_DIR / f"{STEM}.tiff", dpi=600)


def main() -> None:
    data = load_measurements()
    candidates = candidate_table(data)
    selected = candidates.iloc[0]
    selected_date = str(selected["date"])
    day = data.loc[data["date"].astype(str) == selected_date].copy()
    observed = day["power_mw"].to_numpy(dtype=float)
    raw_trend, energy = rank1_hankel_trend(observed)
    trend = np.clip(raw_trend, 0.0, CAPACITY_MW)
    residual = observed - trend
    inset_start, inset_stop = select_residual_window(
        day["hour"].to_numpy(dtype=float),
        residual,
        day["ghi_wm2"].to_numpy(dtype=float),
        observed,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export = day[["timestamp", "hour", "power_mw", "ghi_wm2"]].copy()
    export["raw_rank1_hankel_trend_mw"] = raw_trend
    export["analytical_trend_mw"] = trend
    export["residual_mw"] = residual
    export.to_csv(OUTPUT_DIR / f"{STEM}_source_data.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / f"{STEM}_candidate_audit.csv", index=False)

    selection = {
        "selected_date": selected_date,
        "source_rows_total": int(len(data)),
        "source_start": str(data["timestamp"].min()),
        "source_end": str(data["timestamp"].max()),
        "sampling_interval_minutes": 15,
        "selected_rows": int(len(day)),
        "capacity_mw": CAPACITY_MW,
        "hankel_rows": HANKEL_ROWS,
        "hankel_columns": SAMPLES_PER_DAY - HANKEL_ROWS + 1,
        "rank": 1,
        "physical_projection": "[0, 130 MW]",
        "rank1_hankel_energy_fraction": energy,
        "inset_start_hour": float(day["hour"].iloc[inset_start]),
        "inset_end_hour": float(day["hour"].iloc[inset_stop - 1]),
        "selection_metrics": {
            key: float(selected[key])
            for key in [
                "score",
                "peak_mw",
                "observed_trend_correlation",
                "active_residual_rms_fraction",
                "ramp_q95_fraction",
                "trend_roughness",
                "negative_trend_fraction",
                "midnight_start_offset_fraction",
            ]
        },
    }
    (OUTPUT_DIR / f"{STEM}_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig = render(day, trend, selected_date, inset_start, inset_stop)
    save_outputs(fig)
    plt.close(fig)
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    print(f"Wrote real-data figure bundle to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
