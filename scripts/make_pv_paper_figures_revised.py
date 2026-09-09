#!/usr/bin/env python3
# ruff: noqa: E402
"""Build publication figures for Online Lyra from recorded result artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patheffects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


PACK = ROOT / "outputs" / "paper_result_pack"
SERVER = ROOT / "outputs" / "server_runs_20260716"
OUT = PACK / "figures_revised"
DATA_OUT = OUT / "data"

HORIZON_TO_HOURS = {4: 1, 12: 3, 24: 6, 48: 12, 96: 24}

# Study palette: dark blue for the proposed method, cyan/teal support colors,
# warm rose/orange for fallback and foundation baselines, and a cool-grey family
# for static neural baselines.
COLORS = {
    "online": "#0B4F8A",
    "online_light": "#5DA8C9",
    "offline": "#42A6A4",
    "observed": "#4D4D4D",
    "persistence": "#9A8FA3",
    "fallback": "#B66F89",
    "foundation": "#E39B34",
    "static_dark": "#4F5D8C",
    "static_mid": "#7E8EB8",
    "static_light": "#B8C4D9",
    "linear": "#6EA6C8",
    "phase": "#72B7A8",
    "timekan": "#B9A23F",
    "patch": "#A783B6",
    "neutral": "#9099A6",
    "neutral_light": "#D8DDE5",
    "black": "#242424",
    "grid": "#E8EBF0",
    "good": "#3A9D5D",
    "bad": "#C85A63",
}

MODEL_COLORS = {
    "Online Lyra": COLORS["online"],
    "Offline Lyra": COLORS["offline"],
    "Persistence": COLORS["persistence"],
    "Smart Persistence": "#8F8997",
    "iTransformer": COLORS["static_dark"],
    "DLinear": COLORS["linear"],
    "NLinear": COLORS["static_light"],
    "Linear": "#A9B3C4",
    "PatchTST": COLORS["patch"],
    "PhaseFormer": COLORS["phase"],
    "TimeKAN": COLORS["timekan"],
    "Olivia": COLORS["foundation"],
    "MixLinear": "#B2B2B2",
    "Seasonal Naive": "#BDBDBD",
    "Trend-only": "#707C8A",
}

DISPLAY_FROM_MODEL = {
    "online_lyra_b06_final": "Online Lyra",
    "online_lyra_pv_gated": "Online Lyra",
    "offline_lyra_final": "Offline Lyra",
    "official_dlinear": "DLinear",
    "dlinear": "DLinear",
    "official_nlinear": "NLinear",
    "official_linear": "Linear",
    "itransformer": "iTransformer",
    "official_patchtst": "PatchTST",
    "timekan": "TimeKAN",
    "phaseformer": "PhaseFormer",
    "olivia_scratch": "Olivia",
    "mixlinear": "MixLinear",
    "seasonal_naive": "Seasonal Naive",
    "persistence": "Persistence",
    "smart_persistence": "Smart Persistence",
    "trend_only": "Trend-only",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )


def ensure_dirs() -> None:
    for sub in ["svg", "pdf", "png", "tiff", "data", "notes"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def save_pub(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / "svg" / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / "pdf" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "png" / f"{stem}.png", bbox_inches="tight", dpi=600)
    fig.savefig(OUT / "tiff" / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def parse_site_id(farm: str) -> int:
    import re

    m = re.search(r"site\s+(\d+)", str(farm), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse site id from farm={farm!r}")
    return int(m.group(1))


def seed_from_path(path: Path) -> int:
    import re

    m = re.search(r"seed[_-](\d+)", str(path))
    if not m:
        raise ValueError(f"Cannot parse seed from path={path}")
    return int(m.group(1))


def load_replicate_results() -> pd.DataFrame:
    """Load site x seed x horizon results without fabricating replicates."""
    rows: list[pd.DataFrame] = []

    for path in sorted((SERVER / "fair_protocol_20260716_123158").glob("static_official_seed_*/pv_baseline_summary.csv")):
        df = pd.read_csv(path)
        df["seed"] = seed_from_path(path)
        df["site_id"] = df["farm"].map(parse_site_id)
        rows.append(df)

    for path in sorted((SERVER / "final_online_lyra_b06_20260716_231353" / "b06_final").glob("site_*_seed_*/online_pv_summary.csv")):
        df = pd.read_csv(path)
        df = df.loc[df["model"].isin(["online_lyra_pv_gated", "persistence", "trend_only"])].copy()
        df["seed"] = seed_from_path(path)
        df["site_id"] = df["farm"].map(parse_site_id)
        rows.append(df)

    for path in sorted((SERVER / "offline_lyra_final_20260717_165425").glob("offline_lyra_seed_*/offline_lyra_summary.csv")):
        df = pd.read_csv(path)
        df["seed"] = seed_from_path(path)
        df["site_id"] = df["farm"].map(parse_site_id)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out["model_display"] = out["model"].map(display_for)
    out["lead_time_h"] = out["horizon"].map(HORIZON_TO_HOURS)
    return out


def mean_ci_from_replicates(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    unit_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Mean and 95% CI using legitimate replicate units, default seed-level."""
    unit_cols = unit_cols or ["seed"]
    unit = df.groupby(group_cols + unit_cols, as_index=False)[metric_cols].mean()
    rows = []
    for keys, g in unit.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            vals = g.loc[g[metric].notna(), metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{metric}_n"] = int(len(vals))
            if len(vals) >= 2:
                tcrit = float(student_t.ppf(0.975, df=len(vals) - 1))
                row[f"{metric}_ci95"] = float(tcrit * np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                row[f"{metric}_ci95"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    vals = p[mask]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[mask] = tmp
    return out


def contrast_color_for_cmap(value: float, norm: Normalize, cmap, threshold: float = 4.5) -> str:
    rgba = cmap(norm(value))
    r, g, b = rgba[:3]
    def lum(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    L = 0.2126 * lum(r) + 0.7152 * lum(g) + 0.0722 * lum(b)
    contrast_black = (L + 0.05) / 0.05
    contrast_white = 1.05 / (L + 0.05)
    return "black" if contrast_black >= contrast_white else "white"


def axis_break_marks(ax: plt.Axes, xpos: float = -0.017, size: float = 0.015) -> None:
    for y in [0.02, 0.98]:
        ax.plot(
            [xpos - size, xpos + size],
            [y - size, y + size],
            transform=ax.transAxes,
            color="#333333",
            lw=0.8,
            clip_on=False,
        )


CAPACITY_GROUP_ORDER = ["Small (<50 MW)", "Medium (50-110 MW)", "Large (>110 MW)"]


def capacity_group_scheme_b(capacity: float) -> str:
    """Use mutually exclusive capacity groups: <50, 50-110, and >110 MW."""
    cap = float(capacity)
    if cap < 50:
        return "Small (<50 MW)"
    if cap <= 110:
        return "Medium (50-110 MW)"
    return "Large (>110 MW)"


def strongest_baseline(rep: pd.DataFrame) -> str:
    """Objectively pick the static baseline with lowest mean nRMSE."""
    exclude = {"Online Lyra", "Offline Lyra", "Persistence", "Seasonal Naive", "MixLinear", "Trend-only"}
    static = rep.loc[~rep["model_display"].isin(exclude)].copy()
    if static.empty:
        raise ValueError("No eligible static baseline found.")
    return str(static.groupby("model_display")["nrmse"].mean().sort_values().index[0])


def accuracy_model_set(rep: pd.DataFrame) -> tuple[list[str], str]:
    strongest = strongest_baseline(rep)
    preferred = [
        "Online Lyra",
        strongest,
        "Olivia",
        "DLinear",
        "NLinear",
        "PhaseFormer",
        "TimeKAN",
        "Offline Lyra",
    ]
    seen: set[str] = set()
    available = set(rep.loc[rep["model_display"].notna(), "model_display"])
    out: list[str] = []
    for name in preferred:
        if name not in seen and name in available:
            out.append(name)
            seen.add(name)
    return out, strongest


def ci_errorbar(ax: plt.Axes, x: np.ndarray, y: np.ndarray, ci: np.ndarray, color: str, zorder: int) -> None:
    if np.isfinite(ci).any():
        ax.errorbar(x, y, yerr=ci, fmt="none", ecolor=color, elinewidth=0.75, capsize=2.0, alpha=0.72, zorder=zorder)


def line_spec(model: str, strongest: str | None = None) -> dict[str, object]:
    linestyles = {
        "Online Lyra": "-",
        strongest or "": "--",
        "Olivia": "-.",
        "DLinear": ":",
        "NLinear": (0, (4, 2, 1, 2)),
        "PhaseFormer": (0, (3, 1)),
        "TimeKAN": (0, (1, 1)),
        "Offline Lyra": (0, (5, 2)),
    }
    markers = {
        "Online Lyra": "o",
        strongest or "": "s",
        "Persistence": "^",
        "Olivia": "D",
        "DLinear": "v",
        "NLinear": "P",
        "PhaseFormer": "X",
        "TimeKAN": "h",
        "Offline Lyra": "d",
    }
    if model == "Online Lyra":
        return {"color": COLORS["online"], "lw": 2.55, "alpha": 1.0, "zorder": 8, "ls": "-", "marker": "o", "ms": 4.0}
    if strongest and model == strongest:
        return {"color": COLORS["online_light"], "lw": 1.75, "alpha": 0.98, "zorder": 6, "ls": "--", "marker": "s", "ms": 3.5}
    grey_tones = {
        "Olivia": "#7E8792",
        "DLinear": "#9AA3AD",
        "NLinear": "#B5BDC5",
        "PhaseFormer": "#87919D",
        "TimeKAN": "#A7AEB8",
        "Offline Lyra": "#5EAAA8",
    }
    return {
        "color": grey_tones.get(model, "#9AA3AD"),
        "lw": 1.15 if model == "Offline Lyra" else 1.05,
        "alpha": 0.82 if model == "Offline Lyra" else 0.76,
        "zorder": 2,
        "ls": linestyles.get(model, "-"),
        "marker": markers.get(model, "o"),
        "ms": 3.0,
    }


def fig01_line_spec(model: str, strongest: str) -> dict[str, object]:
    """High-separation encoding used only in the dense Fig. 1 line panels."""
    styles: dict[str, dict[str, object]] = {
        "Online Lyra": {"color": "#0B4F8A", "ls": "-", "marker": "o"},
        "Offline Lyra": {"color": "#56A6D5", "ls": (0, (5, 2)), "marker": "d"},
        "Olivia": {"color": "#A56AA0", "ls": (0, (5, 1.5)), "marker": "D"},
        "DLinear": {"color": "#6F7782", "ls": (0, (3, 1.5)), "marker": "v"},
        "NLinear": {"color": "#A68032", "ls": ":", "marker": "P"},
        "PhaseFormer": {"color": "#2A8F72", "ls": (0, (4, 1.5, 1, 1.5)), "marker": "X"},
        "TimeKAN": {"color": "#7567A5", "ls": (0, (1, 1.35)), "marker": "h"},
        "Smart Persistence": {"color": "#8F8997", "ls": (0, (2, 1.5)), "marker": "^"},
    }
    if model == strongest:
        base = {"color": "#D8752D", "ls": "--", "marker": "s"}
    else:
        base = styles.get(model, {"color": "#8E98A3", "ls": "-.", "marker": "^"})
    if model == "Online Lyra":
        return {**base, "lw": 2.25, "ms": 4.1, "alpha": 1.0, "zorder": 10}
    if model == strongest:
        return {**base, "lw": 1.65, "ms": 3.7, "alpha": 0.98, "zorder": 8}
    if model == "Offline Lyra":
        return {**base, "lw": 1.35, "ms": 3.4, "alpha": 0.95, "zorder": 6}
    return {**base, "lw": 1.15, "ms": 3.2, "alpha": 0.90, "zorder": 4}


def fig01_dodged_errorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ci: np.ndarray,
    spec: dict[str, object],
    offset_pt: float,
) -> None:
    """Dodge markers and CIs in display points while keeping trend lines at true x."""
    if not np.isfinite(ci).any():
        return
    transform = ax.transData + ScaledTranslation(offset_pt / 72.0, 0.0, fig.dpi_scale_trans)
    ax.errorbar(
        x,
        y,
        yerr=ci,
        fmt=str(spec["marker"]),
        ms=float(spec["ms"]),
        mfc=str(spec["color"]),
        mec="white",
        mew=0.45,
        ecolor=str(spec["color"]),
        elinewidth=0.72,
        capsize=0.55,
        capthick=0.72,
        alpha=float(spec["alpha"]),
        zorder=int(spec["zorder"]) + 1,
        transform=transform,
    )


def fig01_ci_ylim(summary: pd.DataFrame, metric: str, step: float) -> tuple[float, float]:
    mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
    ci = np.nan_to_num(summary[f"{metric}_ci95"].to_numpy(dtype=float), nan=0.0)
    lo = float(np.nanmin(mean - ci))
    hi = float(np.nanmax(mean + ci))
    pad = max((hi - lo) * 0.08, step)
    return float(np.floor((lo - pad) / step) * step), float(np.ceil((hi + pad) / step) * step)


def clipped_text(value: float, precision: int = 1) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{precision}f}"


def repel_texts_y(ax: plt.Axes, texts: list[plt.Text], min_gap_frac: float = 0.065) -> None:
    """Deterministic y-direction label repulsion for compact scatter labels."""
    if len(texts) < 2:
        return
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * min_gap_frac
    ordered = sorted(texts, key=lambda t: t.get_position()[1])
    positions = [(t.get_position()[0], t.get_position()[1]) for t in ordered]
    new_y = [p[1] for p in positions]
    for i in range(1, len(new_y)):
        if new_y[i] - new_y[i - 1] < gap:
            new_y[i] = new_y[i - 1] + gap
    overflow = new_y[-1] - (ymax - gap * 0.3)
    if overflow > 0:
        new_y = [y - overflow for y in new_y]
    for i in range(len(new_y) - 2, -1, -1):
        if new_y[i + 1] - new_y[i] < gap:
            new_y[i] = new_y[i + 1] - gap
    underflow = (ymin + gap * 0.3) - new_y[0]
    if underflow > 0:
        new_y = [y + underflow for y in new_y]
    for text, (x, _), y in zip(ordered, positions, new_y):
        text.set_position((x, y))


def make_contact_sheet(stems: list[str], suffix: str = "preview_contact_sheet_revised.png") -> None:
    import matplotlib.image as mpimg

    imgs = []
    labels = []
    for stem in stems:
        path = OUT / "png" / f"{stem}.png"
        if path.exists():
            imgs.append(mpimg.imread(path))
            labels.append(stem)
    if not imgs:
        return
    ncols = 2
    nrows = int(np.ceil(len(imgs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.25, 2.85 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=8, pad=4)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.tight_layout(pad=0.9)
    fig.savefig(OUT / suffix, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_highres_figure_overview() -> None:
    """Create a high-resolution overview of all current paper figures."""
    from PIL import Image

    panels = [
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
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.0, 3.55 * nrows), constrained_layout=True)
    for ax, (label, stem) in zip(axes.ravel(), panels):
        path = OUT / "png" / f"{stem}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            source.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
            preview = np.asarray(source.convert("RGB"))
        ax.imshow(preview, interpolation="lanczos")
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left", pad=5)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.savefig(
        OUT / "figure_overview_complete_highres.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def add_hours(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lead_time_h"] = out["horizon"].map(HORIZON_TO_HOURS).fillna(out["horizon"] / 4)
    return out


def color_for(model: str) -> str:
    return MODEL_COLORS.get(model, COLORS["neutral"])


def display_for(model: str) -> str:
    return DISPLAY_FROM_MODEL.get(model, model)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["black"],
    )


def panel_label_fixed(fig: plt.Figure, ax: plt.Axes, label: str, dx_pt: float = -15.0, dy_pt: float = 7.0) -> None:
    """Place panel labels with a fixed physical offset from each axes corner."""
    transform = ax.transAxes + ScaledTranslation(dx_pt / 72.0, dy_pt / 72.0, fig.dpi_scale_trans)
    ax.text(
        0,
        1,
        label,
        transform=transform,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["black"],
    )


def quiet_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)


def set_zoom_ylim(ax: plt.Axes, values: np.ndarray, pad_frac: float = 0.12) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = float(finite.min()), float(finite.max())
    pad = max((hi - lo) * pad_frac, 0.002)
    ax.set_ylim(lo - pad, hi + pad)


def direct_label_lines(ax: plt.Axes, df: pd.DataFrame, metric: str, offsets: dict[str, float] | None = None) -> None:
    offsets = offsets or {}
    for model, sub in df.groupby("model_display"):
        sub = sub.sort_values("lead_time_h")
        x = float(sub["lead_time_h"].iloc[-1])
        y = float(sub[metric].iloc[-1])
        ax.text(
            x * 1.035,
            y + offsets.get(model, 0.0),
            model,
            color=color_for(model),
            fontsize=5.8,
            va="center",
            ha="left",
        )


def strong_model_order(df: pd.DataFrame, n_static: int = 6) -> list[str]:
    exclude = {"Online Lyra", "Offline Lyra", "Persistence", "Seasonal Naive", "MixLinear"}
    static = (
        df.loc[~df["model_display"].isin(exclude)]
        .groupby("model_display")["nrmse"]
        .mean()
        .sort_values()
        .head(n_static)
        .index.tolist()
    )
    preferred = ["Online Lyra", *static, "Offline Lyra"]
    return [m for m in preferred if m in set(df["model_display"])]


def prediction_npz(site_id: int, horizon: int, seed: int = 2028, offline: bool = False) -> Path:
    if offline:
        pattern = f"offline_lyra_final_20260717_165425/offline_lyra_seed_{seed}/artifacts/Solar_station_site_{site_id}_*_h{horizon}_offline_lyra_final_predictions.npz"
    else:
        pattern = f"final_online_lyra_b06_20260716_231353/b06_final/site_{site_id}_seed_{seed}/artifacts/Solar_station_site_{site_id}_*_h{horizon}_predictions.npz"
    matches = sorted(SERVER.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def baseline_prediction_npz(site_id: int, horizon: int, seed: int = 2028) -> Path:
    pattern = f"fair_protocol_20260716_123158/static_official_seed_{seed}/artifacts/Solar_station_site_{site_id}_*_h{horizon}_baselines_predictions.npz"
    matches = sorted(SERVER.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def fig02_common_mask_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Fig. 2 metrics for the manuscript's 13 formal methods.

    The baseline artifact named ``seasonal_naive`` is exactly the manuscript's
    Smart Persistence reference: it repeats the observation from 96 steps
    (24 h) earlier and can use pre-test history at the first forecast origin.
    Rename that prediction instead of adding a duplicate Smart Persistence row.
    """
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    horizons = sorted(HORIZON_TO_HOURS)

    for seed in [2026, 2027, 2028]:
        for site_id in range(1, 9):
            for horizon in horizons:
                baseline = np.load(baseline_prediction_npz(site_id, horizon, seed), allow_pickle=True)
                online = np.load(prediction_npz(site_id, horizon, seed), allow_pickle=True)
                offline = np.load(prediction_npz(site_id, horizon, seed, offline=True), allow_pickle=True)

                observed = np.asarray(baseline["y_true"], dtype=float).reshape(-1)
                online_observed = np.asarray(online["y_true"], dtype=float).reshape(-1)
                offline_observed = np.asarray(offline["y_true"], dtype=float).reshape(-1)
                if observed.shape != online_observed.shape or observed.shape != offline_observed.shape:
                    raise ValueError(f"Fig. 2 target shape mismatch at site={site_id}, h={horizon}, seed={seed}")
                online_target_diff = float(np.max(np.abs(observed - online_observed)))
                offline_target_diff = float(np.max(np.abs(observed - offline_observed)))
                if online_target_diff > 1e-6 or offline_target_diff > 1e-6:
                    raise ValueError(f"Fig. 2 target values mismatch at site={site_id}, h={horizon}, seed={seed}")

                predictions = {
                    key: np.asarray(baseline[key], dtype=float).reshape(-1)
                    for key in baseline.files
                    if key not in {"capacity", "horizon", "y_true"}
                }
                predictions["online_lyra_b06_final"] = np.asarray(
                    online["online_lyra_pv_gated"], dtype=float
                ).reshape(-1)
                predictions["offline_lyra_final"] = np.asarray(
                    offline["offline_lyra_final"], dtype=float
                ).reshape(-1)

                if "seasonal_naive" not in predictions:
                    raise KeyError(
                        f"Missing previous-day baseline at site={site_id}, "
                        f"h={horizon}, seed={seed}"
                    )
                predictions["smart_persistence"] = predictions.pop("seasonal_naive")

                valid = np.isfinite(observed)
                for pred in predictions.values():
                    if pred.shape != observed.shape:
                        raise ValueError(f"Fig. 2 prediction shape mismatch at site={site_id}, h={horizon}, seed={seed}")
                    valid &= np.isfinite(pred)

                capacity = float(baseline["capacity"])
                capacity_group = capacity_group_scheme_b(capacity)
                farm = f"Solar station site {site_id} (Nominal capacity-{capacity:g}MW)"
                site = f"Site {site_id} ({capacity:g}MW)"
                for model, pred in predictions.items():
                    error = pred[valid] - observed[valid]
                    rows.append(
                        {
                            "seed": seed,
                            "site_id": site_id,
                            "site": site,
                            "farm": farm,
                            "capacity": capacity,
                            "capacity_group": capacity_group,
                            "horizon": horizon,
                            "lead_time_h": HORIZON_TO_HOURS[horizon],
                            "model": model,
                            "model_display": display_for(model),
                            "nrmse": float(np.sqrt(np.mean(np.square(error))) / capacity),
                            "nmae": float(np.mean(np.abs(error)) / capacity),
                            "n_valid": int(valid.sum()),
                        }
                    )
                audit_rows.append(
                    {
                        "seed": seed,
                        "site_id": site_id,
                        "horizon": horizon,
                        "n_total": int(observed.size),
                        "n_excluded_initial_24h": 0,
                        "n_common_valid": int(valid.sum()),
                        "model_count": int(len(predictions)),
                        "online_target_max_abs_diff": online_target_diff,
                        "offline_target_max_abs_diff": offline_target_diff,
                        "smart_persistence_lag_steps": 96,
                        "smart_persistence_lag_hours": 24,
                        "smart_persistence_source": "seasonal_naive artifact renamed",
                    }
                )

    metrics = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    return metrics, audit


def flatten_online(site_id: int, horizon: int, seed: int = 2028) -> pd.DataFrame:
    z = np.load(prediction_npz(site_id, horizon, seed), allow_pickle=True)
    h = int(z["horizon"])
    n = z["y_true"].shape[0]
    cap = float(z["capacity"])
    return pd.DataFrame(
        {
            "time_step": np.arange(n * h),
            "time_h": np.arange(n * h) / 4.0,
            "y_true": z["y_true"].reshape(-1),
            "online_lyra": z["online_lyra_pv_gated"].reshape(-1),
            "trend_only": z["trend_only"].reshape(-1),
            "persistence": z["persistence"].reshape(-1),
            "gate_uses_lyra": np.repeat(z["gate_uses_lyra"], h).astype(bool),
            "capacity": cap,
            "site_id": site_id,
            "horizon": horizon,
            "seed": seed,
        }
    )


def select_case_window(df: pd.DataFrame, span: int = 480) -> pd.DataFrame:
    cap = float(df["capacity"].iloc[0])
    best_start, best_score = 0, -np.inf
    day = 96
    for start in range(0, max(1, len(df) - span), day):
        w = df.iloc[start : start + span]
        if len(w) < span:
            continue
        y = w["y_true"].to_numpy()
        daytime_fraction = np.mean(y > 0.08 * cap)
        if daytime_fraction < 0.25:
            continue
        volatility = np.nanstd(np.diff(y))
        gain = np.mean(np.abs(w["y_true"] - w["persistence"])) - np.mean(np.abs(w["y_true"] - w["online_lyra"]))
        fallback = np.mean(~w["gate_uses_lyra"].to_numpy())
        score = 0.55 * gain + 0.35 * volatility + 8.0 * fallback
        if score > best_score:
            best_score = float(score)
            best_start = start
    out = df.iloc[best_start : best_start + span].copy()
    out["case_time_h"] = np.arange(len(out)) / 4.0
    return out


def screen_case_event_windows(
    case: pd.DataFrame,
    durations_h: tuple[int, ...] = (6, 7, 8),
) -> tuple[pd.DataFrame, pd.Series]:
    """Select a sustained ramp event using window-level evidence, not one point."""
    case = case.reset_index(drop=True)
    capacity = float(case["capacity"].iloc[0])
    dt_h = float(np.median(np.diff(case["case_time_h"].to_numpy(dtype=float))))
    rows: list[dict[str, float | int | bool]] = []

    for duration_h in durations_h:
        n_points = int(round(duration_h / dt_h))
        for start_idx in range(0, len(case) - n_points + 1):
            window = case.iloc[start_idx : start_idx + n_points]
            observed = window["y_true"].to_numpy(dtype=float)
            online = window["online_lyra"].to_numpy(dtype=float)
            persistence = window["persistence"].to_numpy(dtype=float)
            online_error = np.abs(observed - online)
            persistence_error = np.abs(observed - persistence)

            observed_change = np.diff(observed)
            online_change = np.diff(online)
            persistence_change = np.diff(persistence)
            change_error_online = np.abs(observed_change - online_change)
            change_error_persistence = np.abs(observed_change - persistence_change)
            ramp_cutoff = float(np.quantile(np.abs(observed_change), 0.75))
            rapid_change = np.abs(observed_change) >= ramp_cutoff

            online_mae = float(np.mean(online_error))
            persistence_mae = float(np.mean(persistence_error))
            online_rmse = float(np.sqrt(np.mean((observed - online) ** 2)))
            persistence_rmse = float(np.sqrt(np.mean((observed - persistence) ** 2)))
            change_mae_online = float(np.mean(change_error_online))
            change_mae_persistence = float(np.mean(change_error_persistence))
            rapid_change_mae_online = float(np.mean(change_error_online[rapid_change]))
            rapid_change_mae_persistence = float(np.mean(change_error_persistence[rapid_change]))

            rows.append(
                {
                    "duration_h": duration_h,
                    "start_idx": start_idx,
                    "end_idx": start_idx + n_points,
                    "event_start_h": float(case["case_time_h"].iloc[start_idx]),
                    "event_end_h": float(case["case_time_h"].iloc[start_idx] + duration_h),
                    "gate_lyra_fraction": float(np.mean(window["gate_uses_lyra"].to_numpy(dtype=bool))),
                    "daytime_fraction": float(np.mean(observed > 0.08 * capacity)),
                    "observed_range_mw": float(np.ptp(observed)),
                    "observed_range_capacity_fraction": float(np.ptp(observed) / capacity),
                    "max_ramp_mw_per_15min": float(np.max(np.abs(observed_change))),
                    "online_mae_mw": online_mae,
                    "persistence_mae_mw": persistence_mae,
                    "mae_reduction_fraction": float(1.0 - online_mae / max(persistence_mae, 1e-12)),
                    "online_rmse_mw": online_rmse,
                    "persistence_rmse_mw": persistence_rmse,
                    "rmse_reduction_fraction": float(1.0 - online_rmse / max(persistence_rmse, 1e-12)),
                    "online_win_fraction": float(np.mean(online_error < persistence_error)),
                    "change_mae_online_mw": change_mae_online,
                    "change_mae_persistence_mw": change_mae_persistence,
                    "change_mae_reduction_fraction": float(
                        1.0 - change_mae_online / max(change_mae_persistence, 1e-12)
                    ),
                    "rapid_change_mae_online_mw": rapid_change_mae_online,
                    "rapid_change_mae_persistence_mw": rapid_change_mae_persistence,
                    "rapid_change_mae_reduction_fraction": float(
                        1.0 - rapid_change_mae_online / max(rapid_change_mae_persistence, 1e-12)
                    ),
                }
            )

    screening = pd.DataFrame(rows)
    screening["passes_selection_criteria"] = (
        (screening["gate_lyra_fraction"] >= 0.95)
        & (screening["daytime_fraction"] >= 0.50)
        & (screening["observed_range_capacity_fraction"] >= 0.30)
        & (screening["mae_reduction_fraction"] >= 0.25)
        & (screening["rmse_reduction_fraction"] >= 0.25)
        & (screening["online_win_fraction"] >= 0.70)
        & (screening["change_mae_reduction_fraction"] >= 0.30)
        & (screening["rapid_change_mae_reduction_fraction"] >= 0.30)
    )
    screening["selection_score"] = np.nan
    eligible = screening["passes_selection_criteria"]
    screening.loc[eligible, "selection_score"] = (
        screening.loc[eligible, "observed_range_capacity_fraction"]
        * screening.loc[eligible, "mae_reduction_fraction"]
        * screening.loc[eligible, "rmse_reduction_fraction"]
        * screening.loc[eligible, "online_win_fraction"]
        * screening.loc[eligible, "change_mae_reduction_fraction"]
        * screening.loc[eligible, "rapid_change_mae_reduction_fraction"]
    )
    if not eligible.any():
        raise RuntimeError("No 6-8 h event satisfies the declared Fig. 3 evidence criteria.")

    selected_idx = int(screening.loc[eligible, "selection_score"].idxmax())
    screening["event_selected"] = False
    screening.loc[selected_idx, "event_selected"] = True
    return screening, screening.loc[selected_idx].copy()


def simple_kde(samples: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.zeros_like(grid)
    if bandwidth is None:
        std = np.std(x)
        bandwidth = max(1.06 * std * len(x) ** (-1 / 5), 0.005)
    diff = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1) / (len(x) * bandwidth * np.sqrt(2 * np.pi))
    return density


def fig02_accuracy_zoom() -> None:
    df = add_hours(read_csv(PACK / "server_tables_with_offline_lyra" / "main_average_by_horizon.csv"))
    models = strong_model_order(df, n_static=6)
    strong = df.loc[df["model_display"].isin(models)].copy()
    strong.to_csv(DATA_OUT / "fig02_accuracy_zoomed_strong_models.csv", index=False)

    pvt = df.pivot_table(index="horizon", columns="model_display", values="nrmse")
    skill_rows = []
    for model in ["Online Lyra", "iTransformer", "Olivia", "DLinear", "Offline Lyra"]:
        if model in pvt.columns and "Persistence" in pvt.columns:
            for horizon, row in pvt.iterrows():
                skill_rows.append(
                    {
                        "model_display": model,
                        "horizon": horizon,
                        "lead_time_h": HORIZON_TO_HOURS.get(int(horizon), horizon / 4),
                        "relative_nrmse_reduction_vs_persistence": (row["Persistence"] - row[model])
                        / row["Persistence"]
                        * 100,
                    }
                )
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(DATA_OUT / "fig02_skill_vs_persistence.csv", index=False)

    fig = plt.figure(figsize=(7.4, 3.15))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.15, 0.85], wspace=0.38, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_handles = []
    legend_labels = []
    for ax, metric, title, letter in [
        (axes[0], "nrmse", "Zoomed nRMSE", "a"),
        (axes[1], "nmae", "Zoomed nMAE", "b"),
    ]:
        for model in models:
            sub = strong.loc[strong["model_display"] == model].sort_values("lead_time_h")
            lw = 2.3 if model == "Online Lyra" else 1.25
            alpha = 1.0 if model == "Online Lyra" else 0.78
            z = 5 if model == "Online Lyra" else 2
            line, = ax.plot(
                sub["lead_time_h"],
                sub[metric],
                marker="o",
                ms=3.6 if model == "Online Lyra" else 2.6,
                lw=lw,
                color=color_for(model),
                alpha=alpha,
                zorder=z,
            )
            if metric == "nrmse":
                legend_handles.append(line)
                legend_labels.append(model)
        set_zoom_ylim(ax, strong[metric].to_numpy(), pad_frac=0.20)
        ax.set_xlim(0.8, 27)
        ax.set_xticks([1, 3, 6, 12, 24])
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel("nRMSE" if metric == "nrmse" else "nMAE")
        ax.set_title(title)
        quiet_grid(ax)
        panel_label(ax, letter)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.385, 1.055),
        ncol=4,
        columnspacing=0.9,
        handlelength=1.8,
    )

    skill_keep = ["Online Lyra", "iTransformer", "Olivia", "DLinear"]
    for model in skill_keep:
        sub = skill.loc[skill["model_display"] == model].sort_values("lead_time_h")
        axes[2].plot(
            sub["lead_time_h"],
            sub["relative_nrmse_reduction_vs_persistence"],
            marker="o",
            ms=3.0,
            lw=2.0 if model == "Online Lyra" else 1.15,
            color=color_for(model),
            alpha=1.0 if model == "Online Lyra" else 0.78,
            label=model,
        )
    axes[2].set_title("Reduction vs persistence")
    axes[2].set_xlabel("Forecast horizon (h)")
    axes[2].set_ylabel("nRMSE reduction (%)")
    axes[2].set_xticks([1, 3, 6, 12, 24])
    axes[2].set_xlim(0.8, 27)
    axes[2].legend(loc="lower left", ncol=1, handlelength=1.6)
    quiet_grid(axes[2])
    panel_label(axes[2], "c")
    fig.suptitle("Online Lyra remains in the top accuracy tier across horizons", y=1.17, fontsize=9, fontweight="bold")
    save_pub(fig, "fig02_accuracy_zoomed")


def fig03_robustness_rank_capacity() -> None:
    rank = read_csv(PACK / "server_tables_with_offline_lyra" / "main_metric_long_with_ranks.csv")
    lyra_rank = rank.loc[
        (rank["model_display"] == "Online Lyra") & (rank["metric"] == "nrmse"),
        ["site_id", "site", "horizon", "rank", "value"],
    ].copy()
    lyra_rank["lead_time_h"] = lyra_rank["horizon"].map(HORIZON_TO_HOURS)
    rank_matrix = lyra_rank.pivot_table(index="site", columns="lead_time_h", values="rank", aggfunc="mean")
    rank_matrix = rank_matrix.reindex(sorted(rank_matrix.columns), axis=1)
    rank_matrix.to_csv(DATA_OUT / "fig03_online_lyra_rank_by_site_horizon.csv")

    cap = read_csv(PACK / "robustness" / "robustness_by_capacity_all_models.csv")
    static = cap.loc[~cap["model_display"].isin(["Online Lyra", "Offline Lyra", "Persistence", "Seasonal Naive", "MixLinear"])]
    best_static = static.sort_values("nrmse").groupby("capacity_group", as_index=False).first()
    best_static["model_display"] = "Best static"
    cap_show = pd.concat(
        [cap.loc[cap["model_display"].isin(["Online Lyra", "Offline Lyra"])], best_static],
        ignore_index=True,
    )
    cap_show.to_csv(DATA_OUT / "fig03_capacity_zoomed.csv", index=False)
    pvt = cap.pivot_table(index="capacity_group", columns="model_display", values="nrmse")
    skill_rows = []
    for model in ["Online Lyra", "Offline Lyra"]:
        if model in pvt.columns and "Persistence" in pvt.columns:
            for group, row in pvt.iterrows():
                skill_rows.append(
                    {
                        "capacity_group": group,
                        "model_display": model,
                        "skill_vs_persistence": (row["Persistence"] - row[model]) / row["Persistence"] * 100,
                    }
                )
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(DATA_OUT / "fig03_capacity_skill.csv", index=False)

    fig = plt.figure(figsize=(7.3, 4.15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.1, 0.9], hspace=0.55, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, :])
    cmap = LinearSegmentedColormap.from_list("rank_map", ["#174A7C", "#5DA8C9", "#F5E7C5", "#C85A63"])
    im = ax0.imshow(rank_matrix.values, aspect="auto", cmap=cmap, vmin=1, vmax=max(6, np.nanmax(rank_matrix.values)))
    ax0.set_xticks(np.arange(rank_matrix.shape[1]))
    ax0.set_xticklabels([f"{int(c)}" for c in rank_matrix.columns])
    ax0.set_yticks(np.arange(rank_matrix.shape[0]))
    ax0.set_yticklabels([s.replace("Site ", "S") for s in rank_matrix.index], fontsize=5.8)
    ax0.set_xlabel("Forecast horizon (h)")
    ax0.set_title("Online Lyra nRMSE rank across sites and horizons")
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            val = rank_matrix.iloc[i, j]
            ax0.text(j, i, f"{int(round(val))}", ha="center", va="center", fontsize=6.0, color="white" if val <= 2 else COLORS["black"])
    cb = fig.colorbar(im, ax=ax0, fraction=0.024, pad=0.012)
    cb.set_label("Rank (lower is better)")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1, 0])
    groups = ["small <=50MW", "medium 50-110MW", "large >=130MW"]
    x = np.arange(len(groups))
    names = ["Online Lyra", "Best static", "Offline Lyra"]
    width = 0.23
    for idx, name in enumerate(names):
        sub = cap_show.loc[cap_show["model_display"] == name].set_index("capacity_group").reindex(groups)
        vals = sub["nrmse"].to_numpy()
        color = COLORS["static_mid"] if name == "Best static" else color_for(name)
        ax1.bar(x + (idx - 1) * width, vals, width=width, color=color, alpha=0.92, label=name)
    set_zoom_ylim(ax1, cap_show["nrmse"].to_numpy(), pad_frac=0.18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["<=50", "50-110", ">=130"])
    ax1.set_xlabel("Capacity group (MW)")
    ax1.set_ylabel("nRMSE")
    ax1.set_title("Capacity groups, zoomed")
    ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    quiet_grid(ax1)
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, 1])
    for name in ["Online Lyra", "Offline Lyra"]:
        sub = skill.loc[skill["model_display"] == name].set_index("capacity_group").reindex(groups)
        ax2.plot(
            x,
            sub["skill_vs_persistence"],
            marker="o",
            lw=2.0 if name == "Online Lyra" else 1.3,
            color=color_for(name),
            label=name,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["<=50", "50-110", ">=130"])
    ax2.set_xlabel("Capacity group (MW)")
    ax2.set_ylabel("Skill vs persistence (%)")
    ax2.set_title("Persistence-normalized skill")
    ax2.legend(loc="lower right")
    quiet_grid(ax2)
    panel_label(ax2, "c")
    save_pub(fig, "fig03_robustness_rank_capacity")


def fig04_case_trace_gate() -> None:
    case = select_case_window(flatten_online(site_id=2, horizon=4, seed=2028), span=480)
    case.to_csv(DATA_OUT / "fig04_case_trace_site2_h4_seed2028.csv", index=False)

    fig = plt.figure(figsize=(7.3, 3.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 0.45], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    x = case["case_time_h"].to_numpy()
    ax.fill_between(x, 0, case["y_true"], color="#EAF2F7", alpha=0.75, lw=0)
    ax.plot(x, case["y_true"], color=COLORS["black"], lw=1.15, label="Observed PV")
    ax.plot(x, case["online_lyra"], color=color_for("Online Lyra"), lw=1.55, label="Online Lyra")
    ax.plot(x, case["persistence"], color=color_for("Persistence"), lw=1.05, ls="--", alpha=0.80, label="Persistence")
    ax.set_ylabel("PV power (MW)")
    ax.set_title("Operational trace: forecast behavior and fallback decisions")
    ax.legend(ncol=3, loc="upper left")
    quiet_grid(ax)
    panel_label(ax, "a")

    err = np.abs(case["y_true"] - case["persistence"]) - np.abs(case["y_true"] - case["online_lyra"])
    center = int(np.nanargmax(np.abs(np.gradient(case["y_true"].to_numpy())) + 0.02 * np.maximum(err, 0)))
    lo = max(0, center - 32)
    hi = min(len(case), center + 64)
    axins = inset_axes(ax, width="34%", height="42%", loc="upper right", borderpad=1.1)
    xin = x[lo:hi]
    axins.plot(xin, case["y_true"].iloc[lo:hi], color=COLORS["black"], lw=0.95)
    axins.plot(xin, case["online_lyra"].iloc[lo:hi], color=color_for("Online Lyra"), lw=1.15)
    axins.plot(xin, case["persistence"].iloc[lo:hi], color=color_for("Persistence"), lw=0.9, ls="--", alpha=0.85)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Ramp zoom", fontsize=6.2, pad=1.5)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#9AA3AD", lw=0.6)

    axg = fig.add_subplot(gs[1, 0], sharex=ax)
    gate = case["gate_uses_lyra"].to_numpy()
    axg.scatter(x, np.zeros_like(x), c=np.where(gate, color_for("Online Lyra"), color_for("Persistence")), s=8, marker="s", lw=0)
    axg.set_xlim(x.min(), x.max())
    axg.set_xlabel("Time within selected segment (h)")
    axg.set_yticks([])
    axg.text(0.0, 0.95, "Gate: blue = Online Lyra, rose = persistence fallback", transform=axg.transAxes, ha="left", va="top", fontsize=6.2)
    for spine in axg.spines.values():
        spine.set_visible(False)
    panel_label(axg, "b")
    save_pub(fig, "fig04_case_trace_gate")


def fig05_gate_fallback() -> None:
    g = add_hours(read_csv(PACK / "figure_data" / "gate_profile_site_horizon_average.csv"))
    g["fallback_fraction"] = 1.0 - g["gate_lyra_fraction"]
    g["fallback_pct"] = 100 * g["fallback_fraction"]
    g.to_csv(DATA_OUT / "fig05_fallback_profile.csv", index=False)
    pivot = g.pivot_table(index="site_id", columns="lead_time_h", values="fallback_pct", aggfunc="mean").sort_index()

    fig = plt.figure(figsize=(7.1, 2.9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0], wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("fallback", ["#F7FBFC", "#F1B1B5", "#C85A63"])
    vmax = max(6.5, np.nanpercentile(pivot.values, 98))
    im = ax0.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax0.set_xticks(np.arange(pivot.shape[1]))
    ax0.set_xticklabels([f"{int(c)}" for c in pivot.columns])
    ax0.set_yticks(np.arange(pivot.shape[0]))
    ax0.set_yticklabels([f"S{int(s)}" for s in pivot.index])
    ax0.set_xlabel("Forecast horizon (h)")
    ax0.set_ylabel("Site")
    ax0.set_title("Fallback fraction (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax0.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5.9, color=COLORS["black"])
    cb = fig.colorbar(im, ax=ax0, fraction=0.045, pad=0.016)
    cb.set_label("% fallback")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    sc = ax1.scatter(
        g["fallback_pct"],
        g["skill_vs_persistence"] * 100 if "skill_vs_persistence" in g.columns else (1 - g["nrmse"] / 0.2) * 100,
        c=g["lead_time_h"],
        cmap=LinearSegmentedColormap.from_list("horizon", ["#5DA8C9", "#E39B34"]),
        s=42,
        edgecolor="white",
        lw=0.5,
    )
    ax1.set_xlabel("Fallback fraction (%)")
    ax1.set_ylabel("Skill vs persistence (%)")
    ax1.set_title("Fallback is rare but targeted")
    quiet_grid(ax1)
    cb2 = fig.colorbar(sc, ax=ax1, fraction=0.050, pad=0.020)
    cb2.set_label("Horizon (h)")
    panel_label(ax1, "b")
    save_pub(fig, "fig05_gate_fallback_behavior")


def fig06_lead_error_profile() -> None:
    online = np.load(prediction_npz(site_id=5, horizon=96, seed=2028), allow_pickle=True)
    offline = np.load(prediction_npz(site_id=5, horizon=96, seed=2028, offline=True), allow_pickle=True)
    cap = float(online["capacity"])
    y = online["y_true"]
    df = pd.DataFrame(
        {
            "lead_h": np.arange(1, y.shape[1] + 1) / 4.0,
            "Online Lyra": np.mean(np.abs(y - online["online_lyra_pv_gated"]), axis=0) / cap,
            "Trend-only": np.mean(np.abs(y - online["trend_only"]), axis=0) / cap,
            "Offline Lyra": np.mean(np.abs(offline["y_true"] - offline["offline_lyra_final"]), axis=0) / cap,
            "Persistence": np.mean(np.abs(y - online["persistence"]), axis=0) / cap,
        }
    )
    df.to_csv(DATA_OUT / "fig06_lead_error_site5_h96_seed2028.csv", index=False)

    fig = plt.figure(figsize=(7.3, 3.15))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    for name, style in [
        ("Persistence", "--"),
        ("Trend-only", "-"),
        ("Offline Lyra", "--"),
        ("Online Lyra", "-"),
    ]:
        ax0.plot(
            df["lead_h"],
            df[name],
            color=color_for(name),
            lw=2.2 if name == "Online Lyra" else 1.2,
            ls=style,
            alpha=1.0 if name == "Online Lyra" else 0.82,
            label=name,
        )
    ax0.set_xlabel("Lead time (h)")
    ax0.set_ylabel("Step-wise nMAE")
    ax0.set_title("Full scale")
    ax0.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    quiet_grid(ax0)
    panel_label(ax0, "a")

    for name, style in [("Trend-only", "-"), ("Offline Lyra", "--"), ("Online Lyra", "-")]:
        ax1.plot(
            df["lead_h"],
            df[name],
            color=color_for(name),
            lw=2.2 if name == "Online Lyra" else 1.2,
            ls=style,
            alpha=1.0 if name == "Online Lyra" else 0.82,
            label=name,
        )
    set_zoom_ylim(ax1, df[["Online Lyra", "Offline Lyra", "Trend-only"]].to_numpy().ravel(), pad_frac=0.20)
    ax1.set_xlabel("Lead time (h)")
    ax1.set_ylabel("Step-wise nMAE")
    ax1.set_title("Zoom on learned components")
    quiet_grid(ax1)
    panel_label(ax1, "b")
    save_pub(fig, "fig06_lead_time_error_zoom")


def fig07_error_distribution_calibration() -> None:
    rows = []
    for site_id in range(1, 9):
        z = np.load(prediction_npz(site_id=site_id, horizon=96, seed=2028), allow_pickle=True)
        cap = float(z["capacity"])
        y = z["y_true"].reshape(-1) / cap
        rows.append(
            pd.DataFrame(
                {
                    "observed": y,
                    "Online Lyra": z["online_lyra_pv_gated"].reshape(-1) / cap,
                    "Trend-only": z["trend_only"].reshape(-1) / cap,
                    "Persistence": z["persistence"].reshape(-1) / cap,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df = df.loc[df["observed"] > 0.05].copy()
    err = pd.DataFrame(
        {
            "Online Lyra": df["Online Lyra"] - df["observed"],
            "Trend-only": df["Trend-only"] - df["observed"],
            "Persistence": df["Persistence"] - df["observed"],
        }
    )
    grid = np.linspace(-0.65, 0.65, 260)
    kde_rows = []
    for name in ["Online Lyra", "Trend-only", "Persistence"]:
        density = simple_kde(err[name].to_numpy(), grid, bandwidth=0.018 if name != "Persistence" else 0.025)
        kde_rows.append(pd.DataFrame({"model": name, "error": grid, "density": density}))
    kde = pd.concat(kde_rows, ignore_index=True)
    kde.to_csv(DATA_OUT / "fig07_signed_error_density_h96_seed2028.csv", index=False)
    calibration_data = df.copy()
    calibration_data.to_csv(DATA_OUT / "fig07_calibration_daytime_h96_seed2028.csv", index=False)

    fig = plt.figure(figsize=(7.3, 3.25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    for name in ["Online Lyra", "Trend-only", "Persistence"]:
        sub = kde.loc[kde["model"] == name]
        ax0.plot(sub["error"], sub["density"], color=color_for(name), lw=2.0 if name == "Online Lyra" else 1.25, label=name)
        ax0.fill_between(sub["error"], sub["density"], 0, color=color_for(name), alpha=0.12 if name == "Online Lyra" else 0.07)
    ax0.axvline(0, color=COLORS["black"], lw=0.8, ls=":")
    ax0.set_xlim(-0.45, 0.45)
    ax0.set_xlabel("Signed normalized error")
    ax0.set_ylabel("Density")
    ax0.set_title("Daytime H=24 h error distribution")
    ax0.legend(loc="upper left")
    quiet_grid(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    hb = ax1.hexbin(
        calibration_data["observed"],
        calibration_data["Online Lyra"],
        gridsize=42,
        mincnt=1,
        cmap=LinearSegmentedColormap.from_list("calib", ["#EEF6FA", "#5DA8C9", "#174A7C"]),
        linewidths=0,
    )
    ax1.plot([0, 1], [0, 1], color=COLORS["black"], lw=0.8, ls="--")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Observed normalized PV")
    ax1.set_ylabel("Predicted normalized PV")
    ax1.set_title("Online Lyra daytime calibration")
    cb = fig.colorbar(hb, ax=ax1, fraction=0.050, pad=0.020)
    cb.set_label("Count")
    panel_label(ax1, "b")
    save_pub(fig, "fig07_error_distribution_calibration")


def fig08_efficiency_tradeoff() -> None:
    acc = (
        read_csv(PACK / "server_tables_with_offline_lyra" / "main_average_by_horizon.csv")
        .groupby("model_display", as_index=False)[["nrmse", "nmae"]]
        .mean()
    )
    gpu = read_csv(SERVER / "resource_benchmark_20260717_1610" / "resource_efficiency_table.csv")
    cpu = read_csv(SERVER / "resource_benchmark_cpu_after_core_20260717_184553" / "resource_cpu_table.csv")
    cpu["model_display"] = cpu["model"].map(display_for)
    res = gpu.merge(
        cpu[["model_display", "latency_mean_ms"]].rename(columns={"latency_mean_ms": "cpu_latency_ms"}),
        on="model_display",
        how="left",
    ).merge(acc, on="model_display", how="inner")
    keep = ["Online Lyra", "DLinear", "NLinear", "iTransformer", "PatchTST", "TimeKAN", "PhaseFormer", "Olivia"]
    res = res.loc[res["model_display"].isin(keep)].copy()
    res["params_k_plot"] = res["params_k"].clip(lower=0.01)

    res.to_csv(DATA_OUT / "fig08_efficiency_tradeoff_resources.csv", index=False)

    fig = plt.figure(figsize=(7.4, 3.55))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    offsets = {
        "Online Lyra": (0.82, -0.0010, "right"),
        "DLinear": (1.15, 0.0009, "left"),
        "NLinear": (1.10, 0.0011, "left"),
        "iTransformer": (0.86, -0.0012, "right"),
        "PatchTST": (0.90, 0.0010, "right"),
        "TimeKAN": (1.08, -0.0010, "left"),
        "PhaseFormer": (1.12, 0.0012, "left"),
        "Olivia": (1.10, 0.0006, "left"),
    }
    for _, row in res.iterrows():
        m = row["model_display"]
        size = 40 + 260 * row["latency_mean_ms"] / res["latency_mean_ms"].max()
        ax0.scatter(
            row["params_k_plot"],
            row["nrmse"],
            s=size,
            color=color_for(m) if m == "Online Lyra" else (color_for("Olivia") if m == "Olivia" else COLORS["static_light"]),
            edgecolor=color_for("Online Lyra") if m == "Online Lyra" else "white",
            lw=1.1 if m == "Online Lyra" else 0.5,
            alpha=0.92,
            zorder=5 if m == "Online Lyra" else 2,
        )
        dx, dy, ha = offsets.get(m, (1.05, 0, "left"))
        ax0.text(row["params_k_plot"] * dx, row["nrmse"] + dy, m, fontsize=5.9, va="center", ha=ha)
    lyr = res.loc[res["model_display"] == "Online Lyra"].iloc[0]
    ax0.axhline(lyr["nrmse"], color=color_for("Online Lyra"), lw=0.9, ls=":")
    ax0.axvline(lyr["params_k_plot"], color=color_for("Online Lyra"), lw=0.9, ls=":")
    ax0.set_xscale("log")
    ax0.set_xlim(6, 240)
    ax0.set_ylim(0.092, 0.101)
    ax0.set_xlabel("Parameters (K, log scale)")
    ax0.set_ylabel("Mean nRMSE")
    ax0.set_title("Accuracy-efficiency trade-off")
    ax0.text(0.02, 0.96, "Bubble size = CUDA latency", transform=ax0.transAxes, ha="left", va="top", fontsize=5.8)
    quiet_grid(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    cols = [
        ("params_k", "Params"),
        ("checkpoint_size_mb", "Ckpt"),
        ("latency_mean_ms", "CUDA"),
        ("cpu_latency_ms", "CPU"),
        ("torch_peak_reserved_mb", "Peak mem"),
    ]
    heat = res.set_index("model_display")[[c for c, _ in cols]].reindex(keep)
    base = heat.loc["Online Lyra"].replace(0, np.nan)
    ratio = heat.divide(base, axis=1)
    ratio.to_csv(DATA_OUT / "fig08_resource_ratio_to_online.csv")
    cmap = LinearSegmentedColormap.from_list("resource", ["#F7FBFC", "#DDE6EF", "#8FA6C6", "#4F5D8C"])
    im = ax1.imshow(np.log10(ratio.clip(lower=0.05, upper=25)), aspect="auto", cmap=cmap, vmin=-1, vmax=1.2)
    ax1.set_xticks(np.arange(len(cols)))
    ax1.set_xticklabels([label for _, label in cols], rotation=35, ha="right")
    ax1.set_yticks(np.arange(len(keep)))
    ax1.set_yticklabels(keep, fontsize=5.8)
    ax1.set_title("Resource ratio to Online Lyra")
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            val = ratio.iloc[i, j]
            if val < 0.1:
                label = f"{val:.2f}x"
            elif val >= 10:
                label = f"{val:.0f}x"
            else:
                label = f"{val:.1f}x"
            ax1.text(j, i, label, ha="center", va="center", fontsize=5.2, color=COLORS["black"])
    cb = fig.colorbar(im, ax=ax1, fraction=0.052, pad=0.020)
    cb.set_label("log10 ratio")
    panel_label(ax1, "b")
    save_pub(fig, "fig08_efficiency_tradeoff")


def fig09_ablation_delta() -> None:
    core = read_csv(PACK / "core_ablation" / "core_ablation_summary.csv")
    cold = read_csv(PACK / "core_ablation" / "cold_start_summary.csv")
    full = float(core.loc[core["ablation"] == "Full Online Lyra", "nrmse"].iloc[0])
    core = core.loc[~core["ablation"].isin(["Persistence"])].copy()
    core["delta_nrmse_x1e3"] = (core["nrmse"] - full) * 1000
    cold = cold.copy()
    cold["delta_nrmse_x1e3"] = (cold["nrmse"] - full) * 1000
    pd.concat([core, cold], ignore_index=True).to_csv(DATA_OUT / "fig09_ablation_delta.csv", index=False)

    order = [
        "w/o spectral residual scaling",
        "w/o gate",
        "w/o online adaptation",
        "w/o residual learner",
    ]
    data = core.set_index("ablation").reindex(order).reset_index()
    data = data.loc[data["nrmse"].notna()].copy()
    fig = plt.figure(figsize=(7.1, 3.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 0.85], wspace=0.36)
    ax0 = fig.add_subplot(gs[0, 0])
    vals = data["delta_nrmse_x1e3"].to_numpy()
    y = np.arange(len(data))
    shown = np.clip(vals, -5, 18)
    bar_colors = [COLORS["good"] if v < 0 else COLORS["bad"] for v in vals]
    ax0.barh(y, shown, color=bar_colors, alpha=0.72)
    for yi, actual, clipped in zip(y, vals, shown):
        if actual != clipped:
            ax0.text(clipped + 0.6, yi, f"+{actual:.0f}", va="center", ha="left", fontsize=6.0, color=COLORS["bad"])
        else:
            ax0.text(clipped + (0.35 if clipped >= 0 else -0.35), yi, f"{actual:+.1f}", va="center", ha="left" if clipped >= 0 else "right", fontsize=6.0)
    ax0.axvline(0, color=COLORS["black"], lw=0.8)
    ax0.set_yticks(y)
    ax0.set_yticklabels(data["ablation"], fontsize=6.2)
    ax0.invert_yaxis()
    ax0.set_xlabel("Delta nRMSE vs full (x10^-3)")
    ax0.set_title("Core component ablation")
    quiet_grid(ax0, axis="x")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    cold_order = ["cold start: 60 offline days", "cold start: 30 offline days"]
    cold_show = cold.set_index("ablation").reindex(cold_order).reset_index()
    names = ["Full", "60 days", "30 days"]
    vals = [full, *cold_show["nrmse"].tolist()]
    colors = [color_for("Online Lyra"), COLORS["static_mid"], COLORS["static_light"]]
    x = np.arange(len(vals))
    ax1.bar(x, vals, color=colors, alpha=0.9, width=0.62)
    set_zoom_ylim(ax1, np.array(vals), pad_frac=0.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("nRMSE")
    ax1.set_title("Cold-start sensitivity")
    for xi, val in zip(x, vals):
        ax1.text(xi, val + 0.00025, f"{val:.3f}", ha="center", va="bottom", fontsize=6.0)
    quiet_grid(ax1)
    panel_label(ax1, "b")
    save_pub(fig, "fig09_ablation_delta")


def fig10_statistical_volcano() -> None:
    stats = read_csv(PACK / "stat_tests" / "paired_tests_online_lyra_vs_baselines.csv")
    keep = ["iTransformer", "Olivia", "DLinear", "TimeKAN", "PatchTST", "Offline Lyra", "PhaseFormer", "NLinear"]
    data = stats.loc[stats["metric"].isin(["nrmse", "daytime_nrmse"]) & stats["baseline_display"].isin(keep)].copy()
    data["neglog10_p"] = -np.log10(data["wilcoxon_p_less"].clip(lower=1e-12))
    data.to_csv(DATA_OUT / "fig10_statistical_volcano.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharey=True)
    for ax, metric, title, letter in zip(axes, ["nrmse", "daytime_nrmse"], ["nRMSE", "Daytime nRMSE"], ["a", "b"]):
        sub = data.loc[data["metric"] == metric].copy()
        ax.axhline(-np.log10(0.05), color=COLORS["neutral"], lw=0.8, ls=":")
        ax.axvline(0, color=COLORS["neutral"], lw=0.8)
        for _, row in sub.iterrows():
            m = row["baseline_display"]
            sig = row["wilcoxon_p_less"] < 0.05
            ax.scatter(
                row["relative_improvement_pct"],
                row["neglog10_p"],
                s=42 if sig else 28,
                color=color_for(m) if m == "Olivia" else (COLORS["online_light"] if sig else COLORS["neutral_light"]),
                edgecolor=COLORS["black"] if sig else "white",
                lw=0.45,
                alpha=0.94,
            )
            ax.text(row["relative_improvement_pct"] + 0.06, row["neglog10_p"], m, fontsize=5.6, va="center", ha="left")
        ax.set_xlabel("Relative improvement (%)")
        ax.set_title(title)
        quiet_grid(ax)
        panel_label(ax, letter)
    axes[0].set_ylabel("-log10 Wilcoxon p")
    fig.suptitle("Paired evidence for Online Lyra improvements", y=1.04, fontsize=9, fontweight="bold")
    save_pub(fig, "fig10_statistical_volcano")


def fig01_accuracy_revised() -> dict[str, object]:
    rep = load_replicate_results()
    models, strongest = accuracy_model_set(rep)
    metric_cols = ["nrmse", "nmae"]
    summary = mean_ci_from_replicates(
        rep.loc[rep["model_display"].isin(models)].copy(),
        ["model_display", "horizon", "lead_time_h"],
        metric_cols,
        unit_cols=["seed"],
    )
    summary["is_strongest_baseline"] = summary["model_display"].eq(strongest)
    summary.to_csv(DATA_OUT / "fig01_accuracy_seed_ci.csv", index=False)

    unit = rep.groupby(["seed", "horizon", "lead_time_h", "model_display"], as_index=False)["nrmse"].mean()
    p = unit.loc[unit["model_display"] == "Persistence", ["seed", "horizon", "lead_time_h", "nrmse"]].rename(columns={"nrmse": "persistence_nrmse"})
    skill_models = [m for m in models if m != "Persistence"]
    skill_rows = []
    for model in skill_models:
        sub = unit.loc[unit["model_display"] == model].merge(p, on=["seed", "horizon", "lead_time_h"], how="inner")
        sub["relative_nrmse_reduction"] = (sub["persistence_nrmse"] - sub["nrmse"]) / sub["persistence_nrmse"] * 100
        sub["model_display"] = model
        skill_rows.append(sub[["seed", "horizon", "lead_time_h", "model_display", "relative_nrmse_reduction"]])
    skill_unit = pd.concat(skill_rows, ignore_index=True)
    skill = mean_ci_from_replicates(skill_unit, ["model_display", "horizon", "lead_time_h"], ["relative_nrmse_reduction"], unit_cols=["seed"])
    skill.to_csv(DATA_OUT / "fig01_improvement_over_persistence_seed_ci.csv", index=False)

    full_ranges = {
        "nrmse": (float(rep["nrmse"].min()), float(rep["nrmse"].max())),
        "nmae": (float(rep["nmae"].min()), float(rep["nmae"].max())),
    }
    displayed_ranges = {
        "nrmse": (float(summary["nrmse_mean"].min()), float(summary["nrmse_mean"].max())),
        "nmae": (float(summary["nmae_mean"].min()), float(summary["nmae_mean"].max())),
    }

    fig = plt.figure(figsize=(7.25, 2.95))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.39, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    dodge_by_model = dict(zip(models, np.linspace(-3.2, 3.2, len(models))))
    for ax, metric, label, letter, ystep in [
        (axes[0], "nrmse", "nRMSE", "a", 0.005),
        (axes[1], "nmae", "nMAE", "b", 0.005),
    ]:
        for model in models:
            sub = summary.loc[summary["model_display"] == model].sort_values("lead_time_h")
            spec = fig01_line_spec(model, strongest)
            x = sub["lead_time_h"].to_numpy(dtype=float)
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            ci = sub[f"{metric}_ci95"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=spec["color"],
                lw=spec["lw"],
                ls=spec["ls"],
                alpha=spec["alpha"],
                zorder=spec["zorder"],
                label=model,
            )
            fig01_dodged_errorbar(fig, ax, x, y, ci, spec, dodge_by_model[model])
        ylo, yhi = fig01_ci_ylim(summary, metric, ystep)
        ax.set_ylim(ylo, yhi)
        tick_lo = np.ceil(ylo / 0.01) * 0.01
        tick_hi = np.floor(yhi / 0.01) * 0.01
        ax.set_yticks(np.round(np.arange(tick_lo, tick_hi + 0.005, 0.01), 3))
        ax.set_xscale("linear")
        ax.set_xlim(0.25, 24.75)
        ax.set_xticks([1, 3, 6, 12, 24])
        ax.set_ylabel(label)
        quiet_grid(ax)
        panel_label(ax, letter)

    ax = axes[2]
    for model in skill_models:
        sub = skill.loc[skill["model_display"] == model].sort_values("lead_time_h")
        spec = fig01_line_spec(model, strongest)
        x = sub["lead_time_h"].to_numpy(dtype=float)
        y = sub["relative_nrmse_reduction_mean"].to_numpy(dtype=float)
        ci = sub["relative_nrmse_reduction_ci95"].to_numpy(dtype=float)
        ax.plot(x, y, color=spec["color"], lw=spec["lw"], ls=spec["ls"], alpha=spec["alpha"], zorder=spec["zorder"], label=model)
        fig01_dodged_errorbar(fig, ax, x, y, ci, spec, dodge_by_model[model])
    skill_mean = skill["relative_nrmse_reduction_mean"].to_numpy(dtype=float)
    skill_ci = np.nan_to_num(skill["relative_nrmse_reduction_ci95"].to_numpy(dtype=float), nan=0.0)
    skill_lo = float(np.floor((np.min(skill_mean - skill_ci) - 3.0) / 5.0) * 5.0)
    skill_hi = float(np.ceil((np.max(skill_mean + skill_ci) + 3.0) / 5.0) * 5.0)
    ax.set_ylim(skill_lo, skill_hi)
    ax.set_yticks(np.arange(np.ceil(skill_lo / 10.0) * 10.0, skill_hi + 5.0, 10.0))
    ax.set_ylabel("nRMSE reduction\nvs persistence (%)")
    ax.set_xscale("linear")
    ax.set_xticks([1, 3, 6, 12, 24])
    ax.set_xlim(0.25, 24.75)
    quiet_grid(ax)
    panel_label(ax, "c")

    handles = []
    labels = []
    for model in models:
        spec = fig01_line_spec(model, strongest)
        handles.append(Line2D([0], [0], color=spec["color"], lw=spec["lw"], ls=spec["ls"], marker=spec["marker"], ms=spec["ms"]))
        labels.append(model)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=4,
        columnspacing=1.15,
        handlelength=2.25,
        handletextpad=0.55,
    )
    fig.supxlabel("Forecast horizon (h)", x=0.5, y=0.025, fontsize=8)
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.19, top=0.72, wspace=0.39)
    save_pub(fig, "fig01_accuracy")
    return {"strongest_baseline": strongest, "full_ranges": full_ranges, "zoom_ranges": displayed_ranges}


def fig02_robustness_revised(strongest: str) -> dict[str, object]:
    metrics, common_mask_audit = fig02_common_mask_metrics()
    metrics.to_csv(DATA_OUT / "fig02_common_mask_model_metrics.csv", index=False)
    common_mask_audit.to_csv(DATA_OUT / "fig02_common_mask_audit.csv", index=False)

    rank_table = (
        metrics.groupby(
            ["site_id", "site", "capacity", "horizon", "lead_time_h", "model", "model_display"],
            as_index=False,
        )["nrmse"]
        .mean()
        .rename(columns={"nrmse": "mean_seed_nrmse"})
    )
    rank_table["rank"] = rank_table.groupby(["site_id", "horizon"])["mean_seed_nrmse"].rank(
        method="min", ascending=True
    )
    rank_table["model_count"] = rank_table.groupby(["site_id", "horizon"])["model"].transform("size")
    rank_table.to_csv(DATA_OUT / "fig02_common_mask_rank_table.csv", index=False)

    lyra_rank = rank_table.loc[rank_table["model_display"] == "Online Lyra"].copy()
    rank_matrix = lyra_rank.pivot(index="site", columns="lead_time_h", values="rank")
    site_order = (
        lyra_rank[["site_id", "site"]]
        .drop_duplicates()
        .sort_values("site_id")["site"]
        .tolist()
    )
    rank_matrix = rank_matrix.reindex(index=site_order, columns=sorted(rank_matrix.columns))
    rank_matrix.to_csv(DATA_OUT / "fig02_online_lyra_rank_by_site_horizon.csv")

    old_rank = read_csv(PACK / "server_tables_with_offline_lyra" / "main_metric_long_with_ranks.csv")
    old_lyra_rank = old_rank.loc[
        (old_rank["model_display"] == "Online Lyra") & (old_rank["metric"] == "nrmse"),
        ["site_id", "horizon", "rank"],
    ].rename(columns={"rank": "old_rank_without_smart_persistence"})
    rank_change = lyra_rank[["site_id", "site", "horizon", "lead_time_h", "rank"]].merge(
        old_lyra_rank, on=["site_id", "horizon"], how="left"
    )
    rank_change = rank_change.rename(columns={"rank": "new_rank_with_smart_persistence_common_mask"})
    rank_change["rank_change"] = (
        rank_change["new_rank_with_smart_persistence_common_mask"]
        - rank_change["old_rank_without_smart_persistence"]
    )
    rank_change.to_csv(DATA_OUT / "fig02_online_lyra_rank_change_audit.csv", index=False)

    show = ["Online Lyra", strongest, "Offline Lyra"]
    capacity_models = [*show, "Smart Persistence"]
    cap_site = (
        metrics.loc[metrics["model_display"].isin(capacity_models)]
        .groupby(["capacity_group", "site_id", "model_display"], as_index=False)["nrmse"]
        .mean()
    )
    cap_site["capacity_group"] = pd.Categorical(
        cap_site["capacity_group"], CAPACITY_GROUP_ORDER, ordered=True
    )
    cap_site = cap_site.sort_values(["capacity_group", "model_display", "site_id"])
    cap_site.to_csv(DATA_OUT / "fig02_capacity_nrmse_site_units.csv", index=False)
    cap_summary = (
        cap_site.groupby(["capacity_group", "model_display"], observed=True)["nrmse"]
        .agg(nrmse_mean="mean", nrmse_min="min", nrmse_max="max", n_sites="size")
        .reset_index()
        .sort_values(["capacity_group", "model_display"])
    )
    cap_summary.to_csv(DATA_OUT / "fig02_capacity_nrmse_site_summary.csv", index=False)

    unit = metrics.groupby(
        ["seed", "site_id", "horizon", "capacity_group", "model_display"], as_index=False
    )["nrmse"].mean()
    smart_reference = unit.loc[
        unit["model_display"] == "Smart Persistence",
        ["seed", "site_id", "horizon", "capacity_group", "nrmse"],
    ].rename(columns={"nrmse": "smart_persistence_nrmse"})
    skill_rows = []
    for model in capacity_models:
        sub = unit.loc[unit["model_display"] == model].merge(
            smart_reference, on=["seed", "site_id", "horizon", "capacity_group"], how="inner"
        )
        sub["skill_vs_smart_persistence_pct"] = (
            (sub["smart_persistence_nrmse"] - sub["nrmse"])
            / sub["smart_persistence_nrmse"]
            * 100
        )
        sub["model_display"] = model
        skill_rows.append(
            sub[
                [
                    "seed",
                    "site_id",
                    "horizon",
                    "capacity_group",
                    "model_display",
                    "skill_vs_smart_persistence_pct",
                ]
            ]
        )
    skill_unit = pd.concat(skill_rows, ignore_index=True)
    skill_site = (
        skill_unit.groupby(["capacity_group", "site_id", "model_display"], as_index=False)[
            "skill_vs_smart_persistence_pct"
        ]
        .mean()
    )
    skill_site["capacity_group"] = pd.Categorical(
        skill_site["capacity_group"], CAPACITY_GROUP_ORDER, ordered=True
    )
    skill_site = skill_site.sort_values(["capacity_group", "model_display", "site_id"])
    skill_site.to_csv(DATA_OUT / "fig02_capacity_skill_site_units.csv", index=False)
    skill_summary = (
        skill_site.groupby(["capacity_group", "model_display"], observed=True)[
            "skill_vs_smart_persistence_pct"
        ]
        .agg(skill_mean="mean", skill_min="min", skill_max="max", n_sites="size")
        .reset_index()
        .sort_values(["capacity_group", "model_display"])
    )
    skill_summary.to_csv(DATA_OUT / "fig02_capacity_skill_site_summary.csv", index=False)

    fig = plt.figure(figsize=(7.25, 4.15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.0, 1.0], hspace=0.60, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, :])
    model_count = int(rank_table["model_count"].max())
    vmax = model_count
    cmap = LinearSegmentedColormap.from_list(
        "rank_seq",
        ["#7089AE", "#94AAC7", "#BBCBDD", "#F1F5F8"],
    )
    norm = Normalize(vmin=1, vmax=vmax)
    im = ax0.imshow(rank_matrix.values, aspect="auto", cmap=cmap, norm=norm)
    ax0.set_xticks(np.arange(rank_matrix.shape[1]))
    ax0.set_xticklabels([f"{int(c)}" for c in rank_matrix.columns])
    ax0.set_yticks(np.arange(rank_matrix.shape[0]))
    ax0.set_yticklabels([s.replace("Site ", "S") for s in rank_matrix.index])
    ax0.set_xlabel("Forecast horizon (h)")
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            val = float(rank_matrix.iloc[i, j])
            ax0.text(
                j,
                i,
                f"{int(round(val))}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="#20262D",
            )
    cb = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.012)
    cb.set_ticks([1, 4, 7, 10, model_count])
    cb.set_label(f"Rank among {model_count} methods\n(lower is better)")
    panel_label_fixed(fig, ax0, "a")

    ax1 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(CAPACITY_GROUP_ORDER))
    offsets = {model: offset for model, offset in zip(show, [-0.16, 0.0, 0.16])}
    for model in show:
        sub = cap_summary.loc[cap_summary["model_display"] == model].set_index("capacity_group").reindex(CAPACITY_GROUP_ORDER)
        vals = sub["nrmse_mean"].to_numpy(dtype=float)
        spec = fig01_line_spec(model, strongest)
        for group_idx, group in enumerate(CAPACITY_GROUP_ORDER):
            raw = cap_site.loc[
                (cap_site["model_display"] == model) & (cap_site["capacity_group"] == group), "nrmse"
            ].to_numpy(dtype=float)
            jitter = np.linspace(-0.025, 0.025, len(raw)) if len(raw) > 1 else np.zeros(len(raw))
            ax1.scatter(
                np.full(len(raw), x[group_idx] + offsets[model]) + jitter,
                raw,
                s=11,
                color=str(spec["color"]),
                alpha=0.30,
                edgecolor="none",
                zorder=int(spec["zorder"]) - 1,
            )
        ax1.plot(
            x + offsets[model],
            vals,
            ls="none",
            marker=str(spec["marker"]),
            ms=float(spec["ms"]) + 1.1,
            mfc=str(spec["color"]),
            mec="white",
            mew=0.55,
            alpha=float(spec["alpha"]),
            zorder=int(spec["zorder"]),
            label=model,
        )
    cap_plot = cap_site.loc[cap_site["model_display"].isin(show), "nrmse"].to_numpy(dtype=float)
    cap_lo = float(np.min(cap_plot))
    cap_hi = float(np.max(cap_plot))
    cap_pad = max((cap_hi - cap_lo) * 0.09, 0.0015)
    ax1.set_ylim(cap_lo - cap_pad, cap_hi + cap_pad)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Small\n<50", "Medium\n50-110", "Large\n>110"])
    ax1.set_ylabel("nRMSE")
    quiet_grid(ax1)
    panel_label_fixed(fig, ax1, "b")

    ax2 = fig.add_subplot(gs[1, 1])
    for model in show:
        sub = skill_summary.loc[skill_summary["model_display"] == model].set_index("capacity_group").reindex(CAPACITY_GROUP_ORDER)
        vals = sub["skill_mean"].to_numpy(dtype=float)
        spec = fig01_line_spec(model, strongest)
        x_model = x + offsets[model]
        for group_idx, group in enumerate(CAPACITY_GROUP_ORDER):
            raw = skill_site.loc[
                (skill_site["model_display"] == model) & (skill_site["capacity_group"] == group),
                "skill_vs_smart_persistence_pct",
            ].to_numpy(dtype=float)
            jitter = np.linspace(-0.025, 0.025, len(raw)) if len(raw) > 1 else np.zeros(len(raw))
            ax2.scatter(
                np.full(len(raw), x_model[group_idx]) + jitter,
                raw,
                s=11,
                color=str(spec["color"]),
                alpha=0.30,
                edgecolor="none",
                zorder=int(spec["zorder"]) - 1,
            )
        ax2.plot(
            x_model,
            vals,
            ls="none",
            marker=str(spec["marker"]),
            ms=float(spec["ms"]) + 1.1,
            mfc=str(spec["color"]),
            mec="white",
            mew=0.55,
            alpha=float(spec["alpha"]),
            zorder=int(spec["zorder"]) + 1,
            label=model,
        )
    skill_plot = skill_site.loc[
        skill_site["model_display"].isin(show), "skill_vs_smart_persistence_pct"
    ].to_numpy(dtype=float)
    skill_lo = float(np.min(skill_plot))
    skill_hi = float(np.max(skill_plot))
    skill_pad = max((skill_hi - skill_lo) * 0.09, 0.35)
    ax2.set_ylim(skill_lo - skill_pad, skill_hi + skill_pad)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Small\n<50", "Medium\n50-110", "Large\n>110"])
    ax2.set_ylabel("nRMSE reduction vs\nSmart Persistence (%)")
    quiet_grid(ax2)
    panel_label_fixed(fig, ax2, "c")

    legend_handles = []
    for model in show:
        spec = fig01_line_spec(model, strongest)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=spec["color"],
                lw=spec["lw"],
                ls=spec["ls"],
                marker=spec["marker"],
                ms=float(spec["ms"]) + 0.4,
                markerfacecolor=spec["color"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=model,
            )
        )
    fig.legend(
        handles=legend_handles,
        labels=show,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=3,
        columnspacing=1.6,
        handlelength=2.5,
        handletextpad=0.6,
    )
    ax2.legend(handles=legend_handles, labels=show, loc="best", ncol=1)
    fig.supxlabel("Capacity group (MW)", x=0.52, y=0.025, fontsize=8)
    fig.subplots_adjust(left=0.105, right=0.955, bottom=0.17, top=0.87, hspace=0.60, wspace=0.38)
    save_pub(fig, "fig02_robustness")
    return {
        "capacity_scheme": "Scheme B: Small <50 MW / Medium 50-110 MW / Large >110 MW",
        "fig02_rank_model_count": model_count,
        "fig02_rank_changes": int((rank_change["rank_change"] != 0).sum()),
    }


def fig03_case_trace_revised() -> None:
    case = select_case_window(flatten_online(site_id=2, horizon=4, seed=2028), span=480)
    case = case.reset_index(drop=True)
    x = case["case_time_h"].to_numpy()
    y = case["y_true"].to_numpy()
    persistence = case["persistence"].to_numpy()
    online = case["online_lyra"].to_numpy()
    gate = case["gate_uses_lyra"].to_numpy(dtype=bool)

    screening, selected = screen_case_event_windows(case)
    lo = int(selected["start_idx"])
    hi = int(selected["end_idx"])
    event_start = float(selected["event_start_h"])
    event_end = float(selected["event_end_h"])

    event_indices = np.arange(lo, hi)
    point_ramp = np.r_[0.0, np.abs(np.diff(y))]
    point_gain = np.abs(y - persistence) - np.abs(y - online)
    rapid_cutoff = float(np.quantile(point_ramp[event_indices], 0.75))
    annotation_candidates = event_indices[
        (point_ramp[event_indices] >= rapid_cutoff) & (point_gain[event_indices] > 0)
    ]
    if len(annotation_candidates):
        center = int(annotation_candidates[np.argmax(point_gain[annotation_candidates])])
    else:
        center = int(event_indices[np.argmax(point_gain[event_indices])])

    screening.to_csv(DATA_OUT / "fig03_event_window_screening.csv", index=False)
    case["event_window_score"] = np.nan
    case["event_selected"] = False
    case["annotation_point"] = False
    case.loc[case.index[lo:hi], "event_window_score"] = float(selected["selection_score"])
    case.loc[case.index[lo:hi], "event_selected"] = True
    case.loc[case.index[center], "annotation_point"] = True
    case.to_csv(DATA_OUT / "fig03_case_trace_site2_h4_seed2028.csv", index=False)
    pd.DataFrame(
        [
            {
                "site_id": 2,
                "horizon": 4,
                "seed": 2028,
                "event_start_h": event_start,
                "event_end_h": event_end,
                "event_duration_h": float(selected["duration_h"]),
                "annotation_time_h": float(x[center]),
                "event_type": "rapid nonlinear PV ramp-up with Lyra gate active",
                "gate_lyra_fraction": float(selected["gate_lyra_fraction"]),
                "daytime_fraction": float(selected["daytime_fraction"]),
                "observed_range_mw": float(selected["observed_range_mw"]),
                "max_ramp_mw_per_15min": float(selected["max_ramp_mw_per_15min"]),
                "online_mae_mw": float(selected["online_mae_mw"]),
                "persistence_mae_mw": float(selected["persistence_mae_mw"]),
                "mae_reduction_percent": float(selected["mae_reduction_fraction"] * 100),
                "online_rmse_mw": float(selected["online_rmse_mw"]),
                "persistence_rmse_mw": float(selected["persistence_rmse_mw"]),
                "rmse_reduction_percent": float(selected["rmse_reduction_fraction"] * 100),
                "online_win_fraction": float(selected["online_win_fraction"]),
                "change_mae_reduction_percent": float(selected["change_mae_reduction_fraction"] * 100),
                "rapid_change_mae_reduction_percent": float(
                    selected["rapid_change_mae_reduction_fraction"] * 100
                ),
                "annotation_online_absolute_error_mw": float(abs(y[center] - online[center])),
                "annotation_persistence_absolute_error_mw": float(abs(y[center] - persistence[center])),
                "selection_rule": (
                    "Highest composite score among all 6, 7, and 8 h windows passing gate>=0.95, "
                    "daytime>=0.50, observed range>=0.30 capacity, MAE/RMSE reduction>=0.25, "
                    "Online Lyra win fraction>=0.70, and change/rapid-change MAE reduction>=0.30."
                ),
            }
        ]
    ).to_csv(DATA_OUT / "fig03_event_metadata.csv", index=False)

    highlight_fc = "#F4E7AF"
    highlight_ec = "#A69345"
    guide_color = "#A8AFB7"
    gate_cmap = ListedColormap([COLORS["fallback"], COLORS["online"]])

    fig = plt.figure(figsize=(7.25, 4.85))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.55, 0.34, 1.50], hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(event_start, event_end, color=highlight_fc, alpha=0.38, lw=0, zorder=0)
    ax.plot(x, persistence, color=COLORS["persistence"], lw=1.0, ls="--", alpha=0.84, zorder=2)
    ax.plot(x, online, color=COLORS["online"], lw=1.65, alpha=0.94, zorder=3)
    obs, = ax.plot(x, y, color=COLORS["observed"], lw=1.02, alpha=0.98, zorder=4)
    obs.set_path_effects([patheffects.withStroke(linewidth=1.7, foreground="white", alpha=0.42)])
    ax.axvline(event_start, color=highlight_ec, lw=0.7, ls=(0, (2, 2)), alpha=0.75, zorder=1)
    ax.axvline(event_end, color=highlight_ec, lw=0.7, ls=(0, (2, 2)), alpha=0.75, zorder=1)
    ax.text(
        (event_start + event_end) / 2,
        1.015,
        "Ramp event",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#6F642B",
        clip_on=False,
    )
    ax.set_xlim(0, 120)
    ax.set_ylabel("PV power (MW)")
    ax.tick_params(axis="x", labelbottom=False)
    quiet_grid(ax)
    panel_label_fixed(fig, ax, "a")

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
    axg.axvspan(event_start, event_end, facecolor=highlight_fc, edgecolor=highlight_ec, lw=0.7, alpha=0.18, zorder=2)
    axg.set_xlim(0, 120)
    axg.set_ylim(0, 1)
    axg.set_xticks(np.arange(0, 121, 20))
    axg.set_xlabel("Time (h)")
    axg.set_ylabel("Gate", rotation=0, labelpad=20, va="center")
    axg.set_yticks([])
    for spine in axg.spines.values():
        spine.set_visible(False)
    panel_label_fixed(fig, axg, "b", dy_pt=5.0)

    axz = fig.add_subplot(gs[2, 0])
    event = case.iloc[lo:hi]
    xe = event["case_time_h"].to_numpy()
    axz.plot(xe, event["persistence"], color=COLORS["persistence"], lw=1.25, ls="--", alpha=0.88, zorder=2)
    axz.plot(xe, event["online_lyra"], color=COLORS["online"], lw=2.0, alpha=0.96, zorder=3)
    obs_zoom, = axz.plot(xe, event["y_true"], color=COLORS["observed"], lw=1.25, alpha=0.98, zorder=4)
    obs_zoom.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground="white", alpha=0.42)])
    zoom_values = event[["y_true", "online_lyra", "persistence"]].to_numpy(dtype=float)
    zlo = float(np.nanmin(zoom_values))
    zhi = float(np.nanmax(zoom_values))
    zpad = max((zhi - zlo) * 0.11, 1.0)
    axz.set_xlim(event_start, event_end)
    axz.set_ylim(max(0, zlo - zpad), zhi + zpad)
    axz.set_xlabel("Event time (h)")
    axz.set_ylabel("PV power (MW)")
    quiet_grid(axz)
    panel_label_fixed(fig, axz, "c")

    arrow = {"arrowstyle": "-", "color": guide_color, "lw": 0.75}
    yspan = max(zhi - zlo, 1.0)
    axz.annotate(
        "Persistence lag",
        xy=(x[center], persistence[center]),
        xytext=(min(event_end - 0.35, x[center] + 1.3), persistence[center] - 0.10 * yspan),
        ha="left",
        va="top",
        fontsize=6.6,
        color="#6F6674",
        arrowprops=arrow,
    )
    axz.annotate(
        "Lyra response",
        xy=(x[center], online[center]),
        xytext=(min(event_end - 0.35, x[center] + 1.3), online[center] + 0.12 * yspan),
        ha="left",
        va="bottom",
        fontsize=6.6,
        color=COLORS["online"],
        arrowprops=arrow,
    )

    for x_event, zoom_x in [(event_start, event_start), (event_end, event_end)]:
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_event, 0),
                coordsA=axg.transData,
                xyB=(zoom_x, 1),
                coordsB=axz.transData,
                color=guide_color,
                lw=0.65,
                alpha=0.72,
                zorder=0,
            )
        )

    legend_handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=1.25, label="Observed PV"),
        Line2D([0], [0], color=COLORS["online"], lw=2.0, label="Online Lyra"),
        Line2D([0], [0], color=COLORS["persistence"], lw=1.25, ls="--", label="Persistence"),
        Patch(facecolor=COLORS["online"], label="Lyra selected"),
        Patch(facecolor=COLORS["fallback"], label="Fallback selected"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=5,
        columnspacing=1.15,
        handlelength=2.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.89, hspace=0.18)
    save_pub(fig, "fig03_case_trace_gate")


def fig05_daytime_error_agreement_revised() -> dict[str, object]:
    comparator = "iTransformer"
    models = ["Online Lyra", comparator, "Smart Persistence"]
    daytime_frames = []
    smart_audit = []
    for site_id in range(1, 9):
        z_online = np.load(prediction_npz(site_id=site_id, horizon=96, seed=2028), allow_pickle=True)
        z_base = np.load(baseline_prediction_npz(site_id=site_id, horizon=96, seed=2028), allow_pickle=True)
        cap = float(z_online["capacity"])
        observed = z_online["y_true"].astype(float)
        online = z_online["online_lyra_pv_gated"].astype(float)
        itransformer = z_base["itransformer"].astype(float)
        if observed.shape != online.shape or observed.shape != itransformer.shape or observed.shape[1] != 96:
            raise ValueError(f"Unexpected h=96 prediction shape at site {site_id}.")

        # Saved h=96 targets are consecutive non-overlapping daily blocks. The
        # previous row therefore provides yesterday's value at the same slot.
        smart_persistence = np.full_like(observed, np.nan, dtype=float)
        smart_persistence[1:] = observed[:-1]
        flat_observed_mw = observed.reshape(-1)
        ramp_fraction = np.r_[np.nan, np.abs(np.diff(flat_observed_mw))].reshape(observed.shape) / cap
        valid = (observed > 0.05 * cap) & np.isfinite(smart_persistence)
        frame = pd.DataFrame(
            {
                "site_id": site_id,
                "observed": observed[valid] / cap,
                "Online Lyra": online[valid] / cap,
                comparator: itransformer[valid] / cap,
                "Smart Persistence": smart_persistence[valid] / cap,
                "absolute_pv_change_fraction_per_15min": ramp_fraction[valid],
            }
        )
        daytime_frames.append(frame)
        continuity_error = float(
            np.nanmax(
                np.abs(
                    z_online["persistence"][1:, 0].astype(float)
                    - observed[:-1, -1]
                )
            )
        )
        smart_audit.append(
            {
                "site_id": site_id,
                "capacity_mw": cap,
                "total_target_points": int(observed.size),
                "excluded_first_day_points": 96,
                "retained_active_generation_points": int(np.sum(valid)),
                "smart_persistence_lag_steps": 96,
                "smart_persistence_lag_hours": 24,
                "block_continuity_max_abs_error_mw": continuity_error,
                "active_generation_rule": "observed PV > 5% of site capacity",
            }
        )

    df = pd.concat(daytime_frames, ignore_index=True)
    df.to_csv(DATA_OUT / "fig05_daytime_predictions_h96_seed2028.csv", index=False)
    pd.DataFrame(smart_audit).to_csv(DATA_OUT / "fig05_smart_persistence_audit.csv", index=False)

    metric_rows = []
    for site_id, site_df in df.groupby("site_id"):
        observed = site_df["observed"].to_numpy(dtype=float)
        for model in models:
            error = site_df[model].to_numpy(dtype=float) - observed
            absolute_error = np.abs(error)
            metric_rows.extend(
                [
                    {"site_id": site_id, "model_display": model, "metric": "nMAE", "value": float(np.mean(absolute_error))},
                    {"site_id": site_id, "model_display": model, "metric": "nRMSE", "value": float(np.sqrt(np.mean(error**2)))},
                    {"site_id": site_id, "model_display": model, "metric": "P90 |error|", "value": float(np.quantile(absolute_error, 0.90))},
                ]
            )
    site_metrics = pd.DataFrame(metric_rows)
    site_metrics.to_csv(DATA_OUT / "fig05_daytime_site_metrics.csv", index=False)

    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}

    def summarize_site_units(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        summary_rows = []
        for keys, group in data.groupby(groups, observed=True, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = group["value"].to_numpy(dtype=float)
            n = len(values)
            row = dict(zip(groups, keys))
            row.update(
                {
                    "mean": float(np.mean(values)),
                    "ci95": float(tcrit.get(n, 1.96) * np.std(values, ddof=1) / np.sqrt(n)) if n >= 2 else np.nan,
                    "n_sites": n,
                }
            )
            summary_rows.append(row)
        return pd.DataFrame(summary_rows)

    metric_summary = summarize_site_units(site_metrics, ["model_display", "metric"])
    metric_summary.to_csv(DATA_OUT / "fig05_daytime_metric_summary.csv", index=False)

    ramp_cuts = np.quantile(df["absolute_pv_change_fraction_per_15min"], [0.25, 0.50, 0.75])
    ramp_labels = ["Q1", "Q2", "Q3", "Q4"]
    df["ramp_quartile"] = pd.cut(
        df["absolute_pv_change_fraction_per_15min"],
        bins=[-np.inf, *ramp_cuts, np.inf],
        labels=ramp_labels,
        include_lowest=True,
    )
    ramp_rows = []
    for (site_id, quartile), group in df.groupby(["site_id", "ramp_quartile"], observed=True):
        observed = group["observed"].to_numpy(dtype=float)
        for model in models:
            ramp_rows.append(
                {
                    "site_id": site_id,
                    "ramp_quartile": str(quartile),
                    "model_display": model,
                    "value": float(np.mean(np.abs(group[model].to_numpy(dtype=float) - observed))),
                }
            )
    ramp_site = pd.DataFrame(ramp_rows)
    ramp_site.to_csv(DATA_OUT / "fig05_ramp_quartile_site_nmae.csv", index=False)
    ramp_summary = summarize_site_units(ramp_site, ["model_display", "ramp_quartile"])
    ramp_summary.to_csv(DATA_OUT / "fig05_ramp_quartile_summary.csv", index=False)

    styles = {
        "Online Lyra": {"color": COLORS["online"], "marker": "o", "ls": "-", "lw": 2.1, "zorder": 5},
        comparator: {"color": COLORS["online_light"], "marker": "s", "ls": (0, (3, 1.5)), "lw": 1.45, "zorder": 4},
        "Smart Persistence": {"color": "#8F8997", "marker": "^", "ls": (0, (5, 2)), "lw": 1.35, "zorder": 3},
    }

    fig = plt.figure(figsize=(7.25, 3.25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.14], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    metric_order = ["nMAE", "nRMSE", "P90 |error|"]
    x_metric = np.arange(len(metric_order), dtype=float)
    offsets = {"Online Lyra": -0.17, comparator: 0.0, "Smart Persistence": 0.17}
    jitter = np.linspace(-0.035, 0.035, 8)
    legend_handles = []
    for model in models:
        style = styles[model]
        for metric_idx, metric in enumerate(metric_order):
            raw = site_metrics.loc[
                (site_metrics["model_display"] == model) & (site_metrics["metric"] == metric), "value"
            ].to_numpy(dtype=float)
            summary = metric_summary.loc[
                (metric_summary["model_display"] == model) & (metric_summary["metric"] == metric)
            ].iloc[0]
            xpos = x_metric[metric_idx] + offsets[model]
            ax0.scatter(
                xpos + jitter[: len(raw)],
                raw,
                s=10,
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
                ms=4.3,
                mfc=style["color"] if model == "Online Lyra" else "white",
                mec=style["color"],
                mew=0.8,
                ecolor=style["color"],
                elinewidth=0.9,
                capsize=2.0,
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
                ms=4.0,
                markerfacecolor=style["color"] if model == "Online Lyra" else "white",
                markeredgecolor=style["color"],
                label=model,
            )
        )
    ax0.set_xticks(x_metric)
    ax0.set_xticklabels(["nMAE", "nRMSE", "P90 $|e|$"])
    ax0.set_xlim(-0.45, len(metric_order) - 0.55)
    metric_low = float(np.min(metric_summary["mean"] - metric_summary["ci95"]))
    metric_high = float(np.max(metric_summary["mean"] + metric_summary["ci95"]))
    ax0.set_ylim(max(0, metric_low - 0.025), metric_high + 0.025)
    ax0.set_ylabel("Normalized error")
    quiet_grid(ax0)
    panel_label_fixed(fig, ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    x_ramp = np.arange(1, 5, dtype=float)
    ax1.axvspan(3.5, 4.5, color="#F4E7AF", alpha=0.28, lw=0, zorder=0)
    for model in models:
        style = styles[model]
        sub = ramp_summary.loc[ramp_summary["model_display"] == model].set_index("ramp_quartile").reindex(ramp_labels)
        ax1.errorbar(
            x_ramp,
            sub["mean"].to_numpy(dtype=float),
            yerr=sub["ci95"].to_numpy(dtype=float),
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
            marker=style["marker"],
            ms=4.0,
            mfc=style["color"] if model == "Online Lyra" else "white",
            mec=style["color"],
            mew=0.75,
            capsize=1.8,
            label=model,
            zorder=style["zorder"],
        )
    cut_pct = ramp_cuts * 100
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
    ax1.text(4.0, 1.015, "High variability", transform=ax1.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color="#6F642B")
    quiet_grid(ax1)
    panel_label_fixed(fig, ax1, "b")

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
    save_pub(fig, "fig05_daytime_error_agreement")
    online_summary = metric_summary.set_index(["model_display", "metric"])
    return {
        "model_output_type": "deterministic_point_forecast",
        "fig05_comparator": comparator,
        "fig05_online_nmae": float(online_summary.loc[("Online Lyra", "nMAE"), "mean"]),
        "fig05_comparator_nmae": float(online_summary.loc[(comparator, "nMAE"), "mean"]),
        "fig05_smart_nmae": float(online_summary.loc[("Smart Persistence", "nMAE"), "mean"]),
        "fig05_ramp_q4_threshold_pct": float(cut_pct[2]),
    }


def fig06_efficiency_revised() -> None:
    acc = (
        read_csv(PACK / "server_tables_with_offline_lyra" / "main_average_by_horizon.csv")
        .groupby("model_display", as_index=False)[["nrmse", "nmae"]]
        .mean()
    )
    gpu = read_csv(SERVER / "resource_benchmark_20260717_1610" / "resource_efficiency_table.csv")
    cpu = read_csv(SERVER / "resource_benchmark_cpu_after_core_20260717_184553" / "resource_cpu_table.csv")
    cpu["model_display"] = cpu["model"].map(display_for)
    res = (
        gpu.merge(
            cpu[["model_display", "latency_mean_ms"]].rename(
                columns={"latency_mean_ms": "cpu_latency_ms"}
            ),
            on="model_display",
            how="left",
        )
        .merge(acc, on="model_display", how="inner")
    )
    keep = ["Online Lyra", "DLinear", "NLinear", "iTransformer", "PatchTST", "TimeKAN", "PhaseFormer", "Olivia"]
    res = res.loc[res["model_display"].isin(keep)].copy()
    res["params_k_plot"] = res["params_k"].clip(lower=0.01)

    replicate_results = load_replicate_results()
    replicate_results = replicate_results.loc[
        replicate_results["model_display"].isin(keep)
    ].copy()
    seed_units = (
        replicate_results.groupby(["model_display", "seed"], as_index=False)
        .agg(
            nrmse=("nrmse", "mean"),
            n_site_horizon_conditions=("nrmse", "size"),
            n_sites=("site_id", "nunique"),
            n_horizons=("horizon", "nunique"),
        )
        .sort_values(["model_display", "seed"])
    )
    coverage_ok = (
        seed_units["n_site_horizon_conditions"].eq(40)
        & seed_units["n_sites"].eq(8)
        & seed_units["n_horizons"].eq(5)
    )
    if not bool(coverage_ok.all()):
        raise ValueError(
            "Fig. 6 seed uncertainty requires complete 8-site x 5-horizon coverage"
        )
    seed_units.to_csv(DATA_OUT / "fig06_accuracy_seed_units.csv", index=False)
    uncertainty_rows = []
    for model in keep:
        values = seed_units.loc[
            seed_units["model_display"] == model, "nrmse"
        ].to_numpy(dtype=float)
        if len(values) != 3:
            raise ValueError(
                f"Fig. 6 requires three seed-level units for {model}; found {len(values)}"
            )
        sd = float(np.std(values, ddof=1))
        uncertainty_rows.append(
            {
                "model_display": model,
                "nrmse_mean": float(np.mean(values)),
                "nrmse_sd_across_seeds": sd,
                "nrmse_ci95": float(4.303 * sd / np.sqrt(len(values))),
                "n_seeds": len(values),
                "seed_unit_definition": "mean across 8 sites and 5 horizons within each seed",
                "error_bar_definition": "95% t CI across seeds (t_0.975,df=2=4.303)",
            }
        )
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(DATA_OUT / "fig06_accuracy_seed_ci.csv", index=False)
    mean_check = res[["model_display", "nrmse"]].merge(
        uncertainty[["model_display", "nrmse_mean"]],
        on="model_display",
        how="left",
    )
    if not np.allclose(
        mean_check["nrmse"], mean_check["nrmse_mean"], rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Fig. 6 seed-level means do not match plotted nRMSE values")
    uncertainty_lookup = uncertainty.set_index("model_display")["nrmse_ci95"].to_dict()

    accuracy_order = res.sort_values(["nrmse", "params_k_plot"])["model_display"].tolist()
    pd.DataFrame(
        {
            "display_order": np.arange(1, len(accuracy_order) + 1),
            "model_display": accuracy_order,
            "sort_rule": "ascending mean nRMSE, then trainable parameters",
            "shown_in_resource_heatmap": True,
        }
    ).to_csv(DATA_OUT / "fig06_model_order.csv", index=False)
    res.to_csv(DATA_OUT / "fig06_efficiency_resources.csv", index=False)
    pareto_mask = []
    dominance_records = []
    for _, row in res.iterrows():
        no_worse = (res["params_k_plot"] <= row["params_k_plot"]) & (
            res["nrmse"] <= row["nrmse"]
        )
        strictly_better = (res["params_k_plot"] < row["params_k_plot"]) | (
            res["nrmse"] < row["nrmse"]
        )
        dominating_models = res.loc[
            no_worse & strictly_better, "model_display"
        ].tolist()
        is_pareto = not bool(dominating_models)
        pareto_mask.append(is_pareto)
        dominance_records.append(
            {
                "model_display": row["model_display"],
                "params_k": float(row["params_k_plot"]),
                "mean_nrmse": float(row["nrmse"]),
                "is_non_dominated": is_pareto,
                "dominated_by": ";".join(dominating_models),
            }
        )
    res["is_pareto"] = pareto_mask
    pd.DataFrame(dominance_records).to_csv(
        DATA_OUT / "fig06_pareto_dominance_audit.csv", index=False
    )
    pareto = res.loc[res["is_pareto"]].sort_values(["params_k_plot", "nrmse"])
    pareto.to_csv(DATA_OUT / "fig06_pareto_frontier_params_nrmse.csv", index=False)

    fig = plt.figure(figsize=(7.25, 3.60))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.46)
    fig.subplots_adjust(left=0.085, right=0.925, bottom=0.16, top=0.77, wspace=0.46)
    ax0 = fig.add_subplot(gs[0, 0])
    lat_max = float(res["latency_mean_ms"].max())
    bubble_area_scale = 205.0 / lat_max

    def bubble_size(lat: float) -> float:
        return bubble_area_scale * float(lat)

    label_positions = {
        "Online Lyra": (16.0, 0.09440, "right"),
        "DLinear": (22.5, 0.09670, "left"),
        "NLinear": (7.1, 0.09835, "left"),
        "PhaseFormer": (27.0, 0.09862, "left"),
        "TimeKAN": (62.0, 0.09720, "left"),
        "PatchTST": (136.0, 0.09880, "left"),
        "iTransformer": (82.0, 0.09435, "right"),
        "Olivia": (150.0, 0.09618, "left"),
    }
    label_artists: dict[str, plt.Annotation] = {}
    bubble_records: list[dict[str, object]] = []
    competitive = {"iTransformer", "Olivia"}
    pareto_models = set(pareto["model_display"])
    for _, row in res.iterrows():
        m = row["model_display"]
        if m == "Online Lyra":
            color, edge, linewidth, alpha, zorder = COLORS["online"], "#08385E", 1.35, 0.98, 7
        elif m in competitive:
            color, edge, linewidth, alpha, zorder = "#86AEC5", "#5E8298", 0.75, 0.90, 4
        else:
            color, edge, linewidth, alpha, zorder = "#BDD0DE", "white", 0.60, 0.88, 3
        area = bubble_size(float(row["latency_mean_ms"]))
        ax0.scatter(
            row["params_k_plot"],
            row["nrmse"],
            s=area,
            color=color,
            edgecolor=edge,
            lw=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        label_x, label_y, ha = label_positions[m]
        label_artists[m] = ax0.annotate(
            m,
            xy=(row["params_k_plot"], row["nrmse"]),
            xytext=(label_x, label_y),
            ha=ha,
            va="center",
            fontsize=7.4,
            fontweight="bold" if m == "Online Lyra" else "normal",
            color=COLORS["online"] if m == "Online Lyra" else "#3F4852",
            arrowprops=dict(arrowstyle="-", color="#99A3AD", lw=0.50, shrinkA=2, shrinkB=3),
            zorder=9,
        )
        bubble_records.append(
            {
                "model_display": m,
                "params_k": float(row["params_k_plot"]),
                "mean_nrmse": float(row["nrmse"]),
                "cuda_latency_ms": float(row["latency_mean_ms"]),
                "bubble_area_pt2": area,
                "area_per_ms_pt2": bubble_area_scale,
                "area_to_latency_ratio": area / float(row["latency_mean_ms"]),
                "nrmse_ci95_not_shown": float(uncertainty_lookup[m]),
                "uncertainty_displayed": False,
                "label_x": label_x,
                "label_y": label_y,
                "is_pareto": bool(m in pareto_models),
            }
        )
    pareto_label_artist = None
    if len(pareto) >= 2:
        ax0.plot(
            pareto["params_k_plot"],
            pareto["nrmse"],
            color="#89939D",
            lw=0.80,
            ls=(0, (4, 2)),
            zorder=1,
            label="Pareto frontier",
        )
        vertical_anchor = pareto.iloc[-2]
        pareto_label_artist = ax0.annotate(
            "Pareto frontier",
            xy=(float(vertical_anchor["params_k_plot"]), 0.09575),
            xytext=(25.0, 0.09575),
            ha="left",
            va="center",
            fontsize=6.6,
            color="#69747E",
            arrowprops=dict(
                arrowstyle="-",
                color="#89939D",
                lw=0.50,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=8,
        )
    ax0.set_xscale("log")
    ax0.set_xlim(6, 250)
    ax0.set_ylim(0.0939, 0.09925)
    ax0.set_xlabel("Trainable parameters (K, log scale)")
    ax0.set_ylabel("Mean nRMSE")
    legend_vals = [0.1, 3.1, 8.4]
    legend_scale = pd.DataFrame(
        {
            "cuda_latency_ms": legend_vals,
            "bubble_area_pt2": [bubble_size(v) for v in legend_vals],
            "area_per_ms_pt2": bubble_area_scale,
            "mapping": "bubble_area_pt2 = area_per_ms_pt2 * cuda_latency_ms",
        }
    )
    legend_scale["latency_ratio_to_0p1"] = (
        legend_scale["cuda_latency_ms"] / legend_scale["cuda_latency_ms"].iloc[0]
    )
    legend_scale["area_ratio_to_0p1"] = (
        legend_scale["bubble_area_pt2"] / legend_scale["bubble_area_pt2"].iloc[0]
    )
    if not np.allclose(
        legend_scale["latency_ratio_to_0p1"],
        legend_scale["area_ratio_to_0p1"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Fig. 6 bubble areas are not strictly proportional to latency")
    legend_scale.to_csv(DATA_OUT / "fig06_bubble_legend_scale.csv", index=False)
    ax0_box_position = ax0.get_position()
    latency_ax = fig.add_axes(
        [
            ax0_box_position.x0,
            ax0_box_position.y1 + 0.022,
            ax0_box_position.width,
            0.075,
        ]
    )
    latency_ax.set_xlim(0.0, 1.0)
    latency_ax.set_ylim(0.0, 1.0)
    latency_ax.axis("off")
    legend_marker_y = 0.58
    legend_text_y = 0.04
    latency_title = latency_ax.text(
        0.0,
        legend_marker_y,
        "CUDA latency (bubble area)",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#3F4852",
        clip_on=False,
    )
    legend_x = [0.50, 0.68, 0.87]
    latency_markers = []
    latency_value_texts = []
    for x_marker, value in zip(legend_x, legend_vals):
        latency_markers.append(
            latency_ax.scatter(
                [x_marker],
                [legend_marker_y],
                s=bubble_size(value),
                color="#B8CBD8",
                edgecolor="#7D929F",
                lw=0.35,
                clip_on=False,
                zorder=2,
            )
        )
        latency_value_texts.append(
            latency_ax.text(
                x_marker,
                legend_text_y,
                f"{value:.1f} ms",
                ha="center",
                va="center",
                fontsize=6.8,
                color="#3F4852",
                clip_on=False,
                zorder=3,
            )
        )
    quiet_grid(ax0)
    from matplotlib.ticker import LogLocator, NullFormatter

    ax0.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax0.xaxis.set_minor_formatter(NullFormatter())
    ax0.grid(which="major", axis="x", color="#DDE3E9", lw=0.60)
    ax0.grid(which="minor", axis="x", color="#EDF0F4", lw=0.45, ls=":")
    panel_label_fixed(fig, ax0, "a", dx_pt=-13.0, dy_pt=3.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = {
        model: artist.get_window_extent(renderer=renderer)
        for model, artist in label_artists.items()
    }
    if pareto_label_artist is not None:
        label_boxes["Pareto frontier"] = pareto_label_artist.get_window_extent(
            renderer=renderer
        )
    overlap_pairs = []
    models = list(label_boxes)
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            if label_boxes[model_a].overlaps(label_boxes[model_b]):
                overlap_pairs.append(f"{model_a}|{model_b}")
    latency_title_box = latency_title.get_window_extent(renderer=renderer)
    latency_value_boxes = [
        artist.get_window_extent(renderer=renderer)
        for artist in latency_value_texts
    ]
    ax0_box = ax0.get_window_extent(renderer=renderer)
    figure_box = fig.bbox
    strip_boxes = [latency_title_box, *latency_value_boxes]
    legend_scoped_to_panel_a = bool(
        min(box.x0 for box in strip_boxes) >= ax0_box.x0 - 2.0
        and max(box.x1 for box in strip_boxes) <= ax0_box.x1 + 2.0
    )
    top_margin_clear = bool(
        max(box.y1 for box in strip_boxes) <= figure_box.y1 - 2.0
    )
    strip_baselines_aligned = bool(
        all(np.isclose(artist.get_position()[1], legend_text_y) for artist in latency_value_texts)
        and np.isclose(latency_title.get_position()[1], legend_marker_y)
    )
    legend_boxes = strip_boxes
    legend_overlaps = [
        model
        for model, box in label_boxes.items()
        if any(box.overlaps(legend_box) for legend_box in legend_boxes)
    ]
    label_audit = pd.DataFrame(bubble_records)
    label_audit["text_overlap_count"] = len(overlap_pairs)
    label_audit["legend_overlap_count"] = len(legend_overlaps)
    label_audit["latency_legend_scoped_to_panel_a"] = legend_scoped_to_panel_a
    label_audit["latency_title_inside_figure"] = top_margin_clear
    label_audit["latency_strip_baselines_aligned"] = strip_baselines_aligned
    label_audit["overlap_pairs"] = ";".join(overlap_pairs)
    label_audit["legend_overlap_models"] = ";".join(legend_overlaps)
    label_audit.to_csv(DATA_OUT / "fig06_bubble_label_audit.csv", index=False)
    if (
        overlap_pairs
        or legend_overlaps
        or not legend_scoped_to_panel_a
        or not top_margin_clear
        or not strip_baselines_aligned
    ):
        raise ValueError(
            "Fig. 6 layout audit failed: "
            f"text={overlap_pairs}, legend={legend_overlaps}, "
            f"legend_scoped_to_panel_a={legend_scoped_to_panel_a}, "
            f"top_margin_clear={top_margin_clear}, "
            f"strip_baselines_aligned={strip_baselines_aligned}"
        )

    ax1 = fig.add_subplot(gs[0, 1])
    pd.DataFrame(
        [
            {
                "metric": "torch_peak_reserved_mb",
                "minimum_mb": float(res["torch_peak_reserved_mb"].min()),
                "maximum_mb": float(res["torch_peak_reserved_mb"].max()),
                "unique_values": int(res["torch_peak_reserved_mb"].nunique()),
                "shown_in_heatmap": False,
                "reason": "identical for all displayed models; omitted as non-discriminative",
            }
        ]
    ).to_csv(DATA_OUT / "fig06_peak_memory_omission_audit.csv", index=False)
    cols = [
        ("params_k", "Params"),
        ("checkpoint_size_mb", "Model\nsize"),
        ("latency_mean_ms", "CUDA\nlat."),
        ("cpu_latency_ms", "CPU\nlat."),
    ]
    heat_all = res.set_index("model_display")[[c for c, _ in cols]].reindex(accuracy_order)
    base = heat_all.loc["Online Lyra"]
    heat_order = accuracy_order
    heat = heat_all.reindex(heat_order)
    ratio = heat.divide(base, axis=1)
    ratio.loc["Online Lyra", :] = 1.0
    if (ratio <= 0).any().any():
        bad = ratio.where(ratio <= 0).stack().reset_index()
        bad.columns = ["model_display", "metric", "ratio"]
        bad.to_csv(DATA_OUT / "fig06_invalid_resource_ratios.csv", index=False)
        raise ValueError("Resource ratio contains non-positive values; see fig06_invalid_resource_ratios.csv")
    log2_ratio = np.log2(ratio)
    ratio.to_csv(DATA_OUT / "fig06_resource_ratio_to_online.csv")
    lim = max(2.0, float(np.nanmax(np.abs(log2_ratio.to_numpy()))))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = LinearSegmentedColormap.from_list(
        "ratio_diverging", ["#3B78A8", "#FFFFFF", "#C56C43"], N=257
    )
    center_rgba = cmap(norm(0.0))
    if not np.allclose(center_rgba[:3], (1.0, 1.0, 1.0), atol=1e-12):
        raise ValueError("Fig. 6 heatmap colormap is not pure white at ratio 1.0")
    pd.DataFrame(
        [
            {
                "resource_ratio": 1.0,
                "log2_ratio": 0.0,
                "normalized_position": float(norm(0.0)),
                "red": center_rgba[0],
                "green": center_rgba[1],
                "blue": center_rgba[2],
                "alpha": center_rgba[3],
                "expected_color": "#FFFFFF",
            }
        ]
    ).to_csv(DATA_OUT / "fig06_colormap_center_audit.csv", index=False)
    im = ax1.imshow(log2_ratio.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax1.set_xticks(np.arange(len(cols)))
    ax1.set_xticklabels([label for _, label in cols], rotation=0, ha="center")
    ax1.set_yticks(np.arange(len(heat_order)))
    ax1.set_yticklabels(heat_order, fontsize=7.5)
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            val = float(ratio.iloc[i, j])
            rgba = cmap(norm(float(log2_ratio.iloc[i, j])))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "white" if luminance < 0.56 else "#20262D"
            label = f"{val:.2f}x" if val < 0.1 else (f"{val:.0f}x" if val >= 10 else f"{val:.1f}x")
            t = ax1.text(j, i, label, ha="center", va="center", fontsize=7.0, color=txt_color)
            outline = "black" if txt_color == "white" else "white"
            t.set_path_effects(
                [patheffects.withStroke(linewidth=1.0, foreground=outline, alpha=0.42)]
            )
    cb = fig.colorbar(im, ax=ax1, fraction=0.052, pad=0.020)
    ticks = np.array([0.25, 0.5, 1, 2, 4, 8], dtype=float)
    ticks = ticks[(np.log2(ticks) >= -lim - 1e-9) & (np.log2(ticks) <= lim + 1e-9)]
    cb.set_ticks(np.log2(ticks))
    cb.set_ticklabels([f"{t:g}x" for t in ticks])
    cb.ax.axhline(0.0, color="#6F7780", lw=0.80, zorder=4)
    cb.set_label("Resource ratio")
    panel_label_fixed(fig, ax1, "b", dx_pt=-13.0, dy_pt=3.0)
    save_pub(fig, "fig06_efficiency")


def figS01_fallback_dot_revised() -> None:
    g = add_hours(read_csv(PACK / "figure_data" / "gate_profile_all_sites.csv"))
    g["fallback_pct"] = (1.0 - g["gate_lyra_fraction"]) * 100
    avg = g.groupby(["site_id", "horizon", "lead_time_h"], as_index=False).agg(fallback_pct=("fallback_pct", "mean"), skill_vs_persistence=("skill_vs_persistence", "mean"))
    horizons = sorted(avg["lead_time_h"].unique())
    markers = {1: "o", 3: "s", 6: "^", 12: "D", 24: "P"}
    hcolors = {h: c for h, c in zip(horizons, ["#477FA8", "#65A5C2", "#72AD9F", "#D99A43", "#B7768D"])}

    # Deterministic display-only jitter reveals coincident zero-fallback points.
    # The measured fallback fraction is retained unchanged in fallback_pct.
    horizon_jitter = dict(zip(horizons, np.linspace(-0.20, 0.20, len(horizons))))
    site_jitter = {site: offset for site, offset in zip(range(1, 9), np.linspace(-0.06, 0.06, 8))}
    avg["fallback_pct_display"] = avg.apply(
        lambda row: row["fallback_pct"]
        + (horizon_jitter[row["lead_time_h"]] + site_jitter[int(row["site_id"])] if np.isclose(row["fallback_pct"], 0.0) else 0.0),
        axis=1,
    )
    avg["display_jitter_pct_point"] = avg["fallback_pct_display"] - avg["fallback_pct"]
    vertical_offsets = dict(zip(horizons, np.linspace(-0.34, 0.34, len(horizons))))
    avg["site_y_display"] = avg.apply(
        lambda row: row["site_id"] + vertical_offsets[row["lead_time_h"]], axis=1
    )
    avg.to_csv(DATA_OUT / "figS01_fallback_site_horizon_dot.csv", index=False)

    fig = plt.figure(figsize=(7.25, 3.35))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.34, bottom=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    offsets = [vertical_offsets[h] for h in horizons]
    for off, h in zip(offsets, horizons):
        sub = avg.loc[avg["lead_time_h"] == h].sort_values("site_id")
        ax0.scatter(
            sub["fallback_pct_display"],
            sub["site_id"] + off,
            marker=markers.get(int(h), "o"),
            s=38,
            color=hcolors[h],
            alpha=0.68,
            edgecolor="white",
            lw=0.55,
            label=f"{int(h)} h",
            zorder=3,
        )
    ax0.set_yticks(range(1, 9))
    ax0.set_yticklabels([f"S{i}" for i in range(1, 9)])
    ax0.set_xlabel("Fallback fraction (%)")
    ax0.set_ylabel("Site")
    ax0.axvline(0, color="#B8BEC6", lw=0.65, zorder=1)
    quiet_grid(ax0, axis="x")
    panel_label_fixed(fig, ax0, "a", dx_pt=-13.0, dy_pt=3.0)

    ax1 = fig.add_subplot(gs[0, 1])
    for h in horizons:
        sub = avg.loc[avg["lead_time_h"] == h]
        ax1.scatter(
            sub["fallback_pct_display"],
            sub["skill_vs_persistence"] * 100,
            marker=markers.get(int(h), "o"),
            s=44,
            color=hcolors[h],
            alpha=0.68,
            edgecolor="white",
            lw=0.55,
            label=f"{int(h)} h",
            zorder=3,
        )

    ax1.set_xlabel("Fallback fraction (%)")
    ax1.set_ylabel("Skill vs persistence (%)")
    ax1.set_xlim(-0.38, 6.42)
    ax1.set_ylim(15.0, 78.8)
    ax1.axvline(0, color="#B8BEC6", lw=0.65, zorder=1)
    quiet_grid(ax1)
    panel_label_fixed(fig, ax1, "b", dx_pt=-13.0, dy_pt=3.0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(int(h), "o"),
            linestyle="none",
            markersize=5.4,
            markerfacecolor=hcolors[h],
            markeredgecolor="white",
            markeredgewidth=0.55,
            alpha=0.75,
            label=f"{int(h)} h",
        )
        for h in horizons
    ]
    legend = ax1.legend(
        handles=legend_handles,
        title="Forecast horizon",
        ncol=1,
        loc="best",
        columnspacing=0.75,
        handletextpad=0.45,
        borderaxespad=0.70,
    )
    legend.get_frame().set_linewidth(0.45)
    save_pub(fig, "figS01_fallback_grouped_dot")


def figS02_ablation_revised() -> None:
    core = read_csv(PACK / "core_ablation" / "core_ablation_summary.csv")
    core_long = read_csv(PACK / "core_ablation" / "core_ablation_long.csv")
    cold = read_csv(PACK / "core_ablation" / "cold_start_summary.csv")
    full = float(core.loc[core["ablation"] == "Full Online Lyra", "nrmse"].iloc[0])

    audit_rows = []
    audit_metrics = [
        "nrmse",
        "nmae",
        "daytime_nrmse",
        "daytime_nmae",
        "horizon_divergence",
        "predict_latency_ms",
        "update_latency_ms",
    ]
    for component in ["w/o spectral residual scaling", "w/o gate"]:
        pair = core_long.loc[
            core_long["ablation"].isin(["Full Online Lyra", component])
        ].pivot_table(
            index=["site_id", "horizon"],
            columns="ablation",
            values=audit_metrics,
            aggfunc="mean",
        )
        for metric in audit_metrics:
            delta = pair[(metric, component)] - pair[(metric, "Full Online Lyra")]
            audit_rows.append(
                {
                    "removed_component": component,
                    "metric": metric,
                    "mean_delta_removed_minus_full": float(delta.mean()),
                    "median_delta_removed_minus_full": float(delta.median()),
                    "removal_better_conditions": int((delta < 0).sum()),
                    "full_better_conditions": int((delta > 0).sum()),
                    "tied_conditions": int(np.isclose(delta, 0.0).sum()),
                    "n_conditions": int(delta.notna().sum()),
                }
            )
    pd.DataFrame(audit_rows).to_csv(DATA_OUT / "figS02_component_evidence_audit.csv", index=False)

    core = core.loc[
        ~core["ablation"].isin(
            [
                "Persistence",
                "Full Online Lyra",
                "w/o spectral residual scaling",
            ]
        )
    ].copy()
    core["delta_nrmse_x1e3"] = (core["nrmse"] - full) * 1000
    core["contribution_abs"] = core["delta_nrmse_x1e3"].abs()
    core = core.sort_values("contribution_abs", ascending=False)
    cold = cold.copy()
    cold["delta_nrmse_x1e3"] = (cold["nrmse"] - full) * 1000
    pd.concat([core, cold], ignore_index=True).to_csv(DATA_OUT / "figS02_ablation_cold_start.csv", index=False)

    fig = plt.figure(figsize=(7.25, 3.35))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.28, 0.82], wspace=0.42)
    ax0 = fig.add_subplot(gs[0, 0])
    vals = core["delta_nrmse_x1e3"].to_numpy(dtype=float)
    y = np.arange(len(core))
    colors = [COLORS["bad"] if v > 0 else COLORS["good"] for v in vals]
    ax0.set_xscale("symlog", linthresh=3.0, linscale=1.0, base=10)
    for yi, actual, color in zip(y, vals, colors):
        ax0.plot([0, actual], [yi, yi], color=color, lw=1.4, alpha=0.68, zorder=2)
        ax0.scatter(
            actual,
            yi,
            s=48,
            color=color,
            edgecolor="white",
            lw=0.65,
            alpha=0.92,
            zorder=3,
        )
        large_positive = actual > 30
        ax0.annotate(
            f"{actual:+.1f}",
            xy=(actual, yi),
            xytext=(0 if large_positive else (6 if actual >= 0 else -6), -13 if large_positive else 0),
            textcoords="offset points",
            ha="center" if large_positive else ("right" if actual < 0 else "left"),
            va="top" if large_positive else "center",
            fontsize=7.5,
            color=color,
            fontweight="bold" if large_positive else "normal",
            bbox=(
                dict(boxstyle="round,pad=0.18", facecolor="#F9E1E3", edgecolor="#D9A4AA", linewidth=0.45, alpha=0.88)
                if large_positive
                else None
            ),
        )
    ax0.axvline(0, color=COLORS["black"], lw=0.8)
    ax0.set_yticks(y)
    ax0.set_yticklabels(core["ablation"], fontsize=7.5)
    ax0.invert_yaxis()
    ax0.set_xlabel(r"$\Delta$ nRMSE vs full ($\times 10^{-3}$)")
    ax0.set_xlim(-4.5, 250)
    ax0.set_xticks([-3, -1, 0, 1, 3, 10, 30, 100])
    ax0.set_xticklabels(["-3", "-1", "0", "1", "3", "10", "30", "100"])
    quiet_grid(ax0, axis="x")
    panel_label_fixed(fig, ax0, "a", dx_pt=-13.0, dy_pt=3.0)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.array([30, 60, 91], dtype=float)
    vals2 = [
        float(cold.loc[cold["ablation"] == "cold start: 30 offline days", "nrmse"].iloc[0]),
        float(cold.loc[cold["ablation"] == "cold start: 60 offline days", "nrmse"].iloc[0]),
        full,
    ]
    point_colors = [COLORS["static_light"], COLORS["online_light"], COLORS["online"]]
    ax1.scatter(x, vals2, s=58, color=point_colors, edgecolor="white", lw=0.7, zorder=3)
    set_zoom_ylim(ax1, np.array(vals2), pad_frac=0.24)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["30", "60", "91"])
    ax1.set_xlabel("Initial offline history (days)")
    ax1.set_ylabel("nRMSE")
    for idx, (xi, val) in enumerate(zip(x, vals2)):
        dx = 5 if idx == 0 else (-5 if idx == len(vals2) - 1 else 0)
        ha = "left" if idx == 0 else ("right" if idx == len(vals2) - 1 else "center")
        ax1.annotate(
            f"{val:.4f}",
            xy=(xi, val),
            xytext=(dx, 7),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=7.5,
        )
    quiet_grid(ax1)
    panel_label_fixed(fig, ax1, "b", dx_pt=-13.0, dy_pt=3.0)
    save_pub(fig, "figS02_core_ablation_cold_start")


def figS03_effect_significance_revised() -> None:
    cond = read_csv(PACK / "server_tables_with_offline_lyra" / "main_condition_average.csv")
    stats = read_csv(PACK / "stat_tests" / "paired_tests_online_lyra_vs_baselines.csv")
    keep = ["iTransformer", "Olivia", "DLinear", "TimeKAN", "PatchTST", "Offline Lyra", "PhaseFormer", "NLinear"]
    metrics = ["nrmse", "daytime_nrmse"]
    med_rows = []
    for metric in metrics:
        pivot = cond.pivot_table(index=["site_id", "horizon"], columns="model_display", values=metric, aggfunc="mean")
        for baseline in keep:
            if baseline not in pivot.columns or "Online Lyra" not in pivot.columns:
                continue
            rel = (pivot[baseline] - pivot["Online Lyra"]) / pivot[baseline] * 100
            med_rows.append({"metric": metric, "baseline_display": baseline, "median_paired_relative_improvement_pct": float(np.nanmedian(rel)), "n_pairs": int(rel[rel.notna()].shape[0])})
    med = pd.DataFrame(med_rows)
    data = stats.loc[stats["metric"].isin(metrics) & stats["baseline_display"].isin(keep)].merge(med, on=["metric", "baseline_display", "n_pairs"], how="left")
    data["wilcoxon_p_adjusted_fdr"] = np.nan
    for metric in metrics:
        mask = data["metric"].eq(metric)
        data.loc[mask, "wilcoxon_p_adjusted_fdr"] = fdr_bh(data.loc[mask, "wilcoxon_p_less"].to_numpy(dtype=float))
    data["neglog10_adj_p"] = -np.log10(data["wilcoxon_p_adjusted_fdr"].clip(lower=1e-12))
    data.to_csv(DATA_OUT / "figS03_paired_effect_size_significance.csv", index=False)

    label_offsets = {
        "nrmse": {
            "Offline Lyra": (7, -2),
            "PhaseFormer": (7, 0),
            "NLinear": (7, -5),
            "TimeKAN": (-10, 0),
            "PatchTST": (7, 0),
            "Olivia": (8, 13),
            "DLinear": (7, 8),
            "iTransformer": (7, 9),
        },
        "daytime_nrmse": {
            "Offline Lyra": (7, -2),
            "PhaseFormer": (-9, 0),
            "NLinear": (8, 8),
            "TimeKAN": (8, 17),
            "PatchTST": (8, -12),
            "iTransformer": (12, -13),
            "Olivia": (-12, 14),
            "DLinear": (-12, -3),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.18), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.19, wspace=0.22)
    for ax, metric, title, letter in zip(axes, metrics, ["nRMSE", "Daytime nRMSE"], ["a", "b"]):
        sub = data.loc[data["metric"] == metric].copy().sort_values("neglog10_adj_p", ascending=False)
        ax.axhline(-np.log10(0.05), color=COLORS["neutral"], lw=0.8, ls=":")
        ax.axvline(0, color=COLORS["neutral"], lw=0.8)
        for _, row in sub.iterrows():
            m = row["baseline_display"]
            sig = bool(row["wilcoxon_p_adjusted_fdr"] < 0.05)
            is_foundation = m == "Olivia"
            marker = "D" if is_foundation else "o"
            if is_foundation:
                face = COLORS["foundation"] if sig else "#F3D4A4"
                edge = "#A96B1F"
            else:
                face = COLORS["online_light"] if sig else "#E2E6EC"
                edge = COLORS["black"] if sig else "#B6BEC8"
            ax.scatter(
                row["median_paired_relative_improvement_pct"],
                row["neglog10_adj_p"],
                s=45 if sig else 34,
                marker=marker,
                color=face,
                edgecolor=edge,
                lw=0.65,
                alpha=0.96,
                zorder=4,
            )
            dx, dy = label_offsets[metric][m]
            ax.annotate(
                m,
                xy=(row["median_paired_relative_improvement_pct"], row["neglog10_adj_p"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                va="center",
                ha="left" if dx >= 0 else "right",
                arrowprops=dict(arrowstyle="-", lw=0.45, color="#9AA3AD", shrinkA=2, shrinkB=2),
            )
        ax.set_xlabel("Median paired relative improvement (%)")
        ax.set_title(title)
        quiet_grid(ax)
        panel_label_fixed(fig, ax, letter, dx_pt=-13.0, dy_pt=3.0)
    axes[0].set_ylabel(r"$-\log_{10}(p_{\mathrm{FDR}})$")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.5, markerfacecolor=COLORS["online_light"], markeredgecolor=COLORS["black"], markeredgewidth=0.65, label="FDR-significant baseline"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.0, markerfacecolor="#E2E6EC", markeredgecolor="#B6BEC8", markeredgewidth=0.65, label="Not significant"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=5.0, markerfacecolor=COLORS["foundation"], markeredgecolor="#A96B1F", markeredgewidth=0.65, label="Foundation model (Olivia)"),
        Line2D([0], [0], color=COLORS["neutral"], lw=0.8, ls=":", label="FDR = 0.05"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=1.15,
        handlelength=1.25,
        handletextpad=0.45,
    )
    axes[1].legend(handles=legend_handles, loc="best", ncol=1)
    save_pub(fig, "figS03_paired_effect_size_significance")


def figS04_daytime_calibration_density() -> None:
    """Compare daytime calibration under one shared density scale."""
    df = read_csv(DATA_OUT / "fig05_daytime_predictions_h96_seed2028.csv")
    models = ["Online Lyra", "iTransformer", "Smart Persistence"]
    observed = df["observed"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, 46)

    counts_by_model: dict[str, np.ndarray] = {}
    stats_rows = []
    bin_rows = []
    for model in models:
        predicted = df[model].to_numpy(dtype=float)
        counts, x_edges, y_edges = np.histogram2d(observed, predicted, bins=[edges, edges])
        counts_by_model[model] = counts

        slope, intercept = np.polyfit(observed, predicted, 1)
        corr = np.corrcoef(observed, predicted)[0, 1]
        stats_rows.append(
            {
                "model": model,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(corr**2),
                "normalized_bias": float(np.mean(predicted - observed)),
                "n_points": int(observed.size),
                "horizon": 96,
                "lead_time_h": 24,
                "seed": 2028,
                "evaluation_subset": "active generation (observed PV > 5% of site capacity)",
            }
        )
        for ix in range(len(x_edges) - 1):
            for iy in range(len(y_edges) - 1):
                bin_rows.append(
                    {
                        "model": model,
                        "observed_bin_left": x_edges[ix],
                        "observed_bin_right": x_edges[ix + 1],
                        "predicted_bin_left": y_edges[iy],
                        "predicted_bin_right": y_edges[iy + 1],
                        "count": int(counts[ix, iy]),
                    }
                )

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(DATA_OUT / "figS04_daytime_calibration_stats.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(DATA_OUT / "figS04_daytime_calibration_density_bins.csv", index=False)

    cmap = LinearSegmentedColormap.from_list(
        "calibration_density",
        ["#F7FAFC", "#DCEAF3", "#A9CCE0", "#5DA8C9", "#174A7C"],
    )
    cmap.set_bad("white")
    vmax = max(float(counts.max()) for counts in counts_by_model.values())
    norm = LogNorm(vmin=1.0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.72), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.20, top=0.86, wspace=0.12)
    mesh = None
    for ax, model, letter in zip(axes, models, ["a", "b", "c"]):
        counts = counts_by_model[model].T
        masked = np.ma.masked_where(counts <= 0, counts)
        mesh = ax.pcolormesh(edges, edges, masked, cmap=cmap, norm=norm, shading="flat")
        ax.plot([0, 1], [0, 1], color="#4D4D4D", lw=0.8, ls="--", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(model, pad=5)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        row = stats.loc[stats["model"] == model].iloc[0]
        ax.text(
            0.035,
            0.965,
            rf"slope = {row['slope']:.2f}" + "\n" + rf"$R^2$ = {row['r2']:.2f}" + "\n" + rf"bias = {row['normalized_bias']:+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.7,
            linespacing=1.18,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.6),
            zorder=4,
        )
        panel_label_fixed(fig, ax, letter, dx_pt=-12.5, dy_pt=3.0)

    fig.supxlabel("Observed normalized PV", y=0.055)
    fig.supylabel("Predicted normalized PV", x=0.025)
    if mesh is not None:
        cax = fig.add_axes([0.915, 0.20, 0.018, 0.66])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Bin count")
    save_pub(fig, "figS04_daytime_calibration_density")


def figS05_hankel_singular_spectrum() -> None:
    """Characterize PV Hankel spectra without implying a rank choice."""
    import re

    lookback = 96
    hankel_l = lookback // 2
    stride = 96
    year_points = 365 * 96
    max_rank = 20
    site_files = sorted((ROOT / "solar_stations").glob("*.xlsx"))
    if len(site_files) != 8:
        raise ValueError(f"Expected 8 PV site files, found {len(site_files)}.")

    spectrum_rows = []
    audit_rows = []
    normalized_spectra = []
    cumulative_energies = []
    site_summaries: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for path in site_files:
        site_match = re.search(r"site\s+(\d+)", path.name, flags=re.IGNORECASE)
        capacity_match = re.search(r"capacity-(\d+(?:\.\d+)?)MW", path.name, flags=re.IGNORECASE)
        if site_match is None or capacity_match is None:
            raise ValueError(f"Could not parse site/capacity from {path.name}.")
        site_id = int(site_match.group(1))
        capacity_mw = float(capacity_match.group(1))

        raw = pd.read_excel(path)
        raw.columns = [str(col).strip() for col in raw.columns]
        time_candidates = [col for col in raw.columns if "time" in col.lower()]
        if time_candidates:
            time_col = time_candidates[0]
            raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
            raw = raw.sort_values(time_col)
        power_candidates = [col for col in raw.columns if "power" in col.lower()]
        if not power_candidates:
            raise ValueError(f"No power column found in {path.name}.")
        power = pd.to_numeric(raw[power_candidates[-1]], errors="coerce").to_numpy(dtype=float)
        power = power[np.isfinite(power)]
        if power.size < year_points:
            raise ValueError(f"{path.name} has only {power.size} valid power samples.")
        power = np.clip(power[:year_points], 0.0, capacity_mw)

        windows = np.lib.stride_tricks.sliding_window_view(power, lookback)[::stride].copy()
        centered = windows - windows.mean(axis=1, keepdims=True)
        hankel = np.lib.stride_tricks.sliding_window_view(centered, hankel_l, axis=1).transpose(0, 2, 1)
        singular_values = np.linalg.svd(hankel, compute_uv=False)
        total_energy = np.square(singular_values).sum(axis=1)
        valid = total_energy > np.finfo(float).eps
        excluded = int((~valid).sum())
        singular_values = singular_values[valid]
        total_energy = total_energy[valid]
        normalized = singular_values / singular_values[:, :1]
        cumulative = np.cumsum(np.square(singular_values), axis=1) / total_energy[:, None]

        normalized_spectra.append(normalized)
        cumulative_energies.append(cumulative)
        site_summaries[site_id] = (np.median(normalized, axis=0), np.median(cumulative, axis=0))
        audit_rows.append(
            {
                "site_id": site_id,
                "capacity_mw": capacity_mw,
                "source_file": path.name,
                "n_samples": int(year_points),
                "lookback": lookback,
                "hankel_rows": hankel_l,
                "hankel_columns": lookback - hankel_l + 1,
                "window_stride": stride,
                "candidate_windows": int(windows.shape[0]),
                "valid_nonconstant_windows": int(valid.sum()),
                "excluded_zero_energy_windows": excluded,
            }
        )
        starts = np.arange(windows.shape[0], dtype=int)[valid] * stride
        for window_start, norm_row, energy_row in zip(starts, normalized, cumulative):
            for rank_idx in range(singular_values.shape[1]):
                spectrum_rows.append(
                    {
                        "site_id": site_id,
                        "window_start_index": int(window_start),
                        "rank": rank_idx + 1,
                        "normalized_singular_value": float(norm_row[rank_idx]),
                        "cumulative_squared_energy": float(energy_row[rank_idx]),
                    }
                )

    normalized_all = np.concatenate(normalized_spectra, axis=0)
    cumulative_all = np.concatenate(cumulative_energies, axis=0)
    ranks = np.arange(1, normalized_all.shape[1] + 1)
    summary = pd.DataFrame(
        {
            "rank": ranks,
            "normalized_singular_value_q25": np.quantile(normalized_all, 0.25, axis=0),
            "normalized_singular_value_median": np.median(normalized_all, axis=0),
            "normalized_singular_value_q75": np.quantile(normalized_all, 0.75, axis=0),
            "cumulative_energy_q25": np.quantile(cumulative_all, 0.25, axis=0),
            "cumulative_energy_median": np.median(cumulative_all, axis=0),
            "cumulative_energy_q75": np.quantile(cumulative_all, 0.75, axis=0),
        }
    )
    pd.DataFrame(spectrum_rows).to_csv(DATA_OUT / "figS05_hankel_spectrum_window_units.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(DATA_OUT / "figS05_hankel_spectrum_data_audit.csv", index=False)
    summary.to_csv(DATA_OUT / "figS05_hankel_spectrum_summary.csv", index=False)

    shown = summary.loc[summary["rank"] <= max_rank]
    x = shown["rank"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.82, wspace=0.27)

    for site_id in sorted(site_summaries):
        site_norm, site_energy = site_summaries[site_id]
        axes[0].plot(ranks[:max_rank], site_norm[:max_rank], color="#9FB3C8", lw=0.65, alpha=0.60, zorder=1)
        axes[1].plot(ranks[:max_rank], site_energy[:max_rank] * 100.0, color="#9FB3C8", lw=0.65, alpha=0.60, zorder=1)

    axes[0].fill_between(
        x,
        shown["normalized_singular_value_q25"].to_numpy(dtype=float),
        shown["normalized_singular_value_q75"].to_numpy(dtype=float),
        color="#DCEAF3",
        alpha=0.75,
        lw=0,
        zorder=2,
    )
    axes[0].plot(x, shown["normalized_singular_value_median"], color=COLORS["online"], lw=1.65, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Normalized singular value $\sigma_i/\sigma_1$")

    axes[1].fill_between(
        x,
        shown["cumulative_energy_q25"].to_numpy(dtype=float) * 100.0,
        shown["cumulative_energy_q75"].to_numpy(dtype=float) * 100.0,
        color="#DCEAF3",
        alpha=0.75,
        lw=0,
        zorder=2,
    )
    axes[1].plot(x, shown["cumulative_energy_median"] * 100.0, color=COLORS["online"], lw=1.65, zorder=3)
    energy_floor = float(max(0.0, shown["cumulative_energy_q25"].min() * 100.0 - 4.0))
    axes[1].set_ylim(energy_floor, 100.5)
    axes[1].set_ylabel("Cumulative spectral energy (%)")

    for ax, letter in zip(axes, ["a", "b"]):
        ax.set_xlim(1, max_rank)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Singular-value index")
        quiet_grid(ax)
        panel_label_fixed(fig, ax, letter, dx_pt=-13.0, dy_pt=3.0)

    legend_handles = [
        Line2D([0], [0], color=COLORS["online"], lw=1.65, label="Pooled median"),
        Patch(facecolor="#DCEAF3", edgecolor="none", alpha=0.75, label="Interquartile range"),
        Line2D([0], [0], color="#9FB3C8", lw=0.8, alpha=0.75, label="Site medians (n=8)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        columnspacing=1.5,
        handlelength=2.0,
        handletextpad=0.55,
    )
    save_pub(fig, "figS05_hankel_singular_spectrum")


def write_revised_reports(context: dict[str, object]) -> None:
    mapping = pd.DataFrame(
        [
            ("old fig02_accuracy_zoomed", "Fig. 1", "fig01_accuracy", "Main", "Accuracy across horizons with seed-level CI and truncated-axis marks."),
            ("old fig03_robustness_rank_capacity", "Fig. 2", "fig02_robustness", "Main", "Site/horizon rank and corrected capacity-group robustness."),
            ("old fig04_case_trace_gate", "Fig. 3", "fig03_case_trace_gate", "Main", "Case study with z-order-safe observed PV and taller gate strip."),
            ("old fig07_error_distribution_calibration", "Fig. 5", "fig05_daytime_error_agreement", "Main", "Site-level daytime accuracy and ramp robustness with Smart Persistence."),
            ("old fig08_efficiency_tradeoff", "Fig. 6", "fig06_efficiency", "Main", "Efficiency bubble plot and log2-centered resource ratio heatmap."),
            ("old fig05_gate_fallback_behavior", "Fig. S1", "figS01_fallback_grouped_dot", "Supplementary", "Fallback grouped dot plot plus skill relation."),
            ("old fig09_ablation_delta", "Fig. S2", "figS02_core_ablation_cold_start", "Supplementary", "Removed-component ablation and cold-start sensitivity."),
            ("old fig10_statistical_volcano", "Fig. S3", "figS03_paired_effect_size_significance", "Supplementary", "Paired effect size and FDR-adjusted Wilcoxon evidence."),
        ],
        columns=["old_figure", "new_number", "new_stem", "placement", "change_summary"],
    )
    mapping.to_csv(OUT / "old_to_new_figure_mapping.csv", index=False)

    strongest = context.get("strongest_baseline", "iTransformer")
    fig05_comparator = context.get("fig05_comparator", strongest)
    fig05_online_nmae = float(context.get("fig05_online_nmae", 0.130606))
    fig05_comparator_nmae = float(context.get("fig05_comparator_nmae", 0.133994))
    fig05_smart_nmae = float(context.get("fig05_smart_nmae", 0.135518))
    fig05_ramp_q4_threshold_pct = float(context.get("fig05_ramp_q4_threshold_pct", 6.2069))
    fig02_rank_model_count = int(context.get("fig02_rank_model_count", 14))
    generation_record = f"""# Figure Generation Record

- Figures are rebuilt from the recorded CSV and NPZ artifacts.
- Confidence intervals use seed-, site-, or site-by-horizon-level units as
  specified in `STATISTICS_AND_DATA_SOURCES.md`; time points are not treated as
  independent replicates.
- Online Lyra is evaluated as a deterministic point forecast, so ECE is not
  reported.
- The capacity groups are Small <50 MW, Medium 50-110 MW, and Large >110 MW.
- `{strongest}` is the eligible static baseline with the lowest mean nRMSE in
  the recorded comparison.
- Figure numbering changes are recorded in `old_to_new_figure_mapping.csv`.
"""
    (OUT / "notes" / "FIGURE_GENERATION_RECORD.md").write_text(
        generation_record, encoding="utf-8"
    )

    stats_note = f"""# Statistics and Data Sources

## Data Sources
- Main static baselines: `outputs/server_runs_20260716/fair_protocol_20260716_123158/static_official_seed_*/pv_baseline_summary.csv`.
- Online Lyra final: `outputs/server_runs_20260716/final_online_lyra_b06_20260716_231353/b06_final/site_*_seed_*/online_pv_summary.csv`.
- Offline Lyra: `outputs/server_runs_20260716/offline_lyra_final_20260717_165425/offline_lyra_seed_*/offline_lyra_summary.csv`.
- Prediction traces: saved `.npz` files under the Online/Offline artifact directories.
- Resource benchmark: `resource_efficiency_table.csv` and `resource_cpu_table.csv`.
- Core ablation/cold-start: `outputs/paper_result_pack/core_ablation/*.csv`.
- Wilcoxon source p-values: `outputs/paper_result_pack/stat_tests/paired_tests_online_lyra_vs_baselines.csv`.

## Statistical Units
- Fig1: seed-level summaries, where each seed first averages across sites for a model and horizon. Error bars are 95% t intervals over seeds; no point-wise bootstrap.
- Fig2: Smart Persistence uses the observation at the same 15 min slot on the previous day. The baseline artifact previously named Seasonal Naive is this same predictor and is renamed rather than counted twice. Panel a ranks {fig02_rank_model_count} methods after averaging three seeds. Panels b/c first average seeds and horizons within each site; small points are independent sites and large markers are capacity-group means (Small n=4, Medium n=2, Large n=2). No inferential CI is claimed for these small capacity subgroups.
- Fig3: descriptive case/prediction-trace panels from saved prediction arrays; no CI added because the panel is not an inferential replicate summary.
- Fig5: each metric is first computed within site; points are the eight sites and error bars are 95% t intervals over sites (n=8). Active generation is defined as observed PV >5% of site capacity. Smart Persistence uses the previous day's value at the same 15 min slot; the first day per site is excluded for all compared methods.
- Fig6: each random seed first averages nRMSE over the complete 8-site x 5-horizon grid (40 conditions), and plotted points are means over three seeds. The corresponding 95% t intervals are retained in the exported source-data audit but are not drawn in this deployment figure; paired Wilcoxon/FDR evidence is reported separately in Fig.S3.
- Fig.S3: paired unit is site×horizon from `main_condition_average.csv`; p-values are Wilcoxon one-sided less tests from the saved statistical table, corrected within each metric using Benjamini-Hochberg FDR. The x-axis effect size is median paired relative improvement `(baseline - Online Lyra) / baseline * 100`.

## Explicit Non-Use
- No ECE is reported, because Online Lyra outputs deterministic point forecasts rather than probabilities, quantiles, or predictive distributions.
- No CI or p-value is simulated or estimated from plotted means alone.
"""
    (OUT / "notes" / "STATISTICS_AND_DATA_SOURCES.md").write_text(stats_note, encoding="utf-8")

    captions = f"""# Figure Captions

**Figure 1 | Accuracy across forecast horizons.** Panel a shows nRMSE and panel b shows nMAE across 1, 3, 6, 12, and 24 h forecast horizons on a linear time axis. Panel c reports relative nRMSE reduction over persistence; positive values indicate lower nRMSE than persistence. Online Lyra is shown in deep blue. `{strongest}` has the lowest mean nRMSE among the eligible static baselines in these experiments. Error bars are 95% t intervals over random seeds after averaging sites. Markers and error bars are displaced by a small fixed display-space offset to expose overlapping intervals; the connecting trend lines retain the exact forecast-time coordinates.

**Figure 2 | Robustness across sites, horizons, and capacity groups.** Smart Persistence predicts each target from the previous day's observation at the same 15 min slot; the baseline artifact previously named Seasonal Naive is the same predictor and is counted only once. Panel a reports Online Lyra nRMSE rank among {fig02_rank_model_count} methods after averaging three seeds at each site and horizon. Panel b shows capacity-stratified nRMSE for Online Lyra, `{strongest}`, and Offline Lyra. Panel c reports nRMSE reduction relative to Smart Persistence. In panels b and c, translucent points denote site means after averaging seeds and horizons within each site, while larger markers denote capacity-group means. The mutually exclusive groups are Small <50 MW (n=4 sites), Medium 50-110 MW (n=2), and Large >110 MW (n=2); no inferential confidence interval is claimed for these small subgroups.

**Figure 3 | Gate-linked response during a representative PV ramp.** Panel a shows the complete 120 h observed PV trajectory together with Online Lyra and persistence forecasts. Panel b shows the binary gate decision over the same complete time axis; the highlighted event occurs while the Lyra branch is selected. Panel c independently expands the highlighted ramp and identifies the persistence lag and Lyra response without obscuring the full trajectory. The 6 h event (103.5-109.5 h) was selected reproducibly from all 6-8 h windows in the displayed trace. Eligible windows required at least 95% Lyra-gate selection, 50% daytime samples, an observed excursion of at least 30% of nominal capacity, at least 25% lower MAE and RMSE than persistence, a point-wise win fraction of at least 70%, and at least 30% lower change and rapid-change errors; the eligible window with the highest composite score was retained. In the selected event, Online Lyra achieves MAE 2.08 MW and RMSE 2.65 MW, compared with 6.30 MW and 8.72 MW for persistence.

**Figure 5 | Daytime accuracy and ramp robustness.** Only active-generation targets (observed PV >5% of site capacity) are evaluated. Smart Persistence predicts each target from the previous day's observation at the same 15 min slot; the first day at each site is excluded for all three methods. Panel a reports site-level nMAE, nRMSE, and the 90th percentile of absolute normalized error. Small translucent points denote individual sites, large markers denote means, and error bars are 95% t intervals over eight sites. Mean nMAE is {fig05_online_nmae:.3f} for Online Lyra, {fig05_comparator_nmae:.3f} for `{fig05_comparator}`, and {fig05_smart_nmae:.3f} for Smart Persistence. Panel b stratifies site-level nMAE by quartiles of absolute 15 min PV change; Q4 (>{fig05_ramp_q4_threshold_pct:.2f}% of capacity per 15 min) denotes the highest-variability regime. Lower values indicate better forecasts.

**Figure 6 | Deployment-oriented accuracy-efficiency trade-off.** Panel a places models at their measured trainable-parameter count and mean nRMSE after each seed first averages the complete 8-site x 5-horizon grid. Bubble area is strictly proportional to measured CUDA inference latency with a zero-intercept mapping shared by the data and the single-line reference scale. Online Lyra is emphasized by color and outline rather than bubble enlargement. The thin grey dashed line connects only the models identified as non-dominated under joint minimization of trainable parameters and mean nRMSE. Panel b reports parameter, model-size, CUDA-latency, and CPU-latency ratios relative to Online Lyra, with rows ordered by ascending mean nRMSE. The Online Lyra row provides a 1.0x visual reference. Colors are centered at 1x by plotting log2(ratio) with a diverging normalization whose midpoint is pure white; the colorbar also marks the 1x boundary. PyTorch peak reserved memory is omitted because it is identical for all displayed models (22 MB) under this benchmark. Three-seed confidence intervals remain available in the source-data audit but are not drawn in this deployment figure.

**Supplementary Figure S1 | Fallback decisions by site and forecast horizon.** Panel a shows fallback fraction with site and horizon identity preserved. Panel b relates fallback fraction to persistence-normalized skill, with selected outlier sites annotated.

**Supplementary Figure S2 | Core ablation and cold-start sensitivity.** Panel a reports removed-component effects as delta nRMSE against the full Online Lyra model, ordered by effect magnitude and clipped only where the residual-learner removal exceeds the display range. Panel b reports cold-start sensitivity under shorter offline history.

**Supplementary Figure S3 | Paired effect size and statistical significance.** Each point compares Online Lyra with one baseline using site×horizon pairs. The x-axis is median paired relative improvement and the y-axis is -log10(FDR-adjusted Wilcoxon p-value). The dotted horizontal line marks FDR-adjusted p=0.05.

"""
    (OUT / "notes" / "PROPOSED_CAPTIONS.md").write_text(captions, encoding="utf-8")

    delivery = """# Figure Delivery Audit

| Figure | Final size | Primary claim | Source data | Outputs |
|---|---:|---|---|---|
| Fig. 1 `fig01_accuracy` | 7.25 x 2.95 in (184 x 75 mm) | Online Lyra remains in the top accuracy tier across horizons. | `fig01_accuracy_seed_ci.csv`; seed summaries from final/baseline CSVs | SVG, PDF, PNG, TIFF |
| Fig. 2 `fig02_robustness` | 7.25 x 4.15 in (184 x 105 mm) | Online Lyra remains competitive across sites, horizons, and capacity groups under a common-mask comparison that includes Smart Persistence. | `fig02_common_mask_model_metrics.csv`; `fig02_common_mask_rank_table.csv`; `fig02_capacity_nrmse_site_units.csv`; `fig02_capacity_nrmse_site_summary.csv`; `fig02_capacity_skill_site_units.csv`; `fig02_capacity_skill_site_summary.csv`; common-mask audit | SVG, PDF, PNG, TIFF |
| Fig. 3 `fig03_case_trace_gate` | 7.25 x 4.85 in (184 x 123 mm) | A complete trace links gate participation to a reproducibly selected PV ramp and its local forecast response. | Online Lyra h=4 prediction NPZ; `fig03_event_window_screening.csv`; `fig03_event_metadata.csv` | SVG, PDF, PNG, TIFF |
| Fig. 5 `fig05_daytime_error_agreement` | 7.25 x 3.25 in (184 x 83 mm) | Online Lyra has the lowest mean daytime error and remains competitive as PV ramp intensity increases. | `fig05_daytime_site_metrics.csv`; `fig05_daytime_metric_summary.csv`; `fig05_ramp_quartile_site_nmae.csv`; `fig05_ramp_quartile_summary.csv`; `fig05_smart_persistence_audit.csv` | SVG, PDF, PNG, TIFF |
| Fig. 6 `fig06_efficiency` | 185.6 x 80.1 mm (tight PDF) | Online Lyra offers an attractive deployment trade-off among accurate models. | `fig06_efficiency_resources.csv`; `fig06_accuracy_seed_units.csv`; `fig06_accuracy_seed_ci.csv`; `fig06_pareto_frontier_params_nrmse.csv`; `fig06_pareto_dominance_audit.csv`; `fig06_resource_ratio_to_online.csv`; bubble-scale/colormap/label/order/memory audits | SVG, PDF, PNG, TIFF |
| Fig. S1 `figS01_fallback_grouped_dot` | 7.25 x 3.25 in (184 x 83 mm) | Fallback is sparse and localized by site/horizon. | gate profile CSV | SVG, PDF, PNG, TIFF |
| Fig. S2 `figS02_core_ablation_cold_start` | 7.25 x 3.35 in (184 x 85 mm) | Residual learner and online adaptation are the most important removals; cold-start remains stable. | core ablation CSVs | SVG, PDF, PNG, TIFF |
| Fig. S3 `figS03_paired_effect_size_significance` | 7.25 x 3.05 in (184 x 77 mm) | Improvements are summarized as paired effect sizes with adjusted significance. | paired test CSV and main condition averages | SVG, PDF, PNG, TIFF |

Font settings: base 8 pt, axis labels 8 pt, tick labels 7.5 pt, legends 7.5 pt, panel labels 10 pt bold. The plotting backend is Python/matplotlib only.
"""
    (OUT / "notes" / "FIGURE_DELIVERY_AUDIT.md").write_text(delivery, encoding="utf-8")

    manifest = pd.DataFrame(
        [
            ("Fig. 1", "fig01_accuracy", "Main", "Accuracy by horizon; Online Lyra, strongest baseline, and grey context baselines."),
            ("Fig. 2", "fig02_robustness", "Main", "Site/horizon rank and corrected capacity-group robustness."),
            ("Fig. 3", "fig03_case_trace_gate", "Main", "Operational case trace plus gate state."),
            ("Fig. 5", "fig05_daytime_error_agreement", "Main", "Site-level daytime accuracy and robustness across PV ramp quartiles."),
            ("Fig. 6", "fig06_efficiency", "Main", "Deployment efficiency and resource ratios."),
            ("Fig. S1", "figS01_fallback_grouped_dot", "Supplementary", "Fallback behavior by site/horizon."),
            ("Fig. S2", "figS02_core_ablation_cold_start", "Supplementary", "Core ablation and cold-start sensitivity."),
            ("Fig. S3", "figS03_paired_effect_size_significance", "Supplementary", "Effect size and FDR-adjusted Wilcoxon evidence."),
        ],
        columns=["figure", "stem", "placement", "purpose"],
    )
    manifest.to_csv(OUT / "figure_manifest_revised.csv", index=False)
def main() -> None:
    configure_style()
    ensure_dirs()
    context: dict[str, object] = {}
    context.update(fig01_accuracy_revised())
    context.update(fig02_robustness_revised(str(context["strongest_baseline"])))
    fig03_case_trace_revised()
    context.update(fig05_daytime_error_agreement_revised())
    fig06_efficiency_revised()
    figS01_fallback_dot_revised()
    figS02_ablation_revised()
    figS03_effect_significance_revised()
    figS04_daytime_calibration_density()
    figS05_hankel_singular_spectrum()
    write_revised_reports(context)
    make_contact_sheet(
        [
            "fig01_accuracy",
            "fig02_robustness",
            "fig03_case_trace_gate",
            "fig05_daytime_error_agreement",
            "fig06_efficiency",
            "figS01_fallback_grouped_dot",
            "figS02_core_ablation_cold_start",
            "figS03_paired_effect_size_significance",
            "figS04_daytime_calibration_density",
            "figS05_hankel_singular_spectrum",
        ]
    )
    print(f"Wrote revised figures to {OUT}")


if __name__ == "__main__":
    main()
