"""Create paper-ready tables and figures from an Online Lyra PV result folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "persistence",
    "trend_only",
    "online_lyra_pv",
    "online_lyra_pv_gated",
]
MODEL_LABELS = {
    "persistence": "Persistence",
    "trend_only": "Lyra trend only",
    "online_lyra_pv": "Online Lyra",
    "online_lyra_pv_gated": "Gated Online Lyra",
}
COLORS = {
    "persistence": "#707070",
    "trend_only": "#8da0cb",
    "online_lyra_pv": "#fc8d62",
    "online_lyra_pv_gated": "#66c2a5",
}


def configure_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    return plt


def savefig(plt, result_dir: Path, name: str) -> None:
    for ext in ["png", "pdf"]:
        plt.savefig(result_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close()


def short_farm_name(name: str) -> str:
    return (
        name.replace("Solar station site ", "Site ")
        .replace(" (Nominal capacity-", "\n")
        .replace(")", "")
    )


def make_heatmap(
    plt,
    result_dir: Path,
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    name: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    data = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns.astype(str))
    ax.set_yticks(np.arange(matrix.shape[0]), [short_farm_name(s) for s in matrix.index])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_title(title)
    threshold = np.nanmean(data)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = data[i, j]
            text_color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label(cbar_label)
    savefig(plt, result_dir, name)


def make_tables_and_figures(result_dir: Path) -> None:
    summary_path = result_dir / "online_pv_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    mean_by_model = (
        df.groupby("model")[
            [
                "nrmse",
                "daytime_nrmse",
                "nmae",
                "daytime_nmae",
                "skill_vs_persistence",
                "gate_lyra_fraction",
                "predict_latency_ms",
                "update_latency_ms",
            ]
        ]
        .mean(numeric_only=True)
        .reindex(MODEL_ORDER)
    )
    by_horizon = df[df.model.isin(MODEL_ORDER)].pivot_table(
        index="horizon_minutes",
        columns="model",
        values="nrmse",
    )
    gated = df[df.model == "online_lyra_pv_gated"]
    gate_fraction = gated.pivot_table(
        index="farm",
        columns="horizon_minutes",
        values="gate_lyra_fraction",
    )
    gated_by_farm = gated.pivot_table(
        index="farm",
        columns="horizon_minutes",
        values="nrmse",
    )

    mean_by_model.to_csv(result_dir / "table_mean_by_model.csv")
    by_horizon.to_csv(result_dir / "table_nrmse_by_horizon.csv")
    gate_fraction.to_csv(result_dir / "table_gate_fraction.csv")
    gated_by_farm.to_csv(result_dir / "table_gated_nrmse_by_farm.csv")

    plt = configure_matplotlib()

    mean = mean_by_model["nrmse"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(mean)), mean.values, color=[COLORS[m] for m in mean.index], width=0.68)
    ax.set_xticks(range(len(mean)), [MODEL_LABELS[m] for m in mean.index], rotation=20, ha="right")
    ax.set_ylabel("nRMSE")
    ax.set_title("Average forecasting error across PV farms and horizons")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(mean.values):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    savefig(plt, result_dir, "fig_model_mean_nrmse")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for model in MODEL_ORDER:
        if model in by_horizon.columns:
            ax.plot(
                by_horizon.index,
                by_horizon[model],
                marker="o",
                linewidth=2.0,
                label=MODEL_LABELS[model],
                color=COLORS[model],
            )
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("nRMSE")
    ax.set_title("nRMSE by forecast horizon")
    ax.set_xticks(by_horizon.index)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    savefig(plt, result_dir, "fig_nrmse_by_horizon")

    make_heatmap(
        plt,
        result_dir,
        gated_by_farm,
        "Gated Online Lyra nRMSE by farm and horizon",
        "nRMSE",
        "fig_gated_nrmse_heatmap",
        cmap="YlGnBu",
    )
    make_heatmap(
        plt,
        result_dir,
        gate_fraction,
        "Fraction of windows using Lyra correction",
        "Gate Lyra fraction",
        "fig_gate_fraction_heatmap",
        cmap="magma",
        vmin=0,
        vmax=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_tables_and_figures(args.result_dir)


if __name__ == "__main__":
    main()
