#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate IEEE-layout figures without redesigning the final paper figures.

The scientific drawing logic is owned by ``make_pv_paper_figures_final.py``.
This entry point calls that renderer unchanged and only modifies export
geometry:

* multi-panel figures are emitted as independent, source-rendered panels;
* panel axes are placed on consistent single-column canvases;
* analytical/residual characterization panels are recomputed from PV data;
* Fig. 7 keeps the final three-stage case-study design as one single-column
  figure;

No colors, markers, line styles, annotations, legends, data transformations,
or statistical definitions are redefined here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "tmp" / "cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(PROJECT_ROOT))

from scripts import make_pv_paper_figures_final as final  # noqa: E402
from scripts import make_pv_signal_characteristics_real as signal_characteristics  # noqa: E402

ORIGINAL_FINAL_SAVE = final.save_pub_final

FINAL_ROOT = PROJECT_ROOT / "outputs" / "paper_result_pack" / "figures_revised_final"
DATA_ROOT = FINAL_ROOT / "data"
COMPONENT_ROOT = PROJECT_ROOT / "outputs" / "paper_result_pack" / "component_characterization"
OUT = PROJECT_ROOT / "outputs" / "manuscript" / "online_lyra_ieee" / "figures"
FORMATS = ("pdf", "svg", "png")
MANUSCRIPT_FIGURE_DIRS = (
    PROJECT_ROOT / "outputs" / "manuscript" / "ieee" / "figures_ieee",
    PROJECT_ROOT / "outputs" / "manuscript" / "onlinelyra_ieee_final" / "figures_ieee",
    PROJECT_ROOT / "outputs" / "manuscript" / "onlinelyra_ieee_10page_submission" / "figures_ieee",
)

SINGLE_WIDTH = 3.50
SINGLE_HEIGHT = 2.48
DOUBLE_WIDTH = 7.10
RANK_HEIGHT = 1.62
LOW_RANK_HEIGHT = 1.95
DAYTIME_A_HEIGHT = 2.02
DAYTIME_B_HEIGHT = 2.05
DENSE_HEIGHT = 2.72
CASE_HEIGHT = 3.20
COLORBAR_WIDTH = 0.44
ABLATION_WIDTH = 326.262 / 72.0
ABLATION_HEIGHT = 215.374 / 72.0

# Each item is (output stem, axes retained together, canvas height, geometry).
# The axes indices are those produced by the canonical final renderer.
PANEL_EXPORTS: dict[str, list[tuple[str, tuple[int, ...], float, str]]] = {
    "fig01_accuracy": [
        ("fig05_accuracy_a", (0,), SINGLE_HEIGHT, "cartesian"),
        ("fig05_accuracy_b", (1,), SINGLE_HEIGHT, "cartesian"),
        ("fig05_accuracy_c", (2,), SINGLE_HEIGHT, "cartesian"),
    ],
    "fig05_daytime_error_agreement": [
        ("fig06_daytime_error_a", (0,), DAYTIME_A_HEIGHT, "daytime_a"),
        ("fig06_daytime_error_b", (1,), DAYTIME_B_HEIGHT, "daytime_b"),
    ],
    "fig06_efficiency": [
        ("fig13_efficiency_a", (0,), DENSE_HEIGHT, "cartesian"),
        ("fig13_efficiency_b", (1, 2), DENSE_HEIGHT, "colorbar"),
    ],
    "figS01_fallback_grouped_dot": [
        ("fig10_fallback_a", (0,), SINGLE_HEIGHT, "cartesian"),
        ("fig10_fallback_b", (1,), SINGLE_HEIGHT, "cartesian"),
    ],
    "figS02_core_ablation_cold_start": [
        ("fig11_ablation_a", (0,), ABLATION_HEIGHT, "cartesian"),
        ("fig11_ablation_b", (1,), SINGLE_HEIGHT, "cartesian"),
    ],
    "figS03_paired_effect_size_significance": [
        ("fig12_significance_a", (0,), DENSE_HEIGHT, "cartesian"),
        ("fig12_significance_b", (1,), DENSE_HEIGHT, "cartesian"),
    ],
    "figS05_hankel_singular_spectrum": [
        ("fig02_low_rank_a", (0,), LOW_RANK_HEIGHT, "low_rank"),
        ("fig02_low_rank_b", (1,), LOW_RANK_HEIGHT, "low_rank"),
    ],
}


FRAME_COLOR = "#000000"
FRAME_LINEWIDTH = 0.55
TICK_LINEWIDTH = 0.45
GRID_COLOR = "#C9CED4"
GRID_LINEWIDTH = 0.36
GRID_ALPHA = 0.28


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        (OUT / fmt).mkdir(parents=True, exist_ok=True)


def save_formats(fig: plt.Figure, stem: str, *, tight: bool = False) -> None:
    fig.canvas.draw()
    if tight:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
    else:
        kwargs = {}
    fig.savefig(OUT / "pdf" / f"{stem}.pdf", **kwargs)
    fig.savefig(OUT / "svg" / f"{stem}.svg", **kwargs)
    fig.savefig(OUT / "png" / f"{stem}.png", dpi=600, **kwargs)


def normalize_publication_chrome(fig: plt.Figure) -> None:
    """Unify neutral plot furniture without changing visual data encodings."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ax.set_facecolor("white")
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(FRAME_COLOR)
            ax.spines[side].set_linewidth(FRAME_LINEWIDTH)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            width=TICK_LINEWIDTH,
            length=3.0,
            color=FRAME_COLOR,
            labelcolor=FRAME_COLOR,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            width=0.45,
            length=1.8,
            color=FRAME_COLOR,
            labelcolor=FRAME_COLOR,
        )
        ax.xaxis.label.set_color(FRAME_COLOR)
        ax.yaxis.label.set_color(FRAME_COLOR)
        # Restyle only grid lines already enabled by the canonical renderer.
        for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            if gridline.get_visible():
                gridline.set_color(GRID_COLOR)
                gridline.set_linewidth(GRID_LINEWIDTH)
                gridline.set_alpha(GRID_ALPHA)
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_edgecolor("#D7DBDF")
            frame.set_linewidth(0.38)
            frame.set_alpha(0.72)


def _set_panel_geometry(fig: plt.Figure, kept: tuple[int, ...], geometry: str) -> None:
    axes = fig.axes
    for index, ax in enumerate(axes):
        ax.set_visible(index in kept)

    main = axes[kept[0]]
    if geometry == "cartesian":
        main.set_position([0.19, 0.19, 0.77, 0.74])
    elif geometry == "low_rank":
        main.set_position([0.18, 0.23, 0.79, 0.72])
    elif geometry == "daytime_a":
        main.set_position([0.18, 0.22, 0.79, 0.74])
    elif geometry == "daytime_b":
        # The quartile labels use two lines, so retain a slightly deeper
        # bottom margin while keeping the plotting area wide and flat.
        main.set_position([0.18, 0.29, 0.79, 0.67])
        for annotation in main.texts:
            if annotation.get_text() == "High-variability regime":
                # This artist uses the x-axis data transform, so keep it just
                # inside the Q4 boundary rather than using axes coordinates.
                annotation.set_x(3.96)
                annotation.set_horizontalalignment("right")
        legend = main.get_legend()
        if legend is not None:
            legend.set_loc("upper center")
            legend.set_bbox_to_anchor((0.38, 0.985), transform=main.transAxes)
    elif geometry == "colorbar":
        # Preserve the longest model label when the heatmap is exported from
        # the composite as an independent single-column panel.
        main.set_position([0.24, 0.19, 0.58, 0.74])
        axes[kept[1]].set_position([0.85, 0.19, 0.025, 0.74])
    elif geometry == "square":
        main.set_position([0.19, 0.18, 0.73, 0.73])
        main.set_box_aspect(1)
    elif geometry == "square_colorbar":
        main.set_position([0.16, 0.18, 0.70, 0.73])
        main.set_box_aspect(1)
        axes[kept[1]].set_position([0.885, 0.18, 0.030, 0.73])
    elif geometry == "standalone_colorbar":
        main.set_position([0.36, 0.18, 0.22, 0.73])
    else:
        raise ValueError(f"Unknown panel geometry: {geometry}")

    # The final calibration renderer uses shared figure-level labels. When a
    # panel is exported independently, repeat those same labels on its axis.
    if geometry.startswith("square"):
        main.set_xlabel("Observed normalized PV")
        main.set_ylabel("Predicted normalized PV")

    # Figure-level labels belong to the original composite and would otherwise
    # float into an independent panel. This is layout cleanup only.
    for artist in fig.texts:
        artist.set_visible(False)


def export_independent_panel(
    fig: plt.Figure,
    stem: str,
    kept: tuple[int, ...],
    height: float,
    geometry: str,
) -> None:
    original_size = tuple(fig.get_size_inches())
    original_positions = [ax.get_position().frozen() for ax in fig.axes]
    original_visibility = [ax.get_visible() for ax in fig.axes]
    original_text_visibility = [text.get_visible() for text in fig.texts]

    if stem == "fig11_ablation_a":
        panel_width = ABLATION_WIDTH
    else:
        panel_width = COLORBAR_WIDTH if geometry == "standalone_colorbar" else SINGLE_WIDTH
    fig.set_size_inches(panel_width, height, forward=True)
    _set_panel_geometry(fig, kept, geometry)
    normalize_publication_chrome(fig)
    if stem == "fig06_daytime_error_b":
        main = fig.axes[kept[0]]
        legend = main.get_legend()
        if legend is not None:
            # The geometry stage above defines the open region; this offset
            # keeps the legend clear of the Q1 error bar and Q4 annotation.
            legend.set_loc("upper left")
            legend.set_bbox_to_anchor((0.16, 0.965), transform=main.transAxes)
    if stem == "fig11_ablation_a":
        fig.axes[kept[0]].set_position([0.31, 0.17, 0.66, 0.78])
    # Fixed media boxes are essential: LaTeX must not rescale each panel from
    # a different tight crop, which changes apparent frame and font weights.
    save_formats(fig, stem, tight=False)

    fig.set_size_inches(*original_size, forward=True)
    for ax, position, visible in zip(fig.axes, original_positions, original_visibility):
        ax.set_position(position)
        ax.set_visible(visible)
    for text, visible in zip(fig.texts, original_text_visibility):
        text.set_visible(visible)


def final_geometry_save(fig: plt.Figure, source_stem: str) -> None:
    """Final-renderer save hook that changes geometry, never plot semantics."""
    final.style_figure_axes(fig)
    final.remove_panel_letters(fig)
    final.add_missing_data_legends(fig)
    final.apply_final_axis_padding(fig, source_stem)
    normalize_publication_chrome(fig)
    fig.canvas.draw()

    if source_stem == "fig03_case_trace_gate":
        # Preserve the complete final case-study narrative. Only the physical
        # canvas is reduced to IEEE single-column width.
        fig.set_size_inches(SINGLE_WIDTH, CASE_HEIGHT, forward=True)
        save_formats(fig, "fig03_case_trace_gate", tight=True)
    elif source_stem == "fig02_robustness":
        # Export the established composite and vector trims from this run.
        ORIGINAL_FINAL_SAVE(fig, source_stem)
    else:
        for panel_stem, kept, height, geometry in PANEL_EXPORTS.get(source_stem, []):
            export_independent_panel(fig, panel_stem, kept, height, geometry)
    plt.close(fig)


def generate_from_final_renderer() -> None:
    """Run the canonical final renderer with a geometry-only save hook."""
    final.configure_publication_style()
    final.install_inside_legend_policy()
    final.save_pub_final = final_geometry_save
    final.redirect_existing_modules()
    final.generate_core_data_and_figures()
    final.generate_refined_renderers()
    # Fig. 1 is rendered once below through its standalone path.
    final.generate_final_overrides(include_fig1=False)


def generate_real_data_motivation() -> None:
    """Rebuild the measured-data motivation figure before publishing it."""
    signal_characteristics.main()


def generate_component_characterization() -> None:
    """Recompute all four analytical/residual characterization panels."""
    data_dir = PROJECT_ROOT / "solar_stations"
    station_files = sorted(data_dir.glob("Solar station site *.xlsx"))
    if len(station_files) != 8:
        raise FileNotFoundError(
            "Component characterization requires all eight PV-station workbooks; "
            f"found {len(station_files)} in {data_dir}."
        )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_pv_decomposition_characteristics.py"),
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(COMPONENT_ROOT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    stems = (
        "fig_component_a_energy",
        "fig_component_b_effective_rank",
        "fig_component_c_spectral_entropy",
        "fig_component_d_acf",
    )
    missing = [
        path
        for stem in stems
        for fmt in FORMATS
        if not (path := COMPONENT_ROOT / f"{stem}.{fmt}").is_file()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Component characterization did not produce every required panel:\n"
            f"{missing_text}"
        )


def generate_accuracy_panels() -> None:
    """Regenerate the focus-plus-context accuracy design."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "make_fig1.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    revised_v2_root = PROJECT_ROOT / "outputs" / "paper_result_pack" / "figures_revised_v2"
    original_root = final.OUTPUT_ROOT
    original_trims = final.PANEL_TRIMS
    try:
        final.OUTPUT_ROOT = revised_v2_root
        final.PANEL_TRIMS = {"fig01_accuracy": original_trims["fig01_accuracy"]}
        final.export_individual_panels()
    finally:
        final.OUTPUT_ROOT = original_root
        final.PANEL_TRIMS = original_trims

    for panel in ("a", "b", "c"):
        source_stem = f"fig01_accuracy_{panel}"
        destination_stem = f"fig05_accuracy_{panel}"
        for fmt in ("pdf", "png"):
            source = revised_v2_root / "panels" / fmt / f"{source_stem}.{fmt}"
            destination = OUT / fmt / f"{destination_stem}.{fmt}"
            shutil.copy2(source, destination)


def publish_robustness_panels() -> None:
    """Export and publish robustness panels from this render."""
    original_trims = final.PANEL_TRIMS
    try:
        final.PANEL_TRIMS = {"fig02_robustness": original_trims["fig02_robustness"]}
        final.export_individual_panels()
    finally:
        final.PANEL_TRIMS = original_trims

    for panel in ("a", "b", "c"):
        source_stem = f"fig02_robustness_{panel}"
        destination_stem = f"fig08_robustness_{panel}"
        for fmt in ("pdf", "png"):
            source = FINAL_ROOT / "panels" / fmt / f"{source_stem}.{fmt}"
            destination = OUT / fmt / f"{destination_stem}.{fmt}"
            shutil.copy2(source, destination)


def publish_compact_manuscript_panels() -> None:
    """Copy fixed-canvas vector panels into active manuscript packages."""
    stems = (
        "fig02_low_rank_a",
        "fig02_low_rank_b",
        "fig05_accuracy_a",
        "fig05_accuracy_b",
        "fig05_accuracy_c",
        "fig06_daytime_error_a",
        "fig06_daytime_error_b",
        "fig08_robustness_a",
        "fig08_robustness_b",
        "fig08_robustness_c",
    )
    for destination in MANUSCRIPT_FIGURE_DIRS:
        destination.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            for fmt in FORMATS:
                source = OUT / fmt / f"{stem}.{fmt}"
                if source.exists():
                    shutil.copy2(source, destination / f"{stem}.{fmt}")


def publish_all_active_manuscript_figures() -> None:
    """Map newly rendered panels to the stable manuscript filenames."""
    panel_map = {
        "fig10_fallback_a": "fig11_fallback_a",
        "fig10_fallback_b": "fig11_fallback_b",
        "fig11_ablation_a": "fig9_ablation_a",
        "fig12_significance_a": "fig10_significance_a",
        "fig12_significance_b": "fig10_significance_b",
        "fig13_efficiency_a": "fig12_efficiency_a",
        "fig13_efficiency_b": "fig12_efficiency_b",
    }
    component_map = {
        "fig_component_a_energy": "fig_component_characterization_a",
        "fig_component_b_effective_rank": "fig_component_characterization_b",
        "fig_component_c_spectral_entropy": "fig_component_characterization_c",
        "fig_component_d_acf": "fig_component_characterization_d",
    }
    signal_root = FINAL_ROOT / "real_data_pv_characteristics"

    for destination in MANUSCRIPT_FIGURE_DIRS:
        destination.mkdir(parents=True, exist_ok=True)
        for source_stem, destination_stem in panel_map.items():
            for fmt in FORMATS:
                source = OUT / fmt / f"{source_stem}.{fmt}"
                if source.exists():
                    shutil.copy2(source, destination / f"{destination_stem}.{fmt}")
        for source_stem, destination_stem in component_map.items():
            for fmt in FORMATS:
                source = COMPONENT_ROOT / f"{source_stem}.{fmt}"
                shutil.copy2(source, destination / f"{destination_stem}.{fmt}")
        for fmt in FORMATS:
            source = signal_root / f"fig01_pv_signal_characteristics_real.{fmt}"
            if source.exists():
                shutil.copy2(source, destination / f"fig01_pv_signal_characteristics.{fmt}")


def publish_rank_figure6() -> None:
    """Publish the corrected 13-method rank panel as standalone Figure 6."""
    destinations = (OUT, *MANUSCRIPT_FIGURE_DIRS)
    panel_sources = {
        "pdf": FINAL_ROOT / "panels" / "pdf" / "fig02_robustness_a.pdf",
        "png": FINAL_ROOT / "panels" / "png" / "fig02_robustness_a.png",
    }
    for fmt, source in panel_sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing corrected rank-panel source: {source}")
        for destination_root in destinations:
            destination = destination_root / fmt if destination_root == OUT else destination_root
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / f"fig06_robustness.{fmt}")

    # Preserve editable SVG text while applying the same vector crop as the
    # canonical PDF panel: (left, bottom, right, top) = (3, 150, 4, 3) pt.
    source_svg = FINAL_ROOT / "svg" / "fig02_robustness.svg"
    svg_text = source_svg.read_text(encoding="utf-8")
    svg_text = re.sub(
        r'width="[^"]+pt" height="[^"]+pt" viewBox="[^"]+"',
        'width="511.712814pt" height="116.140711pt" viewBox="3 3 511.712814 116.140711"',
        svg_text,
        count=1,
    )
    for destination_root in destinations:
        destination = destination_root / "svg" if destination_root == OUT else destination_root
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "fig06_robustness.svg").write_text(svg_text, encoding="utf-8")


def make_overview() -> None:
    stems = [
        "fig02_low_rank_a", "fig02_low_rank_b",
        "fig05_accuracy_a", "fig05_accuracy_b", "fig05_accuracy_c",
        "fig06_robustness",
        "fig03_case_trace_gate",
        "fig09_daytime_a", "fig09_daytime_b",
        "fig10_fallback_a", "fig10_fallback_b",
        "fig11_ablation_a", "fig11_ablation_b",
        "fig12_significance_a", "fig12_significance_b",
        "fig13_efficiency_a", "fig13_efficiency_b",
    ]
    thumbs: list[tuple[str, Image.Image]] = []
    for stem in stems:
        path = OUT / "png" / f"{stem}.png"
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((920, 650), Image.Resampling.LANCZOS)
            thumbs.append((stem, image.copy()))

    cols = 3
    cell_w, cell_h = 980, 720
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (stem, image) in enumerate(thumbs):
        row, col = divmod(index, cols)
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * cell_h + 40 + (620 - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((col * cell_w + 24, row * cell_h + 12), stem, fill="#24282D")
    sheet.save(OUT / "figure_overview_ieee.png", dpi=(300, 300))


def write_manifest() -> None:
    rows = []
    for source, specs in PANEL_EXPORTS.items():
        for stem, kept, height, geometry in specs:
            width = COLORBAR_WIDTH if geometry == "standalone_colorbar" else SINGLE_WIDTH
            rows.append(
                {
                    "output": stem,
                    "source_final_figure": source,
                    "source_axes": ",".join(map(str, kept)),
                    "width_in": width,
                    "height_in": height,
                    "geometry_only": True,
                    "geometry": geometry,
                }
            )
    rows.extend(
        [
            {
                "output": "fig06_robustness",
                "source_final_figure": "fig02_robustness",
                "source_axes": "rank panel only",
                "width_in": 511.417 / 72,
                "height_in": 116.034 / 72,
                "geometry_only": True,
                "geometry": "canonical revised-final top-panel vector crop",
            },
            {
                "output": "fig03_case_trace_gate",
                "source_final_figure": "fig03_case_trace_gate",
                "source_axes": "all",
                "width_in": SINGLE_WIDTH,
                "height_in": CASE_HEIGHT,
                "geometry_only": True,
                "geometry": "complete final case figure",
            },
        ]
    )
    pd.DataFrame(rows).to_csv(OUT / "ieee_figure_manifest.csv", index=False)
    (OUT / "ieee_figure_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    mpl.use("Agg")
    ensure_dirs()
    # Run the independent window-level SVD analysis before the full renderer
    # allocates its larger collection of figures.
    generate_component_characterization()
    generate_from_final_renderer()
    generate_accuracy_panels()
    generate_real_data_motivation()
    publish_robustness_panels()
    publish_compact_manuscript_panels()
    publish_all_active_manuscript_figures()
    publish_rank_figure6()
    make_overview()
    write_manifest()
    print(f"IEEE geometry-only figures written to {OUT}")


if __name__ == "__main__":
    main()
