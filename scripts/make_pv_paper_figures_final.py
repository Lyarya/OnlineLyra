#!/usr/bin/env python3
"""Final figure renderer for the Online Lyra paper.

Usage:

    python scripts/make_pv_paper_figures_final.py

All outputs are written under:

    outputs/paper_result_pack/figures_revised_final

Existing plotting functions are imported and redirected to the final output
root at runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))
(PROJECT_ROOT / "tmp" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scripts import make_pv_paper_figures_revised as base  # noqa: E402
from scripts import make_pv_paper_figures_revised_v2 as refined  # noqa: E402


OUTPUT_ROOT = base.PACK / "figures_revised_final"
DATA_OUT = OUTPUT_ROOT / "data"
FORMATS = ("svg", "pdf", "png", "tiff")
FIGURES = [
    ("Fig. 1", "fig01_accuracy", "Forecasting accuracy comparison"),
    ("Fig. 2", "fig02_robustness", "Robustness and site-level comparison"),
    ("Fig. 3", "fig03_case_trace_gate", "Case study and gate behavior"),
    ("Fig. 5", "fig05_daytime_error_agreement", "Prediction agreement and reliability"),
    ("Fig. 6", "fig06_efficiency", "Efficiency and deployment cost"),
    ("Fig. S1", "figS01_fallback_grouped_dot", "Fallback behavior"),
    ("Fig. S2", "figS02_core_ablation_cold_start", "Ablation and cold-start analysis"),
    ("Fig. S3", "figS03_paired_effect_size_significance", "Paired effect size and significance"),
    ("Fig. S4", "figS04_daytime_calibration_density", "Prediction agreement details"),
    ("Fig. S5", "figS05_hankel_singular_spectrum", "Hankel singular-value analysis"),
]

# PDF trims are expressed as (left, bottom, right, top) in PostScript points.
# Fig. 3 is intentionally retained as one three-stage narrative. All other
# composite figures are exported as separate
# vector panels for single-column IEEE layout.
PANEL_TRIMS = {
    "fig01_accuracy": {
        "a": (0, 18, 337, 3),
        "b": (164, 18, 164, 3),
        "c": (330, 18, 3, 3),
    },
    "fig02_robustness": {
        "a": (3, 150, 4, 3),
        "b": (8, 15, 265, 132),
        "c": (260, 15, 3, 132),
    },
    "fig05_daytime_error_agreement": {
        "a": (0, 8, 258, 3),
        "b": (245, 4, 2, 3),
    },
    "fig06_efficiency": {
        "a": (0, 4, 255, 3),
        "b": (245, 4, 2, 3),
    },
    "figS01_fallback_grouped_dot": {
        "a": (0, 4, 215, 3),
        "b": (225, 4, 2, 3),
    },
    "figS02_core_ablation_cold_start": {
        "a": (0, 4, 190, 3),
        "b": (318, 4, 2, 3),
    },
    "figS03_paired_effect_size_significance": {
        "a": (0, 4, 210, 3),
        "b": (210, 4, 2, 3),
    },
    "figS04_daytime_calibration_density": {
        "a": (0, 4, 354, 3),
        "b": (172, 4, 184, 3),
        "c": (344, 4, 2, 3),
    },
    "figS05_hankel_singular_spectrum": {
        "a": (0, 4, 245, 3),
        "b": (250, 4, 2, 3),
    },
}
LEGEND_STYLE = {
    "loc": "best",
    "frameon": True,
    "facecolor": "white",
    "framealpha": 0.68,
    "edgecolor": "#D5DADF",
    "fontsize": "small",
    "handlelength": 1.25,
    "borderaxespad": 0.62,
    "borderpad": 0.42,
    "labelspacing": 0.34,
}
FRAME_COLOR = "#000000"
FRAME_LINEWIDTH = 0.55
TICK_LINEWIDTH = 0.45
GRID_COLOR = "#C9CED4"
GRID_LINEWIDTH = 0.36
GRID_ALPHA = 0.28
ORIGINAL_AXES_LEGEND = Axes.legend
ORIGINAL_FIGURE_LEGEND = Figure.legend


def configure_publication_style() -> None:
    """Apply a conservative boxed-axis style for energy-journal figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "axes.edgecolor": FRAME_COLOR,
            "axes.linewidth": FRAME_LINEWIDTH,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.color": FRAME_COLOR,
            "ytick.color": FRAME_COLOR,
            "xtick.major.width": TICK_LINEWIDTH,
            "ytick.major.width": TICK_LINEWIDTH,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": GRID_COLOR,
            "grid.linewidth": GRID_LINEWIDTH,
            "grid.alpha": GRID_ALPHA,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.framealpha": 0.68,
            "legend.edgecolor": "#D5DADF",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )


def install_inside_legend_policy() -> None:
    """Force standard legends inside one plotting axis with opaque white frames."""

    def normalize_legend_kwargs(kwargs: dict) -> dict:
        clean = dict(kwargs)
        for key in ("bbox_to_anchor", "bbox_transform", "mode"):
            clean.pop(key, None)
        clean.update({key: value for key, value in LEGEND_STYLE.items() if key not in clean})
        clean["loc"] = "best"
        clean["frameon"] = True
        clean["facecolor"] = "white"
        clean["framealpha"] = 0.68
        clean["edgecolor"] = "#D5DADF"
        clean["fontsize"] = clean.get("fontsize", "small")
        clean["handlelength"] = min(float(clean.get("handlelength", 1.25)), 1.35)
        clean["borderaxespad"] = max(float(clean.get("borderaxespad", 0.62)), 0.62)
        clean["borderpad"] = min(float(clean.get("borderpad", 0.42)), 0.48)
        clean["labelspacing"] = min(float(clean.get("labelspacing", 0.34)), 0.42)
        if int(clean.get("ncol", 1)) > 2:
            clean["ncol"] = 2
        return clean

    def inside_axes_legend(self, *args, **kwargs):
        legend = ORIGINAL_AXES_LEGEND(self, *args, **normalize_legend_kwargs(kwargs))
        legend.get_frame().set_linewidth(0.45)
        return legend

    def inside_figure_legend(self, *args, **kwargs):
        target_ax = None
        for ax in self.axes:
            if ax.get_visible() and ax.has_data():
                target_ax = ax
                break
        if target_ax is None:
            return ORIGINAL_FIGURE_LEGEND(self, *args, **normalize_legend_kwargs(kwargs))
        legend = ORIGINAL_AXES_LEGEND(target_ax, *args, **normalize_legend_kwargs(kwargs))
        legend.get_frame().set_linewidth(0.45)
        return legend

    Axes.legend = inside_axes_legend
    Figure.legend = inside_figure_legend


def style_axis(ax, *, grid: bool = True, full_box: bool = True) -> None:
    """Normalize ordinary Cartesian axes without changing data or scales."""
    if full_box:
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(FRAME_COLOR)
            ax.spines[side].set_linewidth(FRAME_LINEWIDTH)
    ax.tick_params(
        direction="in", width=TICK_LINEWIDTH, length=3.0,
        color=FRAME_COLOR, labelcolor=FRAME_COLOR,
    )
    ax.xaxis.label.set_color(FRAME_COLOR)
    ax.yaxis.label.set_color(FRAME_COLOR)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)


def style_figure_axes(fig: plt.Figure) -> None:
    """Apply boxed-axis styling only to standard quantitative axes."""
    for ax in fig.axes:
        if not ax.get_visible() or not ax.has_data():
            continue
        # Heatmaps keep their color mapping untouched but receive the same
        # neutral outer frame as Cartesian panels. Colorbar axes are excluded.
        is_colorbar = ax.get_label() == "<colorbar>"
        if is_colorbar:
            continue
        style_axis(ax, grid=len(ax.images) == 0, full_box=True)


def remove_panel_letters(fig: plt.Figure) -> None:
    """Remove a/b/c panel glyphs while preserving scientific annotations."""
    for text_artist in list(fig.texts):
        if text_artist.get_text().strip().lower() in {"a", "b", "c"}:
            text_artist.remove()
    for ax in fig.axes:
        for text_artist in list(ax.texts):
            if text_artist.get_text().strip().lower() in {"a", "b", "c"}:
                text_artist.remove()


def add_missing_data_legends(fig: plt.Figure) -> None:
    """Add a compact legend when an axis exposes multiple labeled data series."""
    for ax in fig.axes:
        if getattr(ax, "_lyra_skip_auto_legend", False):
            continue
        if ax.get_legend() is not None or len(ax.images) > 0:
            continue
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                unique.setdefault(label, handle)
        if len(unique) >= 2:
            ax.legend(list(unique.values()), list(unique.keys()), loc="best")


def apply_final_axis_padding(fig: plt.Figure, stem: str) -> None:
    """Stem-specific containment padding for known edge-label-heavy figures."""
    if stem == "figS02_core_ablation_cold_start":
        for ax in fig.axes:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if "Delta" in xlabel or "nRMSE vs full" in xlabel:
                ax.set_xlim(-5, 350)
            elif ylabel == "nRMSE":
                ax.set_xlim(20, 150)
            ax.margins(x=0.06, y=0.08)
            ax.autoscale_view(scalex=False, scaley=False)
    elif stem == "figS03_paired_effect_size_significance":
        for ax in fig.axes:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xpad = max((xmax - xmin) * 0.32, 1.35)
            ax.set_xlim(xmin - 0.05 * xpad, xmax + xpad)
            ax.set_ylim(ymin, max(ymax + 0.8, 12.0))
            ax.margins(x=0.08, y=0.10)
            ax.autoscale_view(scalex=False, scaley=False)


def ensure_output_directories() -> None:
    for subdir in (*FORMATS, "data", "notes", "panels/pdf", "panels/png"):
        (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def export_individual_panels() -> None:
    """Export vector PDF and 600-dpi PNG panels from final composite PDFs.

    Cropping is performed by a minimal temporary LaTeX wrapper, so fonts and
    line art remain vector in the panel PDFs. This avoids editing raster files
    and keeps panel generation tied to the canonical final renderer.
    """
    pdflatex = Path.home() / "Library" / "TinyTeX" / "bin" / "universal-darwin" / "pdflatex"
    pdftocairo = Path("/opt/anaconda3/bin/pdftocairo")
    if not pdflatex.exists():
        raise FileNotFoundError(f"pdflatex is required for vector panel export: {pdflatex}")
    if not pdftocairo.exists():
        raise FileNotFoundError(f"pdftocairo is required for panel previews: {pdftocairo}")

    panel_pdf_dir = OUTPUT_ROOT / "panels" / "pdf"
    panel_png_dir = OUTPUT_ROOT / "panels" / "png"
    panel_pdf_dir.mkdir(parents=True, exist_ok=True)
    panel_png_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lyra-panels-", dir=PROJECT_ROOT / "tmp") as tmp_name:
        tmp_dir = Path(tmp_name)
        for stem, panels in PANEL_TRIMS.items():
            source = (OUTPUT_ROOT / "pdf" / f"{stem}.pdf").resolve()
            if not source.exists():
                raise FileNotFoundError(f"Missing composite PDF for panel export: {source}")
            info = subprocess.run(
                ["pdfinfo", str(source)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            size_line = next(line for line in info.splitlines() if line.startswith("Page size:"))
            size_tokens = size_line.split()
            page_width = float(size_tokens[2])
            page_height = float(size_tokens[4])
            for panel, (left, bottom, right, top) in panels.items():
                panel_stem = f"{stem}_{panel}"
                crop_width = page_width - left - right
                crop_height = page_height - bottom - top
                if crop_width <= 0 or crop_height <= 0:
                    raise ValueError(f"Invalid crop for {panel_stem}: {(left, bottom, right, top)}")
                tex_path = tmp_dir / f"{panel_stem}.tex"
                tex_path.write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage{graphicx}\n"
                    f"\\usepackage[paperwidth={crop_width:.3f}bp,paperheight={crop_height:.3f}bp,margin=0pt]{{geometry}}\n"
                    "\\pagestyle{empty}\n"
                    "\\setlength{\\topskip}{0pt}\n"
                    "\\setlength{\\parindent}{0pt}\n"
                    "\\begin{document}\n"
                    "\\nointerlineskip\n"
                    f"\\includegraphics[trim={left}pt {bottom}pt {right}pt {top}pt,clip]{{{source.as_posix()}}}\n"
                    "\\end{document}\n",
                    encoding="utf-8",
                )
                subprocess.run(
                    [str(pdflatex), "-interaction=batchmode", "-halt-on-error", tex_path.name],
                    cwd=tmp_dir,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                generated = tmp_dir / f"{panel_stem}.pdf"
                target_pdf = panel_pdf_dir / f"{panel_stem}.pdf"
                # The zero-margin wrapper emits an initial empty page before
                # the precisely sized graphic page; retain only the latter.
                subprocess.run(
                    ["pdfseparate", "-f", "2", "-l", "2", str(generated), str(target_pdf)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                png_prefix = panel_png_dir / panel_stem
                subprocess.run(
                    [str(pdftocairo), "-singlefile", "-png", "-r", "600", str(target_pdf), str(png_prefix)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )

    panel_rows = []
    for stem, panels in PANEL_TRIMS.items():
        for panel, trim in panels.items():
            panel_stem = f"{stem}_{panel}"
            panel_rows.append(
                {
                    "source_figure": stem,
                    "panel": panel,
                    "trim_points": " ".join(map(str, trim)),
                    "pdf": f"panels/pdf/{panel_stem}.pdf",
                    "png": f"panels/png/{panel_stem}.png",
                }
            )
    pd.DataFrame(panel_rows).to_csv(OUTPUT_ROOT / "data" / "individual_panel_manifest.csv", index=False)


def save_pub_final(fig: plt.Figure, stem: str) -> None:
    if stem == "fig02_robustness" and fig.axes:
        first_ax = fig.axes[0]
        first_ax._lyra_skip_auto_legend = True
        legend = first_ax.get_legend()
        if legend is not None:
            legend.remove()
    style_figure_axes(fig)
    remove_panel_letters(fig)
    add_missing_data_legends(fig)
    apply_final_axis_padding(fig, stem)
    fig.canvas.draw()
    fig.savefig(OUTPUT_ROOT / "svg" / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "pdf" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "png" / f"{stem}.png", bbox_inches="tight", dpi=600)
    fig.savefig(OUTPUT_ROOT / "tiff" / f"{stem}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def import_script(script_name: str, module_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required plotting script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def redirect_existing_modules() -> None:
    base.OUT = OUTPUT_ROOT
    base.DATA_OUT = DATA_OUT
    base.save_pub = save_pub_final
    base.configure_style = configure_publication_style

    refined.OUT = OUTPUT_ROOT
    refined.DATA_OUT = DATA_OUT
    refined.save_pub = save_pub_final
    refined.base.OUT = OUTPUT_ROOT
    refined.base.DATA_OUT = DATA_OUT
    refined.base.save_pub = save_pub_final
    refined.base.configure_style = configure_publication_style


def generate_core_data_and_figures() -> dict[str, object]:
    configure_publication_style()
    ensure_output_directories()

    context = base.fig01_accuracy_revised()
    base.fig02_robustness_revised(str(context["strongest_baseline"]))
    base.figS01_fallback_dot_revised()
    base.figS02_ablation_revised()
    base.figS03_effect_significance_revised()

    base.fig03_case_trace_revised()
    base.fig05_daytime_error_agreement_revised()
    base.fig06_efficiency_revised()
    base.figS04_daytime_calibration_density()
    base.figS05_hankel_singular_spectrum()
    return context


def generate_refined_renderers() -> None:
    refined.render_fig03_story()
    refined.render_fig05_error_and_agreement()
    refined.render_fig06_direct_latency()
    refined.render_figS04_square_shared_scale()
    refined.render_figS05_contrast()


def generate_final_overrides(*, include_fig1: bool = True, include_fig3: bool = True) -> None:
    """Render the standalone figure overrides requested by the caller."""
    fig1 = import_script("make_fig1.py", "final_fig1_override") if include_fig1 else None
    fig3 = import_script("make_fig3.py", "final_fig3_override") if include_fig3 else None

    if fig1 is not None:
        fig1.OUT = OUTPUT_ROOT
        fig1.SRC = DATA_OUT
        fig1.ACC_CSV = DATA_OUT / "fig01_accuracy_seed_ci.csv"

        original_fig1_configure = fig1.configure_style

        def configure_fig1_boxed() -> None:
            original_fig1_configure()
            configure_publication_style()

        fig1.configure_style = configure_fig1_boxed

    if fig3 is not None:
        fig3.OUT = OUTPUT_ROOT
        fig3.DATA_OUT = DATA_OUT
        fig3.base.OUT = OUTPUT_ROOT
        fig3.base.DATA_OUT = DATA_OUT
        fig3.base.save_pub = save_pub_final
        fig3.base.configure_style = configure_publication_style
        fig3.save_pub = save_pub_final

    if fig1 is not None:
        fig1.main()
    if fig3 is not None:
        fig3.main()


def make_overview_thumbnail(stem: str, items: list[tuple[str, str]], rows: int, cols: int) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.55 * rows), constrained_layout=True)
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, (label, figure_stem) in zip(axes_flat, items):
        source_path = OUTPUT_ROOT / "png" / f"{figure_stem}.png"
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot build overview; missing PNG: {source_path}")
        with Image.open(source_path) as source:
            source.thumbnail((1600, 1100), Image.Resampling.LANCZOS)
            preview = source.convert("RGB")
        ax.imshow(preview, interpolation="lanczos")
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left", pad=5)
        ax.axis("off")
    for ax in axes_flat[len(items):]:
        ax.axis("off")
    fig.savefig(OUTPUT_ROOT / stem, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_overview_pngs() -> None:
    refined_items = [
        ("Fig. 3", "fig03_case_trace_gate"),
        ("Fig. 5", "fig05_daytime_error_agreement"),
        ("Fig. 6", "fig06_efficiency"),
        ("Fig. S4", "figS04_daytime_calibration_density"),
        ("Fig. S5", "figS05_hankel_singular_spectrum"),
    ]
    make_overview_thumbnail("refined_five_overview_highres.png", refined_items, 2, 3)
    make_overview_thumbnail("figure_overview_complete_highres.png", [(f, s) for f, s, _ in FIGURES], 4, 3)
    make_overview_thumbnail("full_set_overview_highres.png", [(f, s) for f, s, _ in FIGURES], 4, 3)


def write_manifest() -> None:
    rows = []
    for figure_id, stem, logical_name in FIGURES:
        rows.append(
            {
                "figure": figure_id,
                "stem": stem,
                "logical_name": logical_name,
                "outputs": "; ".join(f"{fmt}/{stem}.{fmt}" for fmt in FORMATS),
                "output_formats": ";".join(FORMATS),
                "status": "regenerated",
                "warnings": "",
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "figure_manifest.csv", index=False)


def validate_outputs() -> pd.DataFrame:
    rows = []
    failures = []
    names_seen: set[str] = set()
    for _, stem, _ in FIGURES:
        for fmt in FORMATS:
            path = OUTPUT_ROOT / fmt / f"{stem}.{fmt}"
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            rows.append({"stem": stem, "format": fmt, "path": str(path), "exists": exists, "size_bytes": size})
            if not exists or size <= 0:
                failures.append(f"Missing or empty output: {path}")
            if path.name in names_seen:
                failures.append(f"Duplicate filename: {path.name}")
            names_seen.add(path.name)
            if fmt == "png" and exists:
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as exc:  # pragma: no cover - validation path
                    failures.append(f"PNG cannot be opened: {path} ({exc})")

    for overview in (
        "figure_overview_complete_highres.png",
        "full_set_overview_highres.png",
        "refined_five_overview_highres.png",
    ):
        path = OUTPUT_ROOT / overview
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append({"stem": overview, "format": "overview_png", "path": str(path), "exists": exists, "size_bytes": size})
        if not exists or size <= 0:
            failures.append(f"Missing overview PNG thumbnail: {path}")
        elif overview == "figure_overview_complete_highres.png":
            with Image.open(path) as im:
                im.verify()

    qa = pd.DataFrame(rows)
    qa.to_csv(DATA_OUT / "final_output_validation.csv", index=False)
    if failures:
        raise RuntimeError("Final figure validation failed:\n- " + "\n- ".join(failures))
    return qa


def write_log(qa: pd.DataFrame) -> None:
    counts = qa.groupby("format")["exists"].sum().to_dict()
    log = [
        "Generated final Online Lyra publication figures.",
        f"Output directory: {OUTPUT_ROOT}",
        f"Figure groups: {len(FIGURES)}",
        f"PNG: {counts.get('png', 0)}",
        f"PDF: {counts.get('pdf', 0)}",
        f"SVG: {counts.get('svg', 0)}",
        f"TIFF: {counts.get('tiff', 0)}",
        f"Overview PNG thumbnails: {counts.get('overview_png', 0)}",
        "Warnings: 0",
    ]
    (OUTPUT_ROOT / "generation.log").write_text("\n".join(log) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Export independent panels from existing final PDFs without regenerating figures.",
    )
    args = parser.parse_args()

    ensure_output_directories()
    if args.panels_only:
        export_individual_panels()
        print(f"Exported independent panels to {OUTPUT_ROOT / 'panels'}")
        return

    install_inside_legend_policy()
    redirect_existing_modules()
    generate_core_data_and_figures()
    generate_refined_renderers()
    generate_final_overrides()
    export_individual_panels()
    make_overview_pngs()
    write_manifest()
    qa = validate_outputs()
    write_log(qa)

    counts = qa.groupby("format")["exists"].sum().to_dict()
    print(f"Generated {len(FIGURES)} figure groups")
    print(f"PNG: {counts.get('png', 0)}")
    print(f"PDF: {counts.get('pdf', 0)}")
    print(f"SVG: {counts.get('svg', 0)}")
    print(f"TIFF: {counts.get('tiff', 0)}")
    print(f"Overview PNG thumbnails: {counts.get('overview_png', 0)}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("Warnings: 0")


if __name__ == "__main__":
    main()
