"""Create a LaTeX-style PDF digest of the PV forecasting experiments.

The script writes both a .tex source and a PDF preview. The PDF is generated
with ReportLab so it can run on machines without a TeX distribution.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


MODEL_ORDER = [
    "online_lyra_b06_final",
    "offline_lyra_final",
    "official_dlinear",
    "official_nlinear",
    "official_linear",
    "itransformer",
    "official_patchtst",
    "timekan",
    "phaseformer",
    "mixlinear",
    "olivia_scratch",
    "seasonal_naive",
    "persistence",
]
MODEL_LABEL = {
    "online_lyra_b06_final": "Online Lyra",
    "offline_lyra_final": "Offline Lyra",
    "official_dlinear": "DLinear",
    "official_nlinear": "NLinear",
    "official_linear": "Linear",
    "itransformer": "iTransformer",
    "official_patchtst": "PatchTST",
    "timekan": "TimeKAN",
    "phaseformer": "PhaseFormer",
    "mixlinear": "MixLinear",
    "olivia_scratch": "Olivia",
    "seasonal_naive": "Seasonal Naive",
    "persistence": "Persistence",
}
MODEL_CLASS = {
    "online_lyra_b06_final": "Lyra family",
    "offline_lyra_final": "Lyra family",
    "official_dlinear": "Linear",
    "official_nlinear": "Linear",
    "official_linear": "Linear",
    "itransformer": "Deep TSF",
    "official_patchtst": "Deep TSF",
    "timekan": "Deep TSF",
    "phaseformer": "Deep TSF",
    "mixlinear": "Deep TSF",
    "olivia_scratch": "Foundation-style",
    "seasonal_naive": "Naive",
    "persistence": "Naive",
}
ABLATION_LABEL = {
    "b00_c06_anchor": "Anchor: residual loss",
    "b01_plus_instance_mean": "+ instance mean",
    "b02_plus_revin": "+ RevIN norm",
    "b03_plus_dropout030": "+ dropout 0.30",
    "b04_plus_lookback192": "+ lookback 192",
    "b05_plus_ridge1": "+ ridge 1.0",
    "b06_forecast_loss_fixed_scale": "+ forecast loss (final)",
    "b07_revin_forecast_fixed_scale": "+ RevIN + forecast loss",
}


def fmt_num(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def pdf_cell(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def pdf_mark(value: float, rank: int, is_online_lyra: bool, style: ParagraphStyle) -> Paragraph:
    text = fmt_num(value, 3)
    if rank == 1:
        text = f"<b>{text}</b>"
        if is_online_lyra:
            text += "†"
    elif rank <= 3:
        text = f"<u>{text}</u>"
    return pdf_cell(text, style)


def tex_escape(value: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(value))


def tex_mark(value: float, rank: int, is_online_lyra: bool) -> str:
    text = fmt_num(value, 3)
    if rank == 1:
        text = rf"\textbf{{{text}}}"
        if is_online_lyra:
            text += r"\textsuperscript{\dag}"
    elif rank <= 3:
        text = rf"\underline{{{text}}}"
    return text


def read_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return {
        "overall": pd.read_csv(args.tables_dir / "main_overall_average.csv"),
        "horizon": pd.read_csv(args.tables_dir / "main_average_by_horizon.csv"),
        "rank": pd.read_csv(args.tables_dir / "main_nrmse_rank_summary.csv"),
        "resource": pd.read_csv(args.resource_csv),
        "ablation": pd.read_csv(args.ablation_csv),
        "ablation_vs": pd.read_csv(args.ablation_vs_csv),
    }


def make_main_rows(horizon: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for horizon_value in sorted(horizon["horizon"].unique()):
        subset = horizon[horizon["horizon"].eq(horizon_value)].set_index("model")
        for metric in ["nrmse", "nmae"]:
            ranks = subset[metric].rank(method="min", ascending=True).astype(int)
            rows.append(
                {
                    "horizon": int(horizon_value),
                    "metric": metric.upper(),
                    "values": {
                        model: (
                            float(subset.loc[model, metric]),
                            int(ranks.loc[model]),
                        )
                        for model in MODEL_ORDER
                        if model in subset.index
                    },
                }
            )
    return rows


def table_style(font_size: float = 6.0) -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.0),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
            ("LINEBELOW", (0, 1), (-1, 1), 0.7, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
            ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def make_pdf(data: dict[str, pd.DataFrame], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "DigestTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        spaceBefore=7,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.2,
        leading=7.2,
        alignment=TA_CENTER,
    )
    small_left = ParagraphStyle(
        "SmallLeft",
        parent=small,
        alignment=TA_LEFT,
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=9 * mm,
        bottomMargin=9 * mm,
    )
    story = []
    story.append(Paragraph("Online Lyra PV Results Digest", title))
    story.append(
        Paragraph(
            "Protocol: 8 PV sites, horizons H={4,12,24,48,96}, lookback L=96, "
            "three seeds (2026, 2027, 2028). Metrics are averaged over sites and seeds. "
            "Lower is better. Best results are bold, top-3 results are underlined, "
            "and an Online Lyra rank-1 cell carries a dagger.",
            body,
        )
    )
    story.append(Spacer(1, 5))

    story.append(Paragraph("Table 1. TSF-style main comparison by horizon", h2))
    header = [pdf_cell("<b>H</b>", small), pdf_cell("<b>Metric</b>", small)]
    header += [pdf_cell(f"<b>{html.escape(MODEL_LABEL[m])}</b>", small) for m in MODEL_ORDER]
    subheader = [pdf_cell("", small), pdf_cell("", small)]
    subheader += [pdf_cell(html.escape(MODEL_CLASS[m]), small) for m in MODEL_ORDER]
    table_data = [header, subheader]
    for row in make_main_rows(data["horizon"]):
        values = [pdf_cell(str(row["horizon"]), small), pdf_cell(str(row["metric"]), small)]
        for model in MODEL_ORDER:
            if model not in row["values"]:
                values.append(pdf_cell("-", small))
                continue
            value, rank = row["values"][model]
            values.append(pdf_mark(value, rank, model == "online_lyra_b06_final", small))
        table_data.append(values)
    widths = [10 * mm, 16 * mm] + [18.5 * mm] * len(MODEL_ORDER)
    main_table = Table(table_data, colWidths=widths, repeatRows=2, hAlign="LEFT")
    main_table.setStyle(table_style(5.8))
    story.append(main_table)
    story.append(
        Paragraph(
            "Olivia is reported as a foundation-style baseline; Linear/DLinear/NLinear are linear baselines; "
            "Seasonal Naive and Persistence are non-parametric references.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Table 2. Overall average and rank diagnostics", h2))
    overall = data["overall"].set_index("model")
    rank = data["rank"].set_index("model")
    rows = [[
        pdf_cell("<b>Class</b>", small),
        pdf_cell("<b>Model</b>", small),
        pdf_cell("<b>Rank</b>", small),
        pdf_cell("<b>nRMSE</b>", small),
        pdf_cell("<b>nMAE</b>", small),
        pdf_cell("<b>Day nRMSE</b>", small),
        pdf_cell("<b>Mean rank</b>", small),
        pdf_cell("<b>Top-3 rate</b>", small),
    ]]
    for model in MODEL_ORDER:
        if model not in overall.index:
            continue
        row = overall.loc[model]
        rrow = rank.loc[model] if model in rank.index else None
        rows.append([
            pdf_cell(MODEL_CLASS[model], small_left),
            pdf_cell(MODEL_LABEL[model], small_left),
            pdf_cell(str(int(row["overall_rank"])), small),
            pdf_cell(fmt_num(float(row["nrmse"]), 4), small),
            pdf_cell(fmt_num(float(row["nmae"]), 4), small),
            pdf_cell(fmt_num(float(row["daytime_nrmse"]), 4), small),
            pdf_cell(fmt_num(float(rrow["mean_rank"]), 2) if rrow is not None else "-", small),
            pdf_cell(fmt_num(float(rrow["top3_rate"]) * 100, 1) + "%" if rrow is not None else "-", small),
        ])
    overall_table = Table(rows, colWidths=[30 * mm, 34 * mm, 14 * mm, 22 * mm, 22 * mm, 24 * mm, 22 * mm, 24 * mm])
    overall_table.setStyle(table_style(7.0))
    story.append(overall_table)

    story.append(Paragraph("Table 3. Resource efficiency benchmark", h2))
    res = data["resource"].copy()
    res["class"] = res["model"].map(MODEL_CLASS)
    res = res[res["model"].isin(MODEL_ORDER)]
    res["order"] = res["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    res = res.sort_values("order")
    rows = [[
        pdf_cell("<b>Class</b>", small),
        pdf_cell("<b>Model</b>", small),
        pdf_cell("<b>Params K</b>", small),
        pdf_cell("<b>Ckpt MB</b>", small),
        pdf_cell("<b>Latency ms</b>", small),
        pdf_cell("<b>Torch alloc MB</b>", small),
        pdf_cell("<b>Torch reserved MB</b>", small),
        pdf_cell("<b>nvidia-smi extra MB</b>", small),
    ]]
    for _, row in res.iterrows():
        rows.append([
            pdf_cell(str(row["class"]), small_left),
            pdf_cell(str(row["model_display"]), small_left),
            pdf_cell(fmt_num(float(row["params_k"]), 1), small),
            pdf_cell(fmt_num(float(row["checkpoint_size_mb"]), 3), small),
            pdf_cell(f"{float(row['latency_mean_ms']):.3f}±{float(row['latency_std_ms']):.3f}", small),
            pdf_cell(fmt_num(float(row["torch_peak_allocated_mb"]), 2), small),
            pdf_cell(fmt_num(float(row["torch_peak_reserved_mb"]), 1), small),
            pdf_cell(fmt_num(float(row["nvidia_smi_extra_mb"]), 1), small),
        ])
    res_table = Table(rows, colWidths=[30 * mm, 35 * mm, 18 * mm, 18 * mm, 29 * mm, 24 * mm, 27 * mm, 30 * mm])
    res_table.setStyle(table_style(7.0))
    story.append(res_table)
    story.append(
        Paragraph(
            "Benchmark: site 1, H=96, batch size 1, 30 warm-up and 300 timed repeats on RTX 3090. "
            "nvidia-smi memory is an external reference and includes CUDA context overhead.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Table 4. Online Lyra scoped ablation", h2))
    ab = data["ablation"].copy()
    ab["label"] = ab["config"].map(ABLATION_LABEL).fillna(ab["config"])
    ab = ab.sort_values(["mean_nrmse", "mean_nmae"])
    rows = [[
        pdf_cell("<b>Variant</b>", small),
        pdf_cell("<b>Rows</b>", small),
        pdf_cell("<b>nRMSE</b>", small),
        pdf_cell("<b>nMAE</b>", small),
        pdf_cell("<b>Day nRMSE</b>", small),
        pdf_cell("<b>Skill vs persistence</b>", small),
        pdf_cell("<b>Gate Lyra %</b>", small),
        pdf_cell("<b>Update ms</b>", small),
    ]]
    for i, (_, row) in enumerate(ab.iterrows(), start=1):
        rank = 1 if i == 1 else (2 if i <= 3 else 4)
        label = str(row["label"])
        if row["config"] == "b06_forecast_loss_fixed_scale":
            label = "<b>" + html.escape(label) + "</b>"
        else:
            label = html.escape(label)
        rows.append([
            pdf_cell(label, small_left),
            pdf_cell(str(int(row["rows"])), small),
            pdf_mark(float(row["mean_nrmse"]), rank, False, small),
            pdf_cell(fmt_num(float(row["mean_nmae"]), 3), small),
            pdf_cell(fmt_num(float(row["mean_daytime_nrmse"]), 3), small),
            pdf_cell(fmt_num(float(row["mean_skill_vs_persistence"]), 3), small),
            pdf_cell(fmt_num(float(row["mean_gate_lyra_fraction"]) * 100, 1) + "%", small),
            pdf_cell(fmt_num(float(row["mean_update_latency_ms"]), 3), small),
        ])
    ab_table = Table(rows, colWidths=[56 * mm, 12 * mm, 18 * mm, 18 * mm, 23 * mm, 30 * mm, 22 * mm, 20 * mm])
    ab_table.setStyle(table_style(7.0))
    story.append(ab_table)
    story.append(
        Paragraph(
            "This table summarizes the configured site-1, three-horizon ablation protocol. "
            "It should not be interpreted as an aggregate over the eight-site evaluation.",
            body,
        )
    )

    story.append(Paragraph("Table 5. Artifact and logging status", h2))
    rows = [
        [pdf_cell("<b>Experiment</b>", small), pdf_cell("<b>Stored artifacts</b>", small), pdf_cell("<b>Status</b>", small)],
        [
            pdf_cell("Online Lyra final", small_left),
            pdf_cell("24 summaries; 120 update/training logs; 120 checkpoints; 120 prediction files", small_left),
            pdf_cell("Complete", small),
        ],
        [
            pdf_cell("Offline Lyra control", small_left),
            pdf_cell("3 summaries; 120 epoch logs; 120 training CSVs; 120 checkpoints; 120 prediction files", small_left),
            pdf_cell("Complete", small),
        ],
        [
            pdf_cell("Static baselines", small_left),
            pdf_cell("Aggregate fair-protocol summaries; generated per-run artifacts are stored outside this repository", small_left),
            pdf_cell("Metrics complete", small),
        ],
        [
            pdf_cell("Efficiency benchmark", small_left),
            pdf_cell("Formal one-process-per-model benchmark table and driver log", small_left),
            pdf_cell("Complete", small),
        ],
    ]
    status_table = Table(rows, colWidths=[45 * mm, 125 * mm, 30 * mm])
    status_table.setStyle(table_style(7.0))
    story.append(status_table)
    story.append(
        Paragraph(
            "Online Lyra records online update logs rather than conventional validation-loss-per-epoch logs. "
            "The Offline Lyra control records conventional epoch-level training and validation logs.",
            body,
        )
    )

    doc.build(story)


def latex_main_table(data: dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{TSF-style PV forecasting comparison averaged over 8 sites and 3 seeds. Lower is better.}")
    lines.append(r"\setlength{\tabcolsep}{2.2pt}")
    colspec = "ll" + "c" * len(MODEL_ORDER)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header = ["H", "Metric"] + [tex_escape(MODEL_LABEL[m]) for m in MODEL_ORDER]
    lines.append(" & ".join(header) + r" \\")
    class_row = ["", ""] + [tex_escape(MODEL_CLASS[m]) for m in MODEL_ORDER]
    lines.append(" & ".join(class_row) + r" \\")
    lines.append(r"\midrule")
    for row in make_main_rows(data["horizon"]):
        cells = [str(row["horizon"]), str(row["metric"])]
        for model in MODEL_ORDER:
            if model not in row["values"]:
                cells.append("-")
                continue
            value, rank = row["values"][model]
            cells.append(tex_mark(value, rank, model == "online_lyra_b06_final"))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption*{\footnotesize Olivia is a foundation-style baseline. "
        r"Bold indicates best; underlining indicates top-3; \dag marks an Online Lyra rank-1 cell.}"
    )
    lines.append(r"\end{table}")
    lines.append(r"\end{landscape}")
    return "\n".join(lines)


def latex_simple_table(data: dict[str, pd.DataFrame], name: str) -> str:
    if name == "overall":
        df = data["overall"].copy()
        df["Class"] = df["model"].map(MODEL_CLASS)
        df["Model"] = df["model"].map(MODEL_LABEL)
        cols = ["Class", "Model", "overall_rank", "nrmse", "nmae", "daytime_nrmse", "daytime_nmae"]
        labels = ["Class", "Model", "Rank", "nRMSE", "nMAE", "Day nRMSE", "Day nMAE"]
        caption = "Overall average metrics and model groups."
    elif name == "resource":
        df = data["resource"].copy()
        df = df[df["model"].isin(MODEL_ORDER)].copy()
        df["Class"] = df["model"].map(MODEL_CLASS)
        df["Model"] = df["model_display"]
        cols = ["Class", "Model", "params_k", "checkpoint_size_mb", "latency_mean_ms", "torch_peak_allocated_mb", "nvidia_smi_extra_mb"]
        labels = ["Class", "Model", "Params K", "Ckpt MB", "Latency ms", "Torch MB", "nvidia-smi extra MB"]
        caption = "Resource efficiency benchmark on site 1, H=96, batch size 1."
    else:
        df = data["ablation"].copy()
        df["Variant"] = df["config"].map(ABLATION_LABEL).fillna(df["config"])
        cols = ["Variant", "rows", "mean_nrmse", "mean_nmae", "mean_daytime_nrmse", "mean_skill_vs_persistence"]
        labels = ["Variant", "Rows", "nRMSE", "nMAE", "Day nRMSE", "Skill"]
        caption = "Site-1, three-horizon ablation for Online Lyra."
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", r"\begin{tabular}{l" + "c" * (len(cols) - 1) + r"}", r"\toprule"]
    lines.append(" & ".join(labels) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append(fmt_num(value, 4 if abs(value) < 1 else 2))
            else:
                cells.append(tex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_latex(data: dict[str, pd.DataFrame], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[a4paper,margin=0.7in]{geometry}",
            r"\usepackage{booktabs,array,caption,pdflscape}",
            r"\usepackage[normalem]{ulem}",
            r"\usepackage{graphicx}",
            r"\begin{document}",
            r"\title{Online Lyra PV Results Digest}",
            r"\author{}",
            r"\date{}",
            r"\maketitle",
            r"Protocol: 8 PV sites, horizons $H \in \{4,12,24,48,96\}$, lookback $L=96$, and three seeds. Lower is better.",
            latex_main_table(data),
            latex_simple_table(data, "overall"),
            latex_simple_table(data, "resource"),
            latex_simple_table(data, "ablation"),
            r"\end{document}",
        ]
    )
    out_tex.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PV results digest PDF and LaTeX source")
    parser.add_argument(
        "--tables_dir",
        type=Path,
        default=Path("outputs/server_runs_20260716/offline_lyra_final_20260717_165425/paper_tables_with_offline_lyra"),
    )
    parser.add_argument(
        "--resource_csv",
        type=Path,
        default=Path("outputs/server_runs_20260716/resource_benchmark_20260717_1610/resource_efficiency_table.csv"),
    )
    parser.add_argument(
        "--ablation_csv",
        type=Path,
        default=Path("outputs/server_runs_20260716/tuning_online_lyra_ablation_20260716_195050/tuning_config_summary.csv"),
    )
    parser.add_argument(
        "--ablation_vs_csv",
        type=Path,
        default=Path("outputs/server_runs_20260716/tuning_online_lyra_ablation_20260716_195050/tuning_vs_static_by_config.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_result_pack"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / "online_lyra_results_digest.tex"
    pdf_path = args.output_dir / "online_lyra_results_digest.pdf"
    make_latex(data, tex_path)
    make_pdf(data, pdf_path)
    print(f"Saved LaTeX source: {tex_path}")
    print(f"Saved PDF preview: {pdf_path}")


if __name__ == "__main__":
    main()
