"""Build paper-facing LaTeX tables for the Online Lyra PV study."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "online_lyra_b06_final",
    "offline_lyra_final",
    "itransformer",
    "olivia_scratch",
    "official_dlinear",
    "official_nlinear",
    "phaseformer",
    "timekan",
    "official_patchtst",
    "official_linear",
    "mixlinear",
    "seasonal_naive",
    "persistence",
]

MODEL_LABEL = {
    "online_lyra_b06_final": "Online Lyra",
    "offline_lyra_final": "Offline Lyra",
    "official_dlinear": "DLinear",
    "official_nlinear": "NLinear",
    "official_linear": "Linear",
    "official_patchtst": "PatchTST",
    "itransformer": "iTransformer",
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

LOWER_IS_BETTER = {
    "nrmse",
    "nmae",
    "daytime_nrmse",
    "daytime_nmae",
    "overall_rank",
    "mean_rank",
    "median_rank",
    "worst_rank",
    "params_k",
    "checkpoint_size_mb",
    "gpu_latency_ms",
    "gpu_alloc_mb",
    "gpu_smi_extra_mb",
    "cpu_latency_ms",
}

HIGHER_IS_BETTER = {
    "top1_rate",
    "top3_rate",
    "skill_vs_persistence",
    "relative_improvement_pct",
    "win_rate",
}

METRIC_LABELS = {
    "nrmse": r"nRMSE$\downarrow$",
    "nmae": r"nMAE$\downarrow$",
    "daytime_nrmse": r"Day nRMSE$\downarrow$",
    "daytime_nmae": r"Day nMAE$\downarrow$",
}


def tex_escape(value: object) -> str:
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


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if not math.isfinite(val):
        return "-"
    return f"{val:.{digits}f}"


def fmt_p(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(val):
        return "-"
    if val < 1e-3:
        return f"{val:.1e}"
    return f"{val:.3f}"


def model_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def best_mask(values: pd.Series, higher: bool = False) -> pd.Series:
    finite = pd.to_numeric(values, errors="coerce")
    if finite.dropna().empty:
        return pd.Series(False, index=values.index)
    best = finite.max() if higher else finite.min()
    return finite.eq(best)


def maybe_bold(text: str, bold: bool) -> str:
    return rf"\textbf{{{text}}}" if bold else text


def mark_main(value: float, rank: int, is_online_lyra: bool) -> str:
    text = fmt(value, 4)
    if rank == 1:
        text = rf"\textbf{{{text}}}"
        if is_online_lyra:
            text += r"\textsuperscript{\textdagger}"
    elif rank <= 3:
        text = rf"\underline{{{text}}}"
    return text


def table_block(
    caption: str,
    label: str,
    colspec: str,
    header: list[str],
    rows: list[list[str]],
    *,
    size: str = r"\small",
    landscape: bool = False,
    resize: bool = False,
    notes: str | None = None,
) -> str:
    lines: list[str] = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend([r"\begin{table}[t]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}"])
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"])
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append("}%")
    if notes:
        lines.append(rf"\caption*{{\footnotesize {notes}}}")
    lines.append(r"\end{table}")
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def write_table(output_dir: Path, filename: str, content: str) -> Path:
    path = output_dir / filename
    path.write_text(content + "\n", encoding="utf-8")
    return path


def main_tsf_table(horizon: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for horizon_value in sorted(horizon["horizon"].unique()):
        subset = horizon[horizon["horizon"].eq(horizon_value)].set_index("model")
        for metric in ["nrmse", "nmae"]:
            ranks = subset[metric].rank(method="min", ascending=True).astype(int)
            row = [str(int(horizon_value)), METRIC_LABELS[metric]]
            for model in MODEL_ORDER:
                if model not in subset.index:
                    row.append("-")
                    continue
                row.append(mark_main(float(subset.loc[model, metric]), int(ranks.loc[model]), model == "online_lyra_b06_final"))
            rows.append(row)
    header = ["H", "Metric"] + [tex_escape(MODEL_LABEL[m]) for m in MODEL_ORDER]
    return table_block(
        "Main PV forecasting comparison averaged over 8 sites and 3 seeds.",
        "tab:main_pv_tsf",
        "ll" + "c" * len(MODEL_ORDER),
        header,
        rows,
        size=r"\scriptsize",
        landscape=True,
        resize=True,
        notes=r"Lower is better. Best results are bold, top-3 results are underlined, and \textdagger{} marks an Online Lyra rank-1 cell. Olivia is reported as a foundation-style baseline.",
    )


def overall_table(overall: pd.DataFrame, rank: pd.DataFrame) -> str:
    rank = rank.set_index("model")
    df = overall.copy()
    df["Class"] = df["model"].map(MODEL_CLASS)
    df["Model"] = df["model"].map(MODEL_LABEL)
    df["mean_rank"] = df["model"].map(rank["mean_rank"])
    df["top3_rate"] = df["model"].map(rank["top3_rate"])
    cols = ["overall_rank", "nrmse", "nmae", "daytime_nrmse", "daytime_nmae", "mean_rank", "top3_rate"]
    masks = {
        col: best_mask(df[col], higher=col in HIGHER_IS_BETTER)
        for col in cols
    }
    rows = []
    for _, row in df.iterrows():
        idx = row.name
        rows.append(
            [
                tex_escape(row["Class"]),
                tex_escape(row["Model"]),
                maybe_bold(str(int(row["overall_rank"])), bool(masks["overall_rank"].loc[idx])),
                maybe_bold(fmt(row["nrmse"], 4), bool(masks["nrmse"].loc[idx])),
                maybe_bold(fmt(row["nmae"], 4), bool(masks["nmae"].loc[idx])),
                maybe_bold(fmt(row["daytime_nrmse"], 4), bool(masks["daytime_nrmse"].loc[idx])),
                maybe_bold(fmt(row["daytime_nmae"], 4), bool(masks["daytime_nmae"].loc[idx])),
                maybe_bold(fmt(row["mean_rank"], 2), bool(masks["mean_rank"].loc[idx])),
                maybe_bold(f"{float(row['top3_rate']) * 100:.1f}\\%", bool(masks["top3_rate"].loc[idx])),
            ]
        )
    return table_block(
        "Overall accuracy and rank diagnostics.",
        "tab:overall_rank",
        "llccccccc",
        ["Class", "Model", "Rank", r"nRMSE$\downarrow$", r"nMAE$\downarrow$", r"Day nRMSE$\downarrow$", r"Day nMAE$\downarrow$", r"Mean rank$\downarrow$", r"Top-3 rate$\uparrow$"],
        rows,
        size=r"\scriptsize",
        resize=True,
    )


def resource_table(gpu: pd.DataFrame, cpu: pd.DataFrame) -> str:
    g = gpu.rename(
        columns={
            "latency_mean_ms": "gpu_latency_ms",
            "torch_peak_allocated_mb": "gpu_alloc_mb",
            "nvidia_smi_extra_mb": "gpu_smi_extra_mb",
        }
    )
    c = cpu.rename(columns={"latency_mean_ms": "cpu_latency_ms"})
    df = g[["model", "model_display", "params_k", "checkpoint_size_mb", "gpu_latency_ms", "gpu_alloc_mb", "gpu_smi_extra_mb"]].merge(
        c[["model", "cpu_latency_ms"]],
        on="model",
        how="outer",
    )
    df["order"] = df["model"].map(model_key)
    df["model_display"] = df["model_display"].fillna(df["model"].map(MODEL_LABEL))
    df = df[df["model"].isin(MODEL_ORDER)].sort_values("order").reset_index(drop=True)
    cols = ["params_k", "checkpoint_size_mb", "gpu_latency_ms", "gpu_alloc_mb", "gpu_smi_extra_mb", "cpu_latency_ms"]
    masks = {col: best_mask(df[col], higher=False) for col in cols}
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            [
                tex_escape(MODEL_CLASS.get(row["model"], "")),
                tex_escape(row["model_display"]),
                maybe_bold(fmt(row["params_k"], 1), bool(masks["params_k"].loc[idx])),
                maybe_bold(fmt(row["checkpoint_size_mb"], 3), bool(masks["checkpoint_size_mb"].loc[idx])),
                maybe_bold(fmt(row["gpu_latency_ms"], 3), bool(masks["gpu_latency_ms"].loc[idx])),
                maybe_bold(fmt(row["gpu_alloc_mb"], 2), bool(masks["gpu_alloc_mb"].loc[idx])),
                maybe_bold(fmt(row["gpu_smi_extra_mb"], 1), bool(masks["gpu_smi_extra_mb"].loc[idx])),
                maybe_bold(fmt(row["cpu_latency_ms"], 3), bool(masks["cpu_latency_ms"].loc[idx])),
            ]
        )
    return table_block(
        "Deployment efficiency benchmark on site 1, H=96, seed 2028.",
        "tab:deployment_efficiency",
        "llcccccc",
        ["Class", "Model", r"Params K$\downarrow$", r"Ckpt MB$\downarrow$", r"GPU ms$\downarrow$", r"CUDA alloc MB$\downarrow$", r"smi extra MB$\downarrow$", r"CPU ms$\downarrow$"],
        rows,
        size=r"\scriptsize",
        resize=True,
        notes="Latency is measured after warm-up. GPU memory columns are one-process benchmark measurements; CPU latency is a separate CPU-only benchmark.",
    )


def simple_metric_table(df: pd.DataFrame, caption: str, label: str, name_col: str) -> str:
    cols = ["nrmse", "nmae", "daytime_nrmse", "daytime_nmae"]
    if "skill_vs_persistence" in df.columns:
        cols.append("skill_vs_persistence")
    masks = {col: best_mask(df[col], higher=col in HIGHER_IS_BETTER) for col in cols}
    rows = []
    for idx, row in df.iterrows():
        out = [tex_escape(row[name_col]), str(int(row["conditions"]))]
        for col in cols:
            digits = 3 if col != "skill_vs_persistence" else 3
            out.append(maybe_bold(fmt(row[col], digits), bool(masks[col].loc[idx])))
        rows.append(out)
    headers = ["Variant", "N"] + [METRIC_LABELS.get(col, r"Skill$\uparrow$") for col in cols]
    return table_block(caption, label, "l" + "c" * (len(headers) - 1), headers, rows, size=r"\scriptsize", resize=True)


def robustness_compact(df: pd.DataFrame, group_col: str, caption: str, label: str) -> str:
    rows = []
    for group, sub in df.groupby(group_col, sort=True):
        sub = sub.copy()
        online = sub[sub["model"].eq("online_lyra_b06_final")]
        offline = sub[sub["model"].eq("offline_lyra_final")]
        competitors = sub[~sub["model"].isin(["online_lyra_b06_final", "offline_lyra_final", "persistence", "seasonal_naive"])]
        best_static = competitors.sort_values("nrmse").head(1)
        if online.empty or best_static.empty:
            continue
        online_val = float(online["nrmse"].iloc[0])
        best_val = float(best_static["nrmse"].iloc[0])
        offline_val = float(offline["nrmse"].iloc[0]) if not offline.empty else float("nan")
        vals = pd.Series([online_val, best_val, offline_val], index=["online", "best", "offline"])
        bold = best_mask(vals)
        improvement = (best_val - online_val) / best_val * 100.0
        group_text = tex_escape(group)
        if group_col == "horizon":
            group_text = str(int(group))
        rows.append(
            [
                group_text,
                maybe_bold(fmt(online_val, 4), bool(bold.loc["online"])),
                tex_escape(best_static["model_display"].iloc[0]),
                maybe_bold(fmt(best_val, 4), bool(bold.loc["best"])),
                maybe_bold(fmt(offline_val, 4), bool(bold.loc["offline"])),
                fmt(improvement, 2),
            ]
        )
    return table_block(
        caption,
        label,
        "lccccc",
        ["Group", r"Online Lyra nRMSE$\downarrow$", "Best static", r"Best static nRMSE$\downarrow$", r"Offline Lyra nRMSE$\downarrow$", r"Gain vs static (\%)$\uparrow$"],
        rows,
        size=r"\scriptsize",
        resize=True,
    )


def prediction_proxy_table(df: pd.DataFrame) -> str:
    use = df[df["model"].isin(["online_lyra_b06_final", "offline_lyra_final"])].copy()
    grouped = (
        use.groupby(["group_type", "group", "model", "model_display"], as_index=False)
        .agg(nrmse=("nrmse", "mean"), nmae=("nmae", "mean"), skill_vs_persistence=("skill_vs_persistence", "mean"))
    )
    rows = []
    for (group_type, group), sub in grouped.groupby(["group_type", "group"], sort=True):
        online = sub[sub["model"].eq("online_lyra_b06_final")]
        offline = sub[sub["model"].eq("offline_lyra_final")]
        if online.empty or offline.empty:
            continue
        vals = pd.Series([float(online["nrmse"].iloc[0]), float(offline["nrmse"].iloc[0])], index=["online", "offline"])
        bold = best_mask(vals)
        rows.append(
            [
                tex_escape(group_type),
                tex_escape(group.replace("_", " ")),
                maybe_bold(fmt(online["nrmse"].iloc[0], 4), bool(bold.loc["online"])),
                maybe_bold(fmt(offline["nrmse"].iloc[0], 4), bool(bold.loc["offline"])),
                fmt(online["nmae"].iloc[0], 4),
                fmt(offline["nmae"].iloc[0], 4),
                fmt(online["skill_vs_persistence"].iloc[0], 3),
            ]
        )
    return table_block(
        "Prediction-level robustness under day/night and PV-output proxy groups.",
        "tab:prediction_proxy_robustness",
        "llccccc",
        ["Type", "Group", r"Online nRMSE$\downarrow$", r"Offline nRMSE$\downarrow$", r"Online nMAE$\downarrow$", r"Offline nMAE$\downarrow$", r"Online skill$\uparrow$"],
        rows,
        size=r"\scriptsize",
        resize=True,
    )


def stat_table(df: pd.DataFrame) -> str:
    use = df[df["metric"].isin(["nrmse", "daytime_nrmse"])].copy()
    order = {model: i for i, model in enumerate(MODEL_ORDER)}
    use["order"] = use["baseline_model"].map(order)
    use = use.sort_values(["metric", "order"]).reset_index(drop=True)
    rows = []
    for _, row in use.iterrows():
        sig = float(row["wilcoxon_p_less"]) < 0.05 and float(row["relative_improvement_pct"]) > 0
        p_text = maybe_bold(fmt_p(row["wilcoxon_p_less"]), sig)
        rows.append(
            [
                METRIC_LABELS[row["metric"]],
                tex_escape(row["baseline_display"]),
                fmt(row["online_mean"], 4),
                fmt(row["baseline_mean"], 4),
                maybe_bold(fmt(row["relative_improvement_pct"], 2), sig),
                fmt(float(row["win_rate"]) * 100.0, 1),
                p_text,
            ]
        )
    return table_block(
        "Paired statistical tests comparing Online Lyra with each baseline.",
        "tab:paired_tests",
        "llccccc",
        ["Metric", "Baseline", "Online mean", "Baseline mean", r"Rel. gain (\%)$\uparrow$", r"Win rate (\%)$\uparrow$", r"Wilcoxon $p$"],
        rows,
        size=r"\scriptsize",
        resize=True,
        notes=r"Bold indicates statistically significant positive improvement at $p<0.05$ under the one-sided Wilcoxon signed-rank test.",
    )


def figure_plan_table() -> str:
    rows = [
        ["Fig. 1", "Overall pipeline", "Architecture schematic: Hankel/SVD trend, spectral residual learner, online update, persistence gate.", "Method specification; no data dependency", "Specified"],
        ["Fig. 2", "Main accuracy by horizon", "Line/bar plot of nRMSE and nMAE across horizons for Online Lyra and top baselines.", "figure_data/main_average_by_horizon.csv", "Available"],
        ["Fig. 3", "Site/capacity robustness", "Heatmap or grouped bars over 8 sites and capacity groups.", "robustness_by_site/capacity_all_models.csv", "Available"],
        ["Fig. 4", "Case study trace", "True PV power vs Online Lyra vs persistence, with gate decisions.", "figure_data/case_trace_site1_h96_seed2028.csv", "Available"],
        ["Fig. 5", "Gate behavior", "Gate Lyra fraction by site/horizon and selected time windows.", "figure_data/gate_profile_*.csv", "Available"],
        ["Fig. 7", "Daily error distribution", "Daily nRMSE/nMAE distribution or box plot.", "figure_data/daily_error_data_h96.csv", "Available"],
        ["Fig. 8", "Deployment efficiency", "Params/checkpoint/latency/memory comparison.", "resource_efficiency_table.csv + resource_cpu_table.csv", "Available"],
        ["Fig. 9", "Core ablation", "Bar chart of full model vs removed components.", "core_ablation/core_ablation_summary.csv", "Available"],
        ["Fig. 10", "Statistical evidence", "Relative gain and Wilcoxon p-values against baselines.", "stat_tests/paired_tests_online_lyra_vs_baselines.csv", "Available"],
    ]
    return table_block(
        "Figure specifications and corresponding source data.",
        "tab:figure_plan",
        "lllll",
        ["ID", "Figure", "Content", "Data source", "Status"],
        [[tex_escape(cell) for cell in row] for row in rows],
        size=r"\scriptsize",
        resize=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LaTeX table pack for Online Lyra PV paper")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paper_result_pack/latex_tables"))
    parser.add_argument("--tables_dir", type=Path, default=Path("outputs/paper_result_pack/server_tables_with_offline_lyra"))
    parser.add_argument("--robustness_dir", type=Path, default=Path("outputs/paper_result_pack/robustness"))
    parser.add_argument("--core_ablation_dir", type=Path, default=Path("outputs/paper_result_pack/core_ablation"))
    parser.add_argument("--stat_csv", type=Path, default=Path("outputs/paper_result_pack/stat_tests/paired_tests_online_lyra_vs_baselines.csv"))
    parser.add_argument("--gpu_resource_csv", type=Path, default=Path("outputs/server_runs_20260716/resource_benchmark_20260717_1610/resource_efficiency_table.csv"))
    parser.add_argument("--cpu_resource_csv", type=Path, default=Path("outputs/server_runs_20260716/resource_benchmark_cpu_after_core_20260717_184553/resource_cpu_table.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    horizon = pd.read_csv(args.tables_dir / "main_average_by_horizon.csv")
    overall = pd.read_csv(args.tables_dir / "main_overall_average.csv")
    rank = pd.read_csv(args.tables_dir / "main_nrmse_rank_summary.csv")
    gpu = pd.read_csv(args.gpu_resource_csv)
    cpu = pd.read_csv(args.cpu_resource_csv)
    core = pd.read_csv(args.core_ablation_dir / "core_ablation_summary.csv")
    cold = pd.read_csv(args.core_ablation_dir / "cold_start_summary.csv")
    robust_horizon = pd.read_csv(args.robustness_dir / "robustness_by_horizon_all_models.csv")
    robust_site = pd.read_csv(args.robustness_dir / "robustness_by_site_all_models.csv")
    robust_capacity = pd.read_csv(args.robustness_dir / "robustness_by_capacity_all_models.csv")
    prediction_proxy = pd.read_csv(args.robustness_dir / "robustness_prediction_group_metrics.csv")
    stats = pd.read_csv(args.stat_csv)

    tables = {
        "table_01_main_tsf_by_horizon.tex": main_tsf_table(horizon),
        "table_02_overall_rank.tex": overall_table(overall, rank),
        "table_03_deployment_efficiency.tex": resource_table(gpu, cpu),
        "table_04_core_ablation.tex": simple_metric_table(core, "Core component ablation on representative sites and horizons.", "tab:core_ablation", "ablation"),
        "table_05_cold_start.tex": simple_metric_table(cold, "Cold-start sensitivity of Online Lyra.", "tab:cold_start", "ablation"),
        "table_06_robustness_by_horizon.tex": robustness_compact(robust_horizon, "horizon", "Robustness by forecasting horizon.", "tab:robustness_horizon"),
        "table_07_robustness_by_site.tex": robustness_compact(robust_site, "site", "Robustness across PV sites.", "tab:robustness_site"),
        "table_08_robustness_by_capacity.tex": robustness_compact(robust_capacity, "capacity_group", "Robustness across PV capacity groups.", "tab:robustness_capacity"),
        "table_09_prediction_proxy.tex": prediction_proxy_table(prediction_proxy),
        "table_10_stat_tests.tex": stat_table(stats),
        "table_11_figure_plan.tex": figure_plan_table(),
    }

    written = [write_table(args.output_dir, name, content) for name, content in tables.items()]
    all_tables = "\n\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[a4paper,margin=0.65in]{geometry}",
            r"\usepackage{booktabs,array,caption,pdflscape,graphicx}",
            r"\usepackage[normalem]{ulem}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{textcomp}",
            r"\setlength{\parindent}{0pt}",
            r"\begin{document}",
            r"\section*{Online Lyra PV Results Tables}",
            r"Protocol: 8 PV sites, horizons $H \in \{4,12,24,48,96\}$, lookback $L=96$, and three seeds. Lower is better unless noted.",
            *tables.values(),
            r"\end{document}",
        ]
    )
    write_table(args.output_dir, "all_tables.tex", all_tables)
    manifest = "\n".join(str(path.name) for path in written) + "\nall_tables.tex\n"
    write_table(args.output_dir, "MANIFEST.txt", manifest)
    print(f"Saved {len(tables)} LaTeX tables to {args.output_dir}")


if __name__ == "__main__":
    main()
