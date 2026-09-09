# Script guide

The scripts are grouped below by their role in the reported study. Paths are
relative to the repository root.

## Primary experiment protocols

| Script | Role |
| --- | --- |
| `run_online_lyra_b06_final.sh` | Reported Online Lyra configuration: eight sites, five horizons, and three seeds |
| `run_pv_baselines.py` | Fixed-recipe external, linear, and persistence baselines |
| `run_offline_lyra_pv_final.sh` | Matched Offline Lyra evaluation |
| `run_online_lyra_core_ablation.sh` | Component ablation and cold-start sensitivity |
| `benchmark_pv_resources.py` | Parameter count, model size, latency, and CUDA-memory measurements |
| `run_pv_stat_tests.py` | Paired statistical tests and false-discovery-rate correction |

The Lyra experiment entry points require the PV workbooks and the analytical
component identified in the top-level README. Baseline runners also require
the corresponding upstream model sources.

## Tables and figures

| Script | Role |
| --- | --- |
| `build_pv_main_tables.py` | Aggregate and horizon-level performance tables |
| `build_pv_robustness_tables.py` | Site- and capacity-level robustness tables |
| `build_pv_core_ablation_table.py` | Component-ablation and cold-start tables |
| `build_pv_latex_table_pack.py` | LaTeX table exports |
| `analyze_pv_decomposition_characteristics.py` | Analytical and residual component statistics |
| `make_online_lyra_ieee_figures.py` | Canonical manuscript-figure pipeline |
| `make_pv_paper_figures_final.py` | Final plotting pipeline used by the canonical renderer |
| `make_fig1.py` | Standalone accuracy-panel renderer |
| `make_fig3.py` | Standalone event case-study renderer |
| `make_pv_signal_characteristics_real.py` | Measured-data motivation figure |
| `make_online_pv_figures.py` | Per-run diagnostic figures |

`make_pv_paper_figures_revised.py` and
`make_pv_paper_figures_revised_v2.py` remain implementation dependencies of
the canonical renderer. They are not alternative manuscript entry points.

## Supporting protocols and audits

The remaining scripts cover tuning, supplementary studies, time-integrity and
metric-sensitivity audits, result aggregation, and figure-data preparation.
Most consume generated CSV or NPZ artifacts under `outputs/`; those artifacts
are intentionally excluded from this source distribution. Dated default paths
record the artifact layout used for the reported run and may be overridden by
the corresponding command-line options.
