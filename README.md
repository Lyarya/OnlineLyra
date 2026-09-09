# Online Lyra

This repository contains the evaluation code, experiment protocols, figure
renderers, and reproducibility records for **Online Lyra**, a causal streaming
framework for short-term utility-scale photovoltaic (PV) forecasting.

Online Lyra combines analytical trend extrapolation with online residual
adaptation. The repository is organized around the fixed configurations used
in the reported study, with separate entry points for evaluation, baselines,
ablations, statistical tests, and figure generation.

## Repository map

| Path | Purpose |
| --- | --- |
| `models/OnlineLyraPV.py` | Online Lyra PV interface and online-learning components |
| `models/Lyra.py` | Original long-term time-series forecasting model |
| `data_provider/` | Time-series dataset loading and preprocessing |
| `utils/` | Metrics, training utilities, and source-scope exceptions |
| `scripts/run_online_lyra_b06_final.sh` | Reported eight-site, five-horizon, three-seed protocol |
| `scripts/run_pv_baselines.py` | External, linear, and persistence baseline runner |
| `scripts/make_online_lyra_ieee_figures.py` | Canonical manuscript-figure pipeline |
| `docs/REPRODUCIBILITY.md` | Environment, protocol, checkpoint, and artifact record |
| `scripts/README.md` | Complete script index and entry-point descriptions |

## Setup

Python 3.12 is recommended. Create an isolated environment and install the
recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Data layout

The PV experiment scripts expect authorized workbooks under
`solar_stations/`, with filenames of the following form:

```text
Solar station site 1 (Nominal capacity-50MW).xlsx
...
Solar station site 8 (Nominal capacity-30MW).xlsx
```

The default prediction target is `Power (MW)`. Dataset paths, output roots,
compute devices, seeds, and other runtime settings can be changed through the
command-line options or environment variables documented by each entry point.

The station workbooks are subject to data-sharing restrictions and are not
included in this repository.

## Quick checks

Run the dataset-free synthetic check:

```bash
python scripts/smoke_online_lyra_pv.py
```

Check the Python and shell entry points without starting an experiment:

```bash
python -m compileall -q data_provider exp layers models scripts utils run.py
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

The synthetic check validates the interfaces and utilities included in this
source distribution; it does not regenerate the reported numerical results.

## Main PV experiment

The reported Online Lyra configuration is encoded in a single shell entry
point:

```bash
DEVICE=cuda RUN_ROOT=outputs/final_online_lyra \
  bash scripts/run_online_lyra_b06_final.sh
```

The protocol uses a 96-sample lookback; forecast horizons of 4, 12, 24, 48,
and 96 samples (1, 3, 6, 12, and 24 h); eight sites; and random seeds 2026,
2027, and 2028. The runner writes the resolved configuration to a manifest
before training begins.

The complete parameter record and chronological evaluation rules are given in
`docs/REPRODUCIBILITY.md`.

## Additional experiments

| Purpose | Entry point |
| --- | --- |
| Offline Lyra comparison | `scripts/run_offline_lyra_pv_final.sh` |
| External and statistical baselines | `scripts/run_pv_baselines.py` |
| Component and cold-start ablations | `scripts/run_online_lyra_core_ablation.sh` |
| Resource measurements | `scripts/benchmark_pv_resources.py` |
| Paired statistical tests | `scripts/run_pv_stat_tests.py` |
| Main and horizon-level tables | `scripts/build_pv_main_tables.py` |

See `scripts/README.md` for the figure-data preparation, tuning,
supplementary, robustness, and audit workflows.

## Tables and figures

After the required experiment summaries and figure-data files are available,
run the canonical figure pipeline:

```bash
python scripts/make_online_lyra_ieee_figures.py
```

The table builders and renderers consume generated CSV or NPZ artifacts under
`outputs/`. Checkpoints, prediction arrays, logs, tables, figures, manuscript
sources, and LaTeX build files are excluded from Git.

## Reproducibility notes

- Forecasts are evaluated chronologically. Online updates occur only after a
  forecast block has been fully observed.
- Deployment targets are not used to select neural-baseline checkpoints.
- The canonical runners record seeds, sites, horizons, hyperparameters, and
  output locations in run manifests.
- External baseline revisions and retained source-file hashes are recorded in
  `docs/external_baseline_provenance.json`.
- The reported software and hardware environment is recorded in
  `docs/server_environment_20260727.json`.

## Source availability

This distribution includes the evaluation protocols, fixed configurations,
data-loading and evaluation utilities, standard residual-learning components,
and result-processing scripts. End-to-end Lyra inference additionally requires
the analytical decomposition and latent-rollout module, which is not part of
this distribution. Calls that require that module raise
`ComponentUnavailableError`.

The restricted PV workbooks and generated experiment artifacts are likewise
not redistributed. With authorized data and the required Lyra module, the
retained entry points define the reported evaluation and rendering workflow.
