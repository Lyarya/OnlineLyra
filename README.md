# Online Lyra

**Causal Block-Prequential Evaluation & Streaming Forecasting Suite for Solar Power Generation**

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)
![Protocol](https://img.shields.io/badge/evaluation-block--prequential-orange.svg)

**Online Lyra** is a lightweight, modular toolkit for evaluating short-horizon photovoltaic (PV) power forecasts under streaming deployment constraints.

Two methodological issues deserve particular attention in time-series evaluation:

1. **Temporal target leakage:** shared targets, full-series normalization or updates performed before a forecast is issued can expose future information.
2. **Regime and tail blindness:** all-day averages can obscure daytime errors when extended nighttime periods contain zero generation.

This repository provides a block-prequential execution harness, standardized forecaster interfaces, physical-bound clipping and regime-aware evaluation utilities. It includes three executable classical baselines and a generated-data benchmark. It is a general evaluation toolkit, not a release of a separate learned forecasting model or a reproduction of measured station results.

## Key Features

- **Forecast–Observe–Evaluate–Update ordering:** nonoverlapping H-step blocks are scored before the model receives their targets through `update`.
- **Runtime boundary checks:** lookback timestamps precede the decision boundary, target timestamps follow it, and the time grid is ordered and regular.
- **Regime-aware and tail diagnostics:** overall and active-daytime (`target > 0.01 * capacity`) MSE, MAE and RMSE, plus P95/P99 absolute errors expressed as percent of plant capacity.
- **Initialization-only normalization:** NMSE uses population variance from the initialization prefix; the deployment suffix never determines this scale.
- **Zero-dependency portability:** runtime code uses Python 3.9+ and the standard library, with annotated interfaces and unit tests.

Boundary checks enforce the harness's information flow. They cannot detect future information accessed independently inside an externally supplied model. The CSV auditor checks logged values and timestamps, not the process that generated predictions.

## Repository Architecture

```text
OnlineLyra/
├── models/
│   ├── base.py                       # StreamingForecaster ABC and clipping
│   └── baselines.py                  # Persistence, seasonal persistence, mean
├── exp/
│   └── prequential_runner.py         # Block rollout and timestamp validation
├── utils/
│   └── metrics.py                    # Physical-unit, normalized and tail scores
├── scripts/
│   └── evaluate_csv.py               # Prediction-log validator and scorer
├── examples/
│   └── run_streaming_benchmark.py     # Generated PV stream and baseline benchmark
└── tests/
    └── test_causality_and_metrics.py  # Boundaries, ordering, scores and CSV IO
```

## Quickstart

Run commands from the repository root. No package installation is required.

### 1. Run the streaming benchmark

```bash
python3 -m examples.run_streaming_benchmark
```

The generated 15-minute stream combines a daily half-sine profile, varying daily amplitude, persistent stochastic cloud attenuation and exactly zero nighttime output. Three independent baseline instances run at each horizon H∈{4,12,24}. The default example uses 14 generated days, a three-day initialization prefix, a two-day lookback and a capacity of 50.

Console results describe this simulation only. They are not measurements from operational plants.

### 2. Audit an external prediction log

```bash
# Select a new output path; existing files are never overwritten.
python3 -m examples.run_streaming_benchmark --export-csv /tmp/pv_predictions.csv
python3 scripts/evaluate_csv.py /tmp/pv_predictions.csv --interval-minutes 15 --capacity 50.0
```

The export contains **one model–horizon condition**, defaulting to `LastValuePersistence`, H=4. Use `--export-model DiurnalSmartPersistence --export-horizon 12` to select another condition.

Input CSV columns:

```csv
timestamp,target,prediction
2020-01-04T00:00:00+00:00,0.0,0.0
2020-01-04T00:15:00+00:00,0.0,0.0
```

Logs must contain at least two records, finite target/prediction values, unique column names and a strictly increasing regular time grid. ISO-8601 timestamps may be consistently timezone-aware or consistently naive; UTC is recommended. The interval is inferred if `--interval-minutes` is omitted. Invalid rows are rejected without interpolation, time compression or automatic clipping.

Optionally pass `--initialization-variance` using a population variance calculated from a separate pre-deployment prefix. The CLI never estimates it from the scored deployment log.

### 3. Run unit tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover future-contaminated lookbacks, overlapping boundaries, irregular grids, prediction/scoring/update order, prefix independence from future values, initialization-only variance, partial target blocks, invalid outputs, baseline phase alignment, known numerical scores and CSV round trips.

## Implementing a Custom Streaming Forecaster

```python
from typing import List, Sequence
from models.base import StreamingForecaster


class CustomOnlineForecaster(StreamingForecaster):
    def predict(
        self, lookback_window: Sequence[float], horizon: int
    ) -> List[float]:
        raw_forecast = [lookback_window[-1]] * horizon
        return self.clip_physical_bounds(raw_forecast)

    def update(
        self,
        lookback_window: Sequence[float],
        observed_target: Sequence[float],
    ) -> None:
        # This persistence example has no trainable state.
        return None
```

Create an instance with `CustomOnlineForecaster(capacity=50.0)`, then pass it to `BlockPrequentialEvaluator(model, lookback_steps=192, horizon=4)`. Call `run(timestamps, values, initialization_steps=288)` to obtain per-target records, block scores and overall/daytime summaries. Initialize any learned parameters separately using only pre-deployment observations. Use a fresh instance for each independent condition.

## Evaluation Conventions

- **Decision boundary:** the forecast origin lies halfway between the final observed sample and the first target sample. All history timestamps are strictly earlier and all target timestamps strictly later. This is a discrete-stream convention, not a model of telemetry delay.
- **Completed blocks:** the runner updates once per fully observed block. It records the number of trailing samples that do not complete a block; those samples are not scored or used for updates.
- **Physical bounds:** runner observations must lie in `[0, capacity]`; finite forecasts are clipped to that range before scoring. The standalone CSV auditor scores supplied predictions unchanged.
- **Baseline definitions:** last-value persistence repeats the latest value; `DiurnalSmartPersistenceForecaster` repeats the previous daily cycle with matching phase; rolling mean repeats a trailing simple mean. The diurnal baseline is seasonal persistence, without irradiance normalization. These definitions do not imply equivalence to similarly named methods in other studies.
- **Metric units:** MSE is in squared input units; MAE/RMSE use input units. NMSE divides MSE by initialization population variance. P95/P99 use linear interpolation at `(n - 1) * p` and report `100 * absolute_error / capacity`.
- **Empty/degenerate subsets:** no daytime samples produce `N/A` subset scores. A constant initialization prefix produces undefined (`None`) NMSE, not an arbitrary substitute scale.

## Data and Scope

The example generates its inputs locally. No real station data, prediction logs from measured plants or trained model artifacts are included. The toolkit grants no access to restricted datasets. A successful test run validates the behaviors covered by those tests; it does not independently establish the integrity of an unrelated experiment.

## Roadmap

- [x] Block-prequential evaluation with runtime timestamp checks
- [x] Standard-library CSV validation and daytime/tail metrics
- [x] Three executable classical reference baselines
- [x] Generated-stream benchmark and causal-ordering unit tests
- [ ] Additional generic benchmark adapters and evaluation diagnostics
