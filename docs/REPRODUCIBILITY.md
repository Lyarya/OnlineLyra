# Reproducibility record

## Reported environment

The final controlled experiments were recorded on Ubuntu
22.04.5 LTS with 12 allocated CPU cores and 62 GiB system memory. The host CPU
was an Intel Xeon Platinum 8260 at 2.40 GHz. GPU execution used a virtualized
NVIDIA GeForce RTX 3090 allocation with 48 GiB visible memory, compute
capability 8.6, and driver 580.105.08.

The reported software environment was:

- Python 3.12.3
- PyTorch 2.7.0+cu128
- PyTorch CUDA runtime 12.8
- cuDNN 9.7.1
- NumPy 2.2.6
- pandas 3.0.2
- SciPy 1.17.1
- scikit-learn 1.8.0
- openpyxl 3.1.5

`nvidia-smi` reported CUDA 13.0 as the driver's maximum compatibility level;
the PyTorch binary used CUDA runtime 12.8. The machine-readable capture is in
`docs/server_environment_20260727.json`.

## Final Online Lyra protocol

The exact shell entry point is `scripts/run_online_lyra_b06_final.sh`.

- Sites: 1--8
- Horizons: 4, 12, 24, 48, and 96 samples (1, 3, 6, 12, and 24 h)
- Random seeds: 2026, 2027, and 2028
- Lookback: 96 samples
- Initialization history: 91 days
- Hidden width: 64
- Dropout: 0.10
- Initialization passes: 10
- Online steps after each completed block: 1
- Learning rate: 1e-4
- Replay batch/capacity: 16/3000
- Gate history/warm-up: 48/16 completed blocks
- Hankel rank/ridge: 1/1e-3
- Normalization: mean
- Objective: clipped recomposed forecast MSE

The runner writes the resolved values into `final_manifest.txt` before any
training begins.

## Baseline source and checkpoint rules

External neural baselines use the authors' released implementations. Git
revisions are recorded when upstream metadata was present; otherwise the exact
imported model file is identified by SHA-256 in
`docs/external_baseline_provenance.json`.

The external neural models share the same leakage-free wrapper: AdamW,
learning rate 1e-3, weight decay 1e-4, batch size 256, gradient clipping at
1.0, at most 20 epochs, and patience five. The checkpoint with the lowest
training MSE is restored, and all external models remain frozen during the
chronological deployment period. No deployment target is used for checkpoint
selection.

Offline Lyra uses an 80:20 chronological split inside the initialization
period, at most 30 epochs, patience five, and restores the checkpoint with the
lowest validation loss. Online Lyra uses ten fixed initialization passes and
then performs one causal update only after each forecast block has been fully
observed. Its saved checkpoint is the terminal deployed state, not a state
selected from deployment-period errors.

This is a fixed-recipe system comparison rather than an equal-budget
architecture search. The manuscript states this distinction explicitly when
interpreting small differences among the leading methods.

## Data and generated artifacts

The utility-scale PV workbooks are subject to data-sharing restrictions and
are excluded from Git. The
repository also excludes checkpoints, logs, predictions, and bulk output
directories. Each server run should retain its manifest, environment capture,
source revision, and source-data hash alongside the generated artifacts.

The manuscript source, compiled PDFs, publication figures, and LaTeX temporary
files remain local artifacts under `outputs/` and are excluded from Git. The
code repository retains the deterministic renderers, protocol definitions,
environment capture, and baseline provenance needed to regenerate them when
the authorized data and result bundle are available.
