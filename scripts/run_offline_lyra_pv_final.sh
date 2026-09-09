#!/usr/bin/env bash
set -euo pipefail

# Offline Lyra control for the aligned PV protocol.
# It uses the original Lyra architecture without online adaptation:
# Q1 train/validation, Q2-Q4 rolling test, lookback/horizons aligned
# with the Online Lyra b06 final experiment and static baselines.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ROOT="${RUN_ROOT:-outputs/offline_lyra_final_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/solar_stations}"
TARGET_COL="${TARGET_COL:-Power (MW)}"
LOOKBACK="${LOOKBACK:-96}"
SEEDS="${SEEDS:-2026 2027 2028}"
HORIZONS="${HORIZONS:-4 12 24 48 96}"

mkdir -p "${RUN_ROOT}/logs"

echo "[offline-lyra] RUN_ROOT=${RUN_ROOT}"
echo "[offline-lyra] DATA_DIR=${DATA_DIR}"
echo "[offline-lyra] LOOKBACK=${LOOKBACK}"
echo "[offline-lyra] HORIZONS=${HORIZONS}"
echo "[offline-lyra] SEEDS=${SEEDS}"

for seed in ${SEEDS}; do
  out_dir="${RUN_ROOT}/offline_lyra_seed_${seed}"
  log_file="${RUN_ROOT}/logs/offline_lyra_seed_${seed}.log"
  echo "[offline-lyra] seed=${seed} out=${out_dir}"
  python scripts/run_offline_lyra_pv.py \
    --data_dir "${DATA_DIR}" \
    --target_col "${TARGET_COL}" \
    --horizons ${HORIZONS} \
    --lookback "${LOOKBACK}" \
    --offline_days 91 \
    --year_days 365 \
    --train_stride 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --eval_batch_size 1024 \
    --decompose_batch_size 1024 \
    --patience 5 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --d_model 256 \
    --dropout 0.45 \
    --n_power_iters 3 \
    --ridge_lambda 5.0 \
    --device cuda \
    --seed "${seed}" \
    --save_artifacts \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_file}"
done

echo "${RUN_ROOT}" > outputs/latest_offline_lyra_final.txt
echo "[offline-lyra] done: ${RUN_ROOT}"
