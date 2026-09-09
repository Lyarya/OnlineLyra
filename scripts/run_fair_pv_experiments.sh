#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
RUN_ROOT=${RUN_ROOT:-"outputs/fair_protocol_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${DEVICE:-"cuda"}

BASELINE_EPOCHS=${BASELINE_EPOCHS:-20}
BASELINE_PATIENCE=${BASELINE_PATIENCE:-5}
BASELINE_SEEDS=${BASELINE_SEEDS:-"2026 2027 2028"}
LYRA_OFFLINE_EPOCHS=${LYRA_OFFLINE_EPOCHS:-20}
LYRA_ONLINE_STEPS=${LYRA_ONLINE_STEPS:-1}
LYRA_SEEDS=${LYRA_SEEDS:-"2026 2027 2028"}

BASELINE_MODELS=${BASELINE_MODELS:-"persistence seasonal_naive official_linear official_nlinear official_dlinear official_patchtst itransformer timekan phaseformer mixlinear olivia_scratch"}

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/logs"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "DEVICE=$DEVICE"
echo "BASELINE_MODELS=$BASELINE_MODELS"
echo "BASELINE_SEEDS=$BASELINE_SEEDS"
echo "LYRA_SEEDS=$LYRA_SEEDS"

for seed in $BASELINE_SEEDS; do
  out_dir="$RUN_ROOT/static_official_seed_${seed}"
  echo "== Static official baselines seed=$seed -> $out_dir =="
  python scripts/run_pv_baselines.py \
    --data_dir "$DATA_DIR" \
    --horizons 4 12 24 48 96 \
    --lookback 96 \
    --offline_days 91 \
    --year_days 365 \
    --train_stride 1 \
    --models $BASELINE_MODELS \
    --epochs "$BASELINE_EPOCHS" \
    --patience "$BASELINE_PATIENCE" \
    --batch_size 256 \
    --eval_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --device "$DEVICE" \
    --seed "$seed" \
    --save_artifacts \
    --output_dir "$out_dir" \
    2>&1 | tee "$RUN_ROOT/logs/static_official_seed_${seed}.log"
done

for seed in $LYRA_SEEDS; do
  out_dir="$RUN_ROOT/online_lyra_seed_${seed}"
  echo "== Fair Online Lyra seed=$seed -> $out_dir =="
  python scripts/run_online_pv.py \
    --data_dir "$DATA_DIR" \
    --horizons 4 12 24 48 96 \
    --lookback 96 \
    --offline_days 91 \
    --year_days 365 \
    --offline_stride 1 \
    --offline_epochs "$LYRA_OFFLINE_EPOCHS" \
    --online_steps "$LYRA_ONLINE_STEPS" \
    --device "$DEVICE" \
    --seed "$seed" \
    --save_artifacts \
    --output_dir "$out_dir" \
    2>&1 | tee "$RUN_ROOT/logs/online_lyra_seed_${seed}.log"
done

echo "All fair-protocol jobs finished: $RUN_ROOT"
