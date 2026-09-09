#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
RUN_ROOT=${RUN_ROOT:-"outputs/final_online_lyra_b06_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${DEVICE:-"cuda"}
SEEDS=${SEEDS:-"2026 2027 2028"}
SITE_IDS=${SITE_IDS:-"1 2 3 4 5 6 7 8"}
HORIZONS=${HORIZONS:-"4 12 24 48 96"}
OFFLINE_DAYS=${OFFLINE_DAYS:-"91"}
YEAR_DAYS=${YEAR_DAYS:-"365"}
TARGET_COL=${TARGET_COL:-"Power (MW)"}
BASELINE_ROOT=${BASELINE_ROOT:-""}

CONFIG_NAME=${CONFIG_NAME:-"b06_final"}
LOOKBACK=${LOOKBACK:-"96"}
HIDDEN_DIM=${HIDDEN_DIM:-"64"}
LEARNING_RATE=${LEARNING_RATE:-"0.0001"}
OFFLINE_EPOCHS=${OFFLINE_EPOCHS:-"10"}
ONLINE_STEPS=${ONLINE_STEPS:-"1"}
REPLAY_CAPACITY=${REPLAY_CAPACITY:-"3000"}
REPLAY_BATCH=${REPLAY_BATCH:-"16"}
RIDGE_LAMBDA=${RIDGE_LAMBDA:-"0.001"}
NORMALIZATION=${NORMALIZATION:-"mean"}
RESIDUAL_DROPOUT=${RESIDUAL_DROPOUT:-"0.10"}
GATE_WINDOW=${GATE_WINDOW:-"48"}
GATE_MIN_HISTORY=${GATE_MIN_HISTORY:-"16"}
OFFLINE_BATCH_SIZE=${OFFLINE_BATCH_SIZE:-"1"}

resolve_site_file() {
  local site_id="$1"
  local match
  match=$(find "$DATA_DIR" -maxdepth 1 -type f -name "Solar station site ${site_id} (*" | sort | head -n 1)
  if [[ -z "$match" ]]; then
    echo "Could not find PV file for site ${site_id} in ${DATA_DIR}" >&2
    exit 1
  fi
  echo "$match"
}

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/logs"

{
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "DATA_DIR=$DATA_DIR"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "DEVICE=$DEVICE"
  echo "SEEDS=$SEEDS"
  echo "SITE_IDS=$SITE_IDS"
  echo "HORIZONS=$HORIZONS"
  echo "BASELINE_ROOT=$BASELINE_ROOT"
  echo "CONFIG_NAME=$CONFIG_NAME"
  echo "LOOKBACK=$LOOKBACK"
  echo "HIDDEN_DIM=$HIDDEN_DIM"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "OFFLINE_EPOCHS=$OFFLINE_EPOCHS"
  echo "ONLINE_STEPS=$ONLINE_STEPS"
  echo "REPLAY_CAPACITY=$REPLAY_CAPACITY"
  echo "REPLAY_BATCH=$REPLAY_BATCH"
  echo "RIDGE_LAMBDA=$RIDGE_LAMBDA"
  echo "NORMALIZATION=$NORMALIZATION"
  echo "RESIDUAL_DROPOUT=$RESIDUAL_DROPOUT"
  echo "GATE_WINDOW=$GATE_WINDOW"
  echo "GATE_MIN_HISTORY=$GATE_MIN_HISTORY"
  echo "OFFLINE_BATCH_SIZE=$OFFLINE_BATCH_SIZE"
  echo "LOSS_MODE=forecast"
  echo "INSTANCE_MEAN=no"
  echo "SAVE_ARTIFACTS=true"
} | tee "$RUN_ROOT/final_manifest.txt"

for seed in $SEEDS; do
  for site_id in $SITE_IDS; do
    site_file=$(resolve_site_file "$site_id")
    out_dir="$RUN_ROOT/$CONFIG_NAME/site_${site_id}_seed_${seed}"
    log_path="$RUN_ROOT/logs/${CONFIG_NAME}_site_${site_id}_seed_${seed}.log"
    mkdir -p "$out_dir"
    echo "== Online Lyra final ${CONFIG_NAME}: site=${site_id} seed=${seed} =="
    echo "-- file=${site_file} -> ${out_dir}"
    python scripts/run_online_pv.py \
      --file "$site_file" \
      --target_col "$TARGET_COL" \
      --horizons $HORIZONS \
      --lookback "$LOOKBACK" \
      --offline_days "$OFFLINE_DAYS" \
      --year_days "$YEAR_DAYS" \
      --offline_stride 1 \
      --offline_epochs "$OFFLINE_EPOCHS" \
      --offline_batch_size "$OFFLINE_BATCH_SIZE" \
      --online_steps "$ONLINE_STEPS" \
      --replay_batch "$REPLAY_BATCH" \
      --hidden_dim "$HIDDEN_DIM" \
      --learning_rate "$LEARNING_RATE" \
      --replay_capacity "$REPLAY_CAPACITY" \
      --ridge_lambda "$RIDGE_LAMBDA" \
      --normalization "$NORMALIZATION" \
      --loss_mode forecast \
      --no_instance_mean \
      --residual_dropout "$RESIDUAL_DROPOUT" \
      --gate_window "$GATE_WINDOW" \
      --gate_min_history "$GATE_MIN_HISTORY" \
      --device "$DEVICE" \
      --seed "$seed" \
      --save_artifacts \
      --artifact_dir "$out_dir/artifacts" \
      --output_dir "$out_dir" \
      2>&1 | tee "$log_path"

    if [[ -n "$BASELINE_ROOT" ]]; then
      python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" --baseline_root "$BASELINE_ROOT" || true
    else
      python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" || true
    fi
  done
done

if [[ -n "$BASELINE_ROOT" ]]; then
  python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" --baseline_root "$BASELINE_ROOT"
else
  python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT"
fi

echo "Online Lyra b06 final run complete: $RUN_ROOT"
