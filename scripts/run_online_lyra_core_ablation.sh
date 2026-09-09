#!/usr/bin/env bash
set -euo pipefail

# Compact Online Lyra ablations. This script retrains the spectral-scaling and
# cold-start variants; comparison rows for the full model, gate, residual
# learner, and offline model are read from BASELINE_ROOT.

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
RUN_ROOT=${RUN_ROOT:-"outputs/core_ablation_online_lyra_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${DEVICE:-"cuda"}
SEED=${SEED:-"2028"}
SITE_IDS=${SITE_IDS:-"1 2 5 8"}
HORIZONS=${HORIZONS:-"4 24 96"}
YEAR_DAYS=${YEAR_DAYS:-"365"}
TARGET_COL=${TARGET_COL:-"Power (MW)"}
SAVE_ARTIFACTS=${SAVE_ARTIFACTS:-"1"}
BASELINE_ROOT=${BASELINE_ROOT:-"outputs/reference_results"}

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

# name|offline_days|residual_scale_mode|loss_mode|use_instance_mean
RUN_CONFIGS=(
  "ab_no_spectral_residual_scale|91|none|forecast|0"
  "cold30_final|30|spectral|forecast|0"
  "cold60_final|60|spectral|forecast|0"
)

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
  echo "SEED=$SEED"
  echo "SITE_IDS=$SITE_IDS"
  echo "HORIZONS=$HORIZONS"
  echo "BASELINE_ROOT=$BASELINE_ROOT"
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
  echo "SAVE_ARTIFACTS=$SAVE_ARTIFACTS"
  printf 'RUN_CONFIGS=%s\n' "${RUN_CONFIGS[*]}"
} | tee "$RUN_ROOT/core_ablation_manifest.txt"

for config in "${RUN_CONFIGS[@]}"; do
  IFS='|' read -r cfg offline_days residual_scale_mode loss_mode use_instance_mean <<< "$config"
  echo "== Core ablation: ${cfg} =="
  for site_id in $SITE_IDS; do
    site_file=$(resolve_site_file "$site_id")
    out_dir="$RUN_ROOT/$cfg/site_${site_id}_seed_${SEED}"
    log_path="$RUN_ROOT/logs/${cfg}_site_${site_id}_seed_${SEED}.log"
    instance_flag="--no_instance_mean"
    if [[ "$use_instance_mean" == "1" ]]; then
      instance_flag="--use_instance_mean"
    fi
    artifact_args=()
    if [[ "$SAVE_ARTIFACTS" == "1" ]]; then
      artifact_args=(--save_artifacts --artifact_dir "$out_dir/artifacts")
    fi
    mkdir -p "$out_dir"
    echo "-- cfg=${cfg} site=${site_id} file=${site_file} -> ${out_dir}"
    python scripts/run_online_pv.py \
      --file "$site_file" \
      --target_col "$TARGET_COL" \
      --horizons $HORIZONS \
      --lookback "$LOOKBACK" \
      --offline_days "$offline_days" \
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
      --loss_mode "$loss_mode" \
      --residual_scale_mode "$residual_scale_mode" \
      "$instance_flag" \
      --residual_dropout "$RESIDUAL_DROPOUT" \
      --gate_window "$GATE_WINDOW" \
      --gate_min_history "$GATE_MIN_HISTORY" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --output_dir "$out_dir" \
      "${artifact_args[@]}" \
      2>&1 | tee "$log_path"
  done
  python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" --baseline_root "$BASELINE_ROOT" || true
done

python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" --baseline_root "$BASELINE_ROOT"
echo "$RUN_ROOT" > outputs/latest_core_ablation_online_lyra.txt
echo "Core Online Lyra ablation complete: $RUN_ROOT"
