#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
RUN_ROOT=${RUN_ROOT:-"outputs/tuning_online_lyra_ablation_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${DEVICE:-"cuda"}
SEED=${SEED:-"2026"}
SITE_IDS=${SITE_IDS:-"1"}
HORIZONS=${HORIZONS:-"4 24 96"}
OFFLINE_DAYS=${OFFLINE_DAYS:-"91"}
YEAR_DAYS=${YEAR_DAYS:-"365"}
TARGET_COL=${TARGET_COL:-"Power (MW)"}
BASELINE_ROOT=${BASELINE_ROOT:-""}

# name|lookback|hidden|lr|offline_epochs|online_steps|replay_capacity|gate_window|gate_min_history|ridge|norm|dropout|offline_batch|replay_batch|loss_mode|use_instance_mean
CONFIGS=(
  "b00_c06_anchor|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|residual|0"
  "b01_plus_instance_mean|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|residual|1"
  "b02_plus_revin|96|64|0.0001|10|1|3000|48|16|0.001|revin|0.10|1|16|residual|0"
  "b03_plus_dropout030|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.30|1|16|residual|0"
  "b04_plus_lookback192|192|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|residual|0"
  "b05_plus_ridge1|96|64|0.0001|10|1|3000|48|16|1.0|mean|0.10|1|16|residual|0"
  "b06_forecast_loss_fixed_scale|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|forecast|0"
  "b07_revin_forecast_fixed_scale|96|64|0.0001|10|1|3000|48|16|0.001|revin|0.10|1|16|forecast|0"
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
  printf 'CONFIGS=%s\n' "${CONFIGS[@]}"
} | tee "$RUN_ROOT/tuning_manifest.txt"

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg lookback hidden_dim learning_rate offline_epochs online_steps replay_capacity gate_window gate_min_history ridge_lambda normalization residual_dropout offline_batch_size replay_batch loss_mode use_instance_mean <<< "$config"
  echo "== Online Lyra ablation config ${cfg} =="
  for site_id in $SITE_IDS; do
    site_file=$(resolve_site_file "$site_id")
    out_dir="$RUN_ROOT/$cfg/site_${site_id}"
    log_path="$RUN_ROOT/logs/${cfg}_site_${site_id}.log"
    instance_flag="--no_instance_mean"
    if [[ "$use_instance_mean" == "1" ]]; then
      instance_flag="--use_instance_mean"
    fi
    mkdir -p "$out_dir"
    echo "-- site=${site_id} file=${site_file} -> ${out_dir}"
    python scripts/run_online_pv.py \
      --file "$site_file" \
      --target_col "$TARGET_COL" \
      --horizons $HORIZONS \
      --lookback "$lookback" \
      --offline_days "$OFFLINE_DAYS" \
      --year_days "$YEAR_DAYS" \
      --offline_stride 1 \
      --offline_epochs "$offline_epochs" \
      --offline_batch_size "$offline_batch_size" \
      --online_steps "$online_steps" \
      --replay_batch "$replay_batch" \
      --hidden_dim "$hidden_dim" \
      --learning_rate "$learning_rate" \
      --replay_capacity "$replay_capacity" \
      --ridge_lambda "$ridge_lambda" \
      --normalization "$normalization" \
      --loss_mode "$loss_mode" \
      "$instance_flag" \
      --residual_dropout "$residual_dropout" \
      --gate_window "$gate_window" \
      --gate_min_history "$gate_min_history" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --output_dir "$out_dir" \
      2>&1 | tee "$log_path"
  done
  if [[ -n "$BASELINE_ROOT" ]]; then
    python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" --baseline_root "$BASELINE_ROOT" || true
  else
    python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" || true
  fi
done

if [[ -n "$BASELINE_ROOT" ]]; then
  python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT" --baseline_root "$BASELINE_ROOT"
else
  python scripts/summarize_online_lyra_tuning.py "$RUN_ROOT"
fi

echo "Online Lyra ablation complete: $RUN_ROOT"
