#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
RUN_ROOT=${RUN_ROOT:-"outputs/tuning_online_lyra_stage2_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${DEVICE:-"cuda"}
SEED=${SEED:-"2026"}
SITE_IDS=${SITE_IDS:-"1"}
HORIZONS=${HORIZONS:-"4 24 96"}
OFFLINE_DAYS=${OFFLINE_DAYS:-"91"}
YEAR_DAYS=${YEAR_DAYS:-"365"}
TARGET_COL=${TARGET_COL:-"Power (MW)"}
BASELINE_ROOT=${BASELINE_ROOT:-""}

# name|lookback|hidden|lr|offline_epochs|online_steps|replay_capacity|gate_window|gate_min_history|ridge|norm|dropout|offline_batch|replay_batch
CONFIGS=(
  "a01_lb336_dm256_dp030_ridge1|336|256|0.0003|10|1|5000|48|16|1.0|revin|0.30|64|64"
  "a02_lb336_dm512_dp045_ridge5|336|512|0.0003|10|1|5000|48|16|5.0|revin|0.45|64|64"
  "a03_lb336_dm512_dp050_ridge5_lr1e-4|336|512|0.0001|15|1|5000|48|16|5.0|revin|0.50|64|64"
  "a04_lb720_dm256_dp030_ridge1|720|256|0.0003|8|1|5000|48|16|1.0|revin|0.30|64|64"
  "a05_lb720_dm512_dp045_ridge5|720|512|0.0003|8|1|5000|48|16|5.0|revin|0.45|64|64"
  "a06_lb720_dm512_dp050_ridge5_lr1e-4|720|512|0.0001|12|1|5000|48|16|5.0|revin|0.50|64|64"
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
  IFS='|' read -r cfg lookback hidden_dim learning_rate offline_epochs online_steps replay_capacity gate_window gate_min_history ridge_lambda normalization residual_dropout offline_batch_size replay_batch <<< "$config"
  echo "== Online Lyra stage2 config ${cfg} =="
  for site_id in $SITE_IDS; do
    site_file=$(resolve_site_file "$site_id")
    out_dir="$RUN_ROOT/$cfg/site_${site_id}"
    log_path="$RUN_ROOT/logs/${cfg}_site_${site_id}.log"
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

echo "Online Lyra stage2 tuning complete: $RUN_ROOT"
