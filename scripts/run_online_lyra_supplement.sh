#!/usr/bin/env bash
set -euo pipefail

# Supplementary Online Lyra experiments:
# 1) component ablation on all 8 PV sites with one seed;
# 2) cold-start sensitivity with shorter offline history.
#
# By default this stores summaries and driver logs only. Set SAVE_ARTIFACTS=1
# if prediction files, update logs, and checkpoints are needed.

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
RUN_ROOT=${RUN_ROOT:-"outputs/supplement_online_lyra_$(date +%Y%m%d_%H%M%S)"}
DEVICE=${DEVICE:-"cuda"}
SEED=${SEED:-"2028"}
SITE_IDS=${SITE_IDS:-"1 2 3 4 5 6 7 8"}
HORIZONS=${HORIZONS:-"4 12 24 48 96"}
YEAR_DAYS=${YEAR_DAYS:-"365"}
TARGET_COL=${TARGET_COL:-"Power (MW)"}
BASELINE_ROOT=${BASELINE_ROOT:-""}
SAVE_ARTIFACTS=${SAVE_ARTIFACTS:-"0"}

# name|lookback|hidden|lr|offline_epochs|online_steps|replay_capacity|gate_window|gate_min_history|ridge|norm|dropout|offline_batch|replay_batch|loss_mode|use_instance_mean|offline_days
COMPONENT_CONFIGS=(
  "ab_anchor_residual_loss|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|residual|0|91"
  "ab_final_forecast_loss|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|forecast|0|91"
  "ab_revin_forecast_loss|96|64|0.0001|10|1|3000|48|16|0.001|revin|0.10|1|16|forecast|0|91"
  "ab_instance_mean_residual|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|residual|1|91"
)

COLD_START_CONFIGS=(
  "cold30_final|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|forecast|0|30"
  "cold60_final|96|64|0.0001|10|1|3000|48|16|0.001|mean|0.10|1|16|forecast|0|60"
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

run_study() {
  local study_name="$1"
  shift
  local configs=("$@")
  local study_root="$RUN_ROOT/$study_name"
  mkdir -p "$study_root/logs"
  printf '%s\n' "${configs[@]}" > "$study_root/configs.txt"

  for config in "${configs[@]}"; do
    IFS='|' read -r cfg lookback hidden_dim learning_rate offline_epochs online_steps replay_capacity gate_window gate_min_history ridge_lambda normalization residual_dropout offline_batch_size replay_batch loss_mode use_instance_mean offline_days <<< "$config"
    echo "== Supplement ${study_name}: ${cfg} =="
    for site_id in $SITE_IDS; do
      site_file=$(resolve_site_file "$site_id")
      out_dir="$study_root/$cfg/site_${site_id}"
      log_path="$study_root/logs/${cfg}_site_${site_id}.log"
      instance_flag="--no_instance_mean"
      if [[ "$use_instance_mean" == "1" ]]; then
        instance_flag="--use_instance_mean"
      fi
      artifact_args=()
      if [[ "$SAVE_ARTIFACTS" == "1" ]]; then
        artifact_args=(--save_artifacts --artifact_dir "$out_dir/artifacts")
      fi
      mkdir -p "$out_dir"
      echo "-- site=${site_id} file=${site_file} -> ${out_dir}"
      python scripts/run_online_pv.py \
        --file "$site_file" \
        --target_col "$TARGET_COL" \
        --horizons $HORIZONS \
        --lookback "$lookback" \
        --offline_days "$offline_days" \
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
        "${artifact_args[@]}" \
        2>&1 | tee "$log_path"
    done
    if [[ -n "$BASELINE_ROOT" ]]; then
      python scripts/summarize_online_lyra_tuning.py "$study_root" --baseline_root "$BASELINE_ROOT" || true
    else
      python scripts/summarize_online_lyra_tuning.py "$study_root" || true
    fi
  done

  if [[ -n "$BASELINE_ROOT" ]]; then
    python scripts/summarize_online_lyra_tuning.py "$study_root" --baseline_root "$BASELINE_ROOT"
  else
    python scripts/summarize_online_lyra_tuning.py "$study_root"
  fi
}

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"
{
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "DATA_DIR=$DATA_DIR"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "DEVICE=$DEVICE"
  echo "SEED=$SEED"
  echo "SITE_IDS=$SITE_IDS"
  echo "HORIZONS=$HORIZONS"
  echo "SAVE_ARTIFACTS=$SAVE_ARTIFACTS"
  echo "BASELINE_ROOT=$BASELINE_ROOT"
} | tee "$RUN_ROOT/supplement_manifest.txt"

run_study "component_ablation_seed_${SEED}" "${COMPONENT_CONFIGS[@]}"
run_study "cold_start_seed_${SEED}" "${COLD_START_CONFIGS[@]}"

echo "$RUN_ROOT" > outputs/latest_supplement_online_lyra.txt
echo "Supplementary Online Lyra experiments complete: $RUN_ROOT"
