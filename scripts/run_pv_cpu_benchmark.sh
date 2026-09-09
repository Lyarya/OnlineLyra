#!/usr/bin/env bash
set -euo pipefail

# CPU latency benchmark for edge-deployment reporting.
# Uses the same site/horizon/seed as the GPU resource benchmark.

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"$PROJECT_ROOT/solar_stations"}
BASELINE_ROOT=${BASELINE_ROOT:-"outputs/fair_protocol_20260716_123158"}
ONLINE_ROOT=${ONLINE_ROOT:-"outputs/final_online_lyra_b06_20260716_231353"}
RUN_ROOT=${RUN_ROOT:-"outputs/resource_benchmark_cpu_$(date +%Y%m%d_%H%M%S)"}
SITE=${SITE:-"1"}
SEED=${SEED:-"2028"}
HORIZON=${HORIZON:-"96"}
LOOKBACK=${LOOKBACK:-"96"}
WARMUP=${WARMUP:-"50"}
REPEATS=${REPEATS:-"500"}
MODELS=${MODELS:-"online_lyra_b06_final official_dlinear official_nlinear official_linear itransformer official_patchtst timekan phaseformer mixlinear olivia_scratch seasonal_naive persistence"}

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
mkdir -p "$RUN_ROOT"
SITE_FILE=$(resolve_site_file "$SITE")
OUTPUT_CSV="$RUN_ROOT/resource_cpu_site${SITE}_h${HORIZON}_seed${SEED}.csv"

{
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "DATA_DIR=$DATA_DIR"
  echo "BASELINE_ROOT=$BASELINE_ROOT"
  echo "ONLINE_ROOT=$ONLINE_ROOT"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "SITE=$SITE"
  echo "SEED=$SEED"
  echo "HORIZON=$HORIZON"
  echo "LOOKBACK=$LOOKBACK"
  echo "WARMUP=$WARMUP"
  echo "REPEATS=$REPEATS"
  echo "MODELS=$MODELS"
} | tee "$RUN_ROOT/cpu_benchmark_manifest.txt"

for model in $MODELS; do
  echo "== CPU benchmark: $model =="
  python scripts/benchmark_pv_resources.py \
    --model "$model" \
    --file "$SITE_FILE" \
    --baseline_root "$BASELINE_ROOT" \
    --online_root "$ONLINE_ROOT" \
    --output_csv "$OUTPUT_CSV" \
    --site "$SITE" \
    --seed "$SEED" \
    --horizon "$HORIZON" \
    --lookback "$LOOKBACK" \
    --batch_size 1 \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --device cpu
done

python - <<'PY' "$OUTPUT_CSV" "$RUN_ROOT/resource_cpu_table.csv"
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
df["params_k"] = df["params"] / 1000.0
df["checkpoint_size_mb"] = df["checkpoint_size_kb"] / 1024.0
cols = [
    "model", "device", "site", "seed", "horizon", "params", "params_k",
    "checkpoint_size_kb", "checkpoint_size_mb", "latency_mean_ms",
    "latency_std_ms", "warmup", "repeats", "checkpoint_path",
]
df[cols].to_csv(dst, index=False)
print(df[cols].to_string(index=False))
PY

echo "$RUN_ROOT" > outputs/latest_resource_benchmark_cpu.txt
echo "CPU benchmark complete: $RUN_ROOT"
