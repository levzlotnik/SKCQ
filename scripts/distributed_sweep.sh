#!/usr/bin/env bash
# Distributed sweep: baseline + distributed build (multi-machine) + quantized eval (3090).
# Usage: bash scripts/distributed_sweep.sh <config.yaml> [workers.yaml]
set -euo pipefail

CONFIG="${1:?usage: $0 <config.yaml> [workers.yaml]}"
WORKERS_YAML="${2:-workers.yaml}"

LABEL=$(basename "$CONFIG" .yaml)
LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${LABEL}.log"

# Extract output_dir from config
OUT_DIR=$(grep "^output_dir:" "$CONFIG" | awk -F'"' '{print $2}')
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="codebooks_${LABEL}"
fi

echo "Distributed sweep: ${LABEL}"
echo "  config:   $CONFIG"
echo "  workers:  $WORKERS_YAML"
echo "  output:   $OUT_DIR"
echo

# Phase 1: baseline + distributed build
echo "  -> Phase 1: baseline + distributed build..."
rocm/.venv/bin/python distributed_run.py --config "$CONFIG" \
    --workers "$WORKERS_YAML" --baseline-cache baseline.pt \
    2>&1 | tee "$LOG_FILE"

# Phase 2: quantized eval on 3090
echo "  -> Phase 2: quantized eval on 3090..."
cuda/.venv/bin/python eval_quantized.py --config "$CONFIG" \
    --codebook-dir "$OUT_DIR" --eval-samples 100 --kld-tokens 2048 \
    --output "${OUT_DIR}/compare_cuda.json" \
    2>&1 | tee -a "$LOG_FILE"

echo "  -> done (see ${OUT_DIR}/compare_cuda.json and ${LOG_FILE})"
