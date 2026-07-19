#!/usr/bin/env bash
# Focused sweep: tier 1 configs (highest probability of good results)
# Runs baseline+build on ROCm, quantized eval on 3090.
set -euo pipefail

LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"

EVAL_SAMPLES="${1:-100}"
KLD_TOKENS="${2:-2048}"

# Tier 1 config in run order
configs=(
    "kbs16_nb4_cb2"    # asymmetric K (residual_k=256), 4 blocks, 2 codebooks
)

total=${#configs[@]}
echo "Tier 1 sweep: ${total} configs, ${EVAL_SAMPLES} samples, ${KLD_TOKENS} kld tokens"
echo

i=0
for label in "${configs[@]}"; do
    i=$((i + 1))
    cfg="configs/sweep/${label}.yaml"
    log_file="${LOG_DIR}/${label}.log"

    echo "[$i/$total] ${label}"

    # Phase 1: ROCm — baseline + build codebooks
    out_dir=$(grep "^output_dir:" "$cfg" | awk -F'"' '{print $2}')
    if [ -z "$out_dir" ]; then
        out_dir="codebooks_${label}"
    fi

    if [ -d "${out_dir}/layer_0" ]; then
        echo "  -> codebooks exist, skipping build"
    else
        echo "  -> building codebooks into ${out_dir}/"
    fi

    rocm/.venv/bin/python build.py --config "$cfg" --compare \
        --use-cuda-worker --eval-samples "$EVAL_SAMPLES" --kld-tokens "$KLD_TOKENS" \
        2>&1 | tee "$log_file"

    # Phase 2: 3090 — quantized eval
    echo "  -> running quantized eval on 3090..."
    cuda/.venv/bin/python eval_quantized.py --config "$cfg" \
        --codebook-dir "$out_dir" --eval-samples "$EVAL_SAMPLES" \
        --kld-tokens "$KLD_TOKENS" --output "${out_dir}/compare_cuda.json" \
        2>&1 | tee -a "$log_file"

    echo "  -> done (see ${out_dir}/compare_cuda.json and ${log_file})"
    echo
done

echo "Tier 1 sweep complete."
