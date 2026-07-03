#!/usr/bin/env bash
# Run the full sweep: for each YAML config in configs/sweep/, run --compare.
# Skips configs whose codebooks already exist (eval-only on rerun).
# Results land in compare.json per codebook dir; logs in sweep_logs/.
#
# Usage: scripts/sweep.sh [EVAL_SAMPLES]    (default: from YAML, typically 100)
set -euo pipefail

CONFIG_DIR="configs/sweep"
LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"

EVAL_ARG=""
if [ $# -ge 1 ]; then
    EVAL_ARG="--eval-samples $1"
fi

configs=("${CONFIG_DIR}"/*.yaml)
if [ ${#configs[@]} -eq 0 ]; then
    echo "No configs found in ${CONFIG_DIR}/. Run scripts/gen_sweep_configs.sh first."
    exit 1
fi

echo "Found ${#configs[@]} configs. Starting sweep..."
echo

i=0
total=${#configs[@]}
for cfg in "${configs[@]}"; do
    i=$((i + 1))
    label=$(basename "$cfg" .yaml)
    log_file="${LOG_DIR}/${label}.log"

    echo "[$i/$total] ${label}"

    # Determine output dir from the config to check if codebooks already exist
    out_dir=$(grep "^output_dir:" "$cfg" | awk -F'"' '{print $2}')
    if [ -z "$out_dir" ]; then
        out_dir="codebooks_${label}"
    fi

    if [ -d "${out_dir}/layer_0" ]; then
        echo "  -> codebooks exist in ${out_dir}/, skipping build (load + eval only)"
    else
        echo "  -> building codebooks into ${out_dir}/"
    fi

    rocm/.venv/bin/python build.py --config "$cfg" --compare --use-cuda-worker --baseline-cache baseline.pt $EVAL_ARG 2>&1 | tee "$log_file"

    echo "  -> done (see ${out_dir}/ and ${log_file})"
    echo
done

echo "Sweep complete. Collecting results..."

# Gather all compare.json into a summary
summary="${LOG_DIR}/summary.json"
uv run python -c "
import json, glob, os
results = []
for path in sorted(glob.glob('codebooks_*/compare.json')):
    with open(path) as f:
        d = json.load(f)
    d['dir'] = os.path.dirname(path)
    results.append(d)
with open('${summary}', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'Wrote ${summary} with {len(results)} results')
print()
print(f'{\"label\":25} {\"base_ppl\":>10} {\"quant_ppl\":>10} {\"ratio\":>7} {\"kld\":>8}')
for r in sorted(results, key=lambda x: x.get('quantized_ppl', 0)):
    label = os.path.basename(r['dir']).replace('codebooks_', '')
    bp = r.get('baseline_ppl', 0)
    qp = r.get('quantized_ppl', 0)
    ratio = r.get('ppl_ratio', 0)
    kld = r.get('kld_base_to_quant', 0)
    print(f'{label:25} {bp:10.4f} {qp:10.4f} {ratio:7.4f} {kld:8.6f}')
"

echo
echo "Per-run logs in ${LOG_DIR}/, summary in ${summary}"
