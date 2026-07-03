#!/usr/bin/env bash
# Generate sweep YAML configs into configs/sweep/.
# Sweeps:
#   ratio (K/block_size): 4, 16, 64
#   n_codebooks (residuals): 1, 2, 3  (1=no residual, 2=+1, 3=+2)
#   n_blocks: 1, 2, 4
#
# in_dim for gate/up = 2048 (hidden_size), for down = 512 (intermediate_size)
# block_size_gu = 2048 / n_blocks_gu, block_size_dn = 512 / n_blocks_dn
# K = ratio * block_size
#
# Usage: scripts/gen_sweep_configs.sh
set -euo pipefail

OUT_DIR="configs/sweep"
mkdir -p "$OUT_DIR"

gen() {
    local label="$1"; shift
    local ratio="$1"; shift
    local nb="$1"; shift
    local n_cb="$1"; shift

    local k_gu=$(( ratio * (2048 / nb) ))
    local k_dn=$(( ratio * (512 / nb) ))

    cat > "${OUT_DIR}/${label}.yaml" <<EOF
model_id: "Qwen/Qwen3.6-35B-A3B"
eval_samples: 100
output_dir: "codebooks_${label}"

defaults:
  k_gate: ${k_gu}
  k_up: ${k_gu}
  k_down: ${k_dn}
  n_blocks_gate_up: ${nb}
  n_blocks_down: ${nb}
  n_codebooks: ${n_cb}
  max_iters: 100
  norm_threshold: 0.001
  skip_zeros: true

layer_overrides:
  0:
    skip_zeros: true
    norm_threshold: 0.001
EOF
    local bs_gu=$(( 2048 / nb ))
    local bs_dn=$(( 512 / nb ))
    echo "  ${label}.yaml  (N_gu=${nb}*${bs_gu}=${nb}*${bs_gu}, k_gu=${k_gu}, N_dn=${nb}*${bs_dn}=${nb}*${bs_dn}, k_dn=${k_dn}, ratio=${ratio}, n_cb=${n_cb})"
}

echo "Generating sweep configs (ratio x n_blocks x n_codebooks):"
echo

count=0
for ratio in 4 16 64; do
    for nb in 1 2 4; do
        for n_cb in 1 2 3; do
            gen "r${ratio}_nb${nb}_cb${n_cb}" "$ratio" "$nb" "$n_cb"
            count=$((count + 1))
        done
    done
done

echo
echo "Generated ${count} configs in ${OUT_DIR}/"
