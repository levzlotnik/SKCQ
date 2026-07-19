#!/usr/bin/env python3
"""Test build_codebook directly with real data."""

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from skcq.clustering import build_codebook, CodebookParams

# Load real data
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
model_dir = Path(snapshot_download(MODEL_ID))
index_path = model_dir / "model.safetensors.index.json"

import json
with open(index_path) as f:
    index = json.load(f)

weight_map = index["weight_map"]
shards = {}
for key, shard in weight_map.items():
    if "gate" in key and "model.layers.24" in key:
        shards["gate"] = shard
        break

gu_path = model_dir / shards["gate"]
with safe_open(gu_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if "gate" in key and "model.layers.24" in key:
            gate = f.get_tensor(key)
            break

# Reshape to (num_experts * out_dim, in_dim)
gate = gate.reshape(-1, gate.shape[-1])
print(f"Gate shape: {gate.shape}")

# Take first 131072 rows (full gate)
W = gate[:131072].float()
n_rows, in_dim = W.shape
n_blocks = 32
block_size = in_dim // n_blocks
K = 8192

print(f"Testing: n_rows={n_rows}, in_dim={in_dim}, n_blocks={n_blocks}, block_size={block_size}, K={K}")

# Build codebook
params = CodebookParams(
    k_gate=K, k_up=K, k_down=K,
    n_blocks_gate_up=n_blocks, n_blocks_down=n_blocks,
    n_codebooks=1,
    residual_k=None,
    max_iters=100,
    norm_threshold=0.001,
    skip_zeros=True,
    chunk_budget_mb=256,
)

# For gate: out_dim = intermediate_size = 1024, num_experts = 256
# n_rows = num_experts * out_dim = 256 * 1024 = 262144
# But we're using only 10000 rows, so we need to adjust
# Let's fake it: num_experts=10000/out_dim, but that doesn't work
# Instead, let's just use n_rows directly and set num_experts=n_rows, out_dim=1

# Use real dimensions: num_experts=256, out_dim=512 (since 256*512=131072)
result = build_codebook(
    rows=W.to(torch.device("cuda:0")),
    params=params,
    k=K,
    n_blocks=n_blocks,
    n_codebooks=1,
    num_experts=256,
    out_dim=512,
    device=torch.device("cuda:0"),
    name="test_build_codebook",
    distance_metric="cosine",
)

print(f"Result codebooks: {len(result.codebooks)}")
print(f"Result assignments: {len(result.assignments)}")
print(f"Result scales shape: {result.scales.shape}")

# Reconstruct
recon_blocks = []
for b in range(n_blocks):
    final_direction = torch.zeros(n_rows, block_size, dtype=torch.float32)
    for c in range(result.n_codebooks):
        cb_b = result.codebooks[c][b].float()
        asg_b = result.assignments[c].reshape(n_rows, n_blocks)[:, b]
        final_direction = final_direction + cb_b.t()[asg_b]
    
    scale_b = result.scales.reshape(n_rows, n_blocks)[:, b].float()
    recon_block = scale_b.unsqueeze(-1) * final_direction
    recon_blocks.append(recon_block)

recon = torch.cat(recon_blocks, dim=1)

# Compute error
err = torch.norm(W - recon) / torch.norm(W)
print(f"Reconstruction error: {err:.4f}")

# Check cosine per block
for b in range(3):
    cb_b = result.codebooks[0][b].float()
    asg_b = result.assignments[0].reshape(n_rows, n_blocks)[:, b]
    assigned = cb_b.t()[asg_b]
    
    raw_block = W[:, b*block_size:(b+1)*block_size]
    raw_norms = raw_block.norm(dim=-1)
    assigned_norms = assigned.norm(dim=-1)
    dots = (raw_block * assigned).sum(dim=-1)
    cosines = dots / (raw_norms * assigned_norms + 1e-10)
    
    scale_b = result.scales.reshape(n_rows, n_blocks)[:, b].float()
    
    print(f"Block {b}: cos={cosines.mean():.4f}±{cosines.std():.4f}, scale={scale_b.mean():.4f}±{scale_b.std():.4f}")
