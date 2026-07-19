#!/usr/bin/env python3
"""Test codebook quality directly from _cluster_block."""

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from skcq.clustering import _cluster_block

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

print(f"Gate shape: {gate.shape}")

# Reshape to (num_experts * out_dim, in_dim)
gate = gate.reshape(-1, gate.shape[-1])
print(f"Gate reshaped: {gate.shape}")

# Take first 10000 rows for speed (still see real cosine)
W = gate[:10000].float()
n_rows, in_dim = W.shape
n_blocks = 32
block_size = in_dim // n_blocks

print(f"Block size: {block_size}")

# Normalize to unit sphere
unit = F.normalize(W, dim=-1)
unit_blocks = unit.reshape(n_rows, n_blocks, block_size)

# Cluster block 0
block_data = unit_blocks[:, 0, :]
result = _cluster_block(
    block_data,
    k=8192,
    max_iters=100,
    norm_threshold=0.001,
    skip_zeros=True,
    device=torch.device("cuda:0"),
    name="test_block0",
    distance_metric="cosine",
    chunk_budget_mb=256,
)

# Check codebook
codebook = result.codebook  # (block_size, K)
labels = result.labels  # (n_rows,)
scales = result.scales  # (n_rows,)

print(f"Codebook shape: {codebook.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Scales shape: {scales.shape}")

# Check codebook norms (centroids are columns)
cb_norms = codebook.norm(dim=0)  # norm of each column (centroid)
print(f"Codebook norms (per centroid): mean={cb_norms.mean():.4f}, std={cb_norms.std():.4f}")

# Check assigned centroids
assigned = codebook.t()[labels]  # (n_rows, block_size)
assigned_norms = assigned.norm(dim=-1)
print(f"Assigned norms: mean={assigned_norms.mean():.4f}, std={assigned_norms.std():.4f}")

# Check cosine similarity
raw_block = block_data
raw_norms = raw_block.norm(dim=-1)
dots = (raw_block * assigned).sum(dim=-1)
cosines = dots / (raw_norms * assigned_norms + 1e-10)
print(f"Cosine similarity: mean={cosines.mean():.4f}, std={cosines.std():.4f}, min={cosines.min():.4f}")

# Check scale distribution
print(f"Scales: mean={scales.mean():.4f}, std={scales.std():.4f}, min={scales.min():.4f}, max={scales.max():.4f}")
print(f"Negative scales: {(scales < 0).sum().item()}/{scales.shape[0]}")

# Reconstruction error
recon = scales.unsqueeze(-1) * assigned
err = torch.norm(raw_block - recon) / torch.norm(raw_block)
print(f"Reconstruction error: {err:.4f}")
