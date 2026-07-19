#!/usr/bin/env python3
"""Quick test: cluster one block with 131k rows."""

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

# Reshape to (num_experts * out_dim, in_dim)
gate = gate.reshape(-1, gate.shape[-1]).float()
print(f"Gate shape: {gate.shape}")

# Take block 0
block_data = gate[:, :64]  # (131072, 64)
print(f"Block shape: {block_data.shape}")

# Cluster
result = _cluster_block(
    block_data,
    k=8192,
    max_iters=100,
    norm_threshold=0.001,
    skip_zeros=True,
    device=torch.device("cuda:0"),
    name="test_large",
    distance_metric="cosine",
    chunk_budget_mb=256,
)

# Check cosine
codebook = result.codebook  # (64, K)
labels = result.labels  # (131072,)
scales = result.scales  # (131072,)

assigned = codebook.t()[labels]  # (131072, 64)
raw_norms = block_data.norm(dim=-1)
assigned_norms = assigned.norm(dim=-1)
dots = (block_data * assigned).sum(dim=-1)
cosines = dots / (raw_norms * assigned_norms + 1e-10)

print(f"Cosine: mean={cosines.mean():.4f}, std={cosines.std():.4f}, min={cosines.min():.4f}")
print(f"Scales: mean={scales.mean():.4f}, std={scales.std():.4f}")
print(f"Negative scales: {(scales < 0).sum().item()}/{scales.shape[0]}")

# Reconstruction error
recon = scales.unsqueeze(-1) * assigned
err = torch.norm(block_data - recon) / torch.norm(block_data)
print(f"Reconstruction error: {err:.4f}")
