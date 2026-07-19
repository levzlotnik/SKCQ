#!/usr/bin/env python3
"""Test if reshape/permute corrupts assignments."""

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
W = gate[:131072]
n_rows, in_dim = W.shape
n_blocks = 32
block_size = in_dim // n_blocks

print(f"n_rows={n_rows}, in_dim={in_dim}, n_blocks={n_blocks}, block_size={block_size}")

# Cluster each block
labels_list = []
codebooks_list = []
for b in range(n_blocks):
    block_data = W[:, b*block_size:(b+1)*block_size]
    result = _cluster_block(
        block_data,
        k=8192,
        max_iters=100,
        norm_threshold=0.001,
        skip_zeros=True,
        device=torch.device("cuda:0"),
        name=f"test_blk_{b}",
        distance_metric="cosine",
        chunk_budget_mb=256,
    )
    labels_list.append(result.labels)
    codebooks_list.append(result.codebook)

# Stack labels: (n_blocks, n_rows)
labels_stacked = torch.stack(labels_list, dim=0)
print(f"labels_stacked shape: {labels_stacked.shape}")

# Simulate build_codebook's reshape/permute
num_experts = 256
out_dim = 512

# Method 1: direct labels
for b in range(3):
    cb_b = codebooks_list[b].float()
    asg_direct = labels_stacked[b]  # (131072,)
    assigned_direct = cb_b.t()[asg_direct]
    raw_block = W[:, b*block_size:(b+1)*block_size]
    raw_norms = raw_block.norm(dim=-1)
    assigned_norms = assigned_direct.norm(dim=-1)
    dots = (raw_block * assigned_direct).sum(dim=-1)
    cos_direct = dots / (raw_norms * assigned_norms + 1e-10)
    print(f"Block {b} direct: cos={cos_direct.mean():.4f}±{cos_direct.std():.4f}")

# Method 2: reshape/permute (like build_codebook)
labels_reshaped = labels_stacked.reshape(n_blocks, num_experts, out_dim)  # (32, 256, 512)
labels_permuted = labels_reshaped.permute(1, 0, 2).contiguous()  # (256, 32, 512)
print(f"labels_permuted shape: {labels_permuted.shape}")

# Reconstruct: for block b, get assignments from permuted
for b in range(3):
    cb_b = codebooks_list[b].float()
    asg_permuted = labels_permuted.reshape(n_rows, n_blocks)[:, b]  # (131072,)
    assigned_permuted = cb_b.t()[asg_permuted]
    raw_block = W[:, b*block_size:(b+1)*block_size]
    raw_norms = raw_block.norm(dim=-1)
    assigned_norms = assigned_permuted.norm(dim=-1)
    dots = (raw_block * assigned_permuted).sum(dim=-1)
    cos_permuted = dots / (raw_norms * assigned_norms + 1e-10)
    print(f"Block {b} permuted: cos={cos_permuted.mean():.4f}±{cos_permuted.std():.4f}")

# Check if assignments match
for b in range(3):
    asg_direct = labels_stacked[b]
    asg_permuted = labels_permuted.reshape(n_rows, n_blocks)[:, b]
    match = (asg_direct == asg_permuted).all().item()
    diff = (asg_direct != asg_permuted).sum().item()
    print(f"Block {b}: match={match}, diff={diff}/{n_rows}")
