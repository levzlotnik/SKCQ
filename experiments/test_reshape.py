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
gate = gate.reshape(-1, gate.shape[-1])
W = gate[:131072].float()
n_rows, in_dim = W.shape
n_blocks = 32
block_size = in_dim // n_blocks

print(f"Testing: n_rows={n_rows}, in_dim={in_dim}, n_blocks={n_blocks}, block_size={block_size}")

# Cluster each block directly (no build_codebook)
labels_per_block = []
codebooks_per_block = []
for b in range(n_blocks):
    block_data = W[:, b*block_size:(b+1)*block_size]
    result = _cluster_block(
        block_data,
        k=8192,
        max_iters=100,
        norm_threshold=0.001,
        skip_zeros=True,
        device=torch.device("cuda:0"),
        name=f"direct_block_{b}",
        distance_metric="cosine",
        chunk_budget_mb=256,
    )
    labels_per_block.append(result.labels.cpu())
    codebooks_per_block.append(result.codebook.cpu())

# Now simulate build_codebook's reshape/permute
# cb_assignments[c] has shape (n_blocks, n_rows)
cb_assignments = torch.stack(labels_per_block, dim=0)  # (32, 131072)
print(f"cb_assignments shape: {cb_assignments.shape}")

# Reshape to (n_blocks, num_experts, out_dim) = (32, 256, 512)
num_experts = 256
out_dim = 512
reshaped = cb_assignments.reshape(n_blocks, num_experts, out_dim)
print(f"reshaped shape: {reshaped.shape}")

# Permute to (num_experts, n_blocks, out_dim) = (256, 32, 512)
permuted = reshaped.permute(1, 0, 2).contiguous()
print(f"permuted shape: {permuted.shape}")

# Now reconstruct: for block b, get assignments
for b in range(3):
    # Method 1: direct from cb_assignments
    asg_direct = cb_assignments[b]  # (131072,)
    
    # Method 2: from permuted (like experiment script)
    asg_from_permuted = permuted.reshape(n_rows, n_blocks)[:, b]  # (131072,)
    
    # Check if they match
    match = (asg_direct == asg_from_permuted).all().item()
    print(f"Block {b}: assignments match = {match}")
    
    if not match:
        diff = (asg_direct != asg_from_permuted).sum().item()
        print(f"  Differences: {diff}/{n_rows}")
    
    # Check cosine with direct assignments
    cb_b = codebooks_per_block[b].float()
    assigned_direct = cb_b.t()[asg_direct]
    raw_block = W[:, b*block_size:(b+1)*block_size]
    raw_norms = raw_block.norm(dim=-1)
    assigned_norms = assigned_direct.norm(dim=-1)
    dots = (raw_block * assigned_direct).sum(dim=-1)
    cos_direct = dots / (raw_norms * assigned_norms + 1e-10)
    
    # Check cosine with permuted assignments
    assigned_permuted = cb_b.t()[asg_from_permuted]
    dots_permuted = (raw_block * assigned_permuted).sum(dim=-1)
    cos_permuted = dots_permuted / (raw_norms * assigned_permuted.norm(dim=-1) + 1e-10)
    
    print(f"Block {b}: cos_direct={cos_direct.mean():.4f}±{cos_direct.std():.4f}, cos_permuted={cos_permuted.mean():.4f}±{cos_permuted.std():.4f}")
