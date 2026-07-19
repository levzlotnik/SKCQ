#!/usr/bin/env python3
"""Test reconstruction logic with synthetic data matching the real experiment."""

import torch
import torch.nn.functional as F

# Synthetic data matching real experiment
n_rows, in_dim = 131072, 2048
n_blocks = 32
block_size = in_dim // n_blocks  # 64

# Generate data with small norms (like real weights)
raw_full = torch.randn(n_rows, in_dim) * 0.05  # Scale down to match real weights
W_norm = raw_full.norm().item()
print(f"Full matrix norm: {W_norm:.4f}")
print(f"Per-row norm: mean={raw_full.norm(dim=-1).mean():.4f}, std={raw_full.norm(dim=-1).std():.4f}")

# Reshape into blocks
raw_blocks = raw_full.reshape(n_rows, n_blocks, block_size)

# Normalize each row to unit sphere
unit = F.normalize(raw_full, dim=-1)
unit_blocks = unit.reshape(n_rows, n_blocks, block_size)

# Fake codebook: random unit vectors
K = 8192
cb0 = F.normalize(torch.randn(block_size, K), dim=0)

# Fake assignments
assign0 = torch.randint(0, K, (n_rows, n_blocks))

# Reconstruct block by block (matching the real code)
recon_blocks = []
for b in range(n_blocks):
    # Get unit vectors for this block
    unit_b = unit_blocks[:, b, :]  # (n_rows, block_size)
    raw_b = raw_blocks[:, b, :]  # (n_rows, block_size)
    
    # Direction from codebook
    direction = cb0.t()[assign0[:, b]]  # (n_rows, block_size)
    
    # Optimal scale
    scale = (raw_b * direction).sum(dim=-1) / (direction.norm(dim=-1) ** 2 + 1e-10)
    
    # Reconstruct
    recon_b = scale.unsqueeze(-1) * direction
    recon_blocks.append(recon_b)

recon_full = torch.cat(recon_blocks, dim=1)  # (n_rows, in_dim)

# Compute error
err = torch.norm(raw_full - recon_full) / torch.norm(raw_full)
print(f"\nReconstruction error: {err:.6f}")
print(f"Expected: <= 1.0")

# Check per-row error
per_row_err = torch.norm(raw_full - recon_full, dim=-1) / torch.norm(raw_full, dim=-1)
print(f"Per-row error: mean={per_row_err.mean():.6f}, max={per_row_err.max():.6f}")

# Check scale stats
all_scales = []
for b in range(n_blocks):
    raw_b = raw_blocks[:, b, :]
    direction = cb0.t()[assign0[:, b]]
    scale = (raw_b * direction).sum(dim=-1) / (direction.norm(dim=-1) ** 2 + 1e-10)
    all_scales.append(scale)
all_scales = torch.cat(all_scales)
print(f"\nScale stats: mean={all_scales.mean():.4f}, std={all_scales.std():.4f}, min={all_scales.min():.4f}, max={all_scales.max():.4f}")

# Check direction norms
all_dirs = []
for b in range(n_blocks):
    direction = cb0.t()[assign0[:, b]]
    all_dirs.append(direction.norm(dim=-1))
all_dirs = torch.cat(all_dirs)
print(f"Direction norms: mean={all_dirs.mean():.4f}, std={all_dirs.std():.4f}")

# Check raw block norms
all_raw_norms = []
for b in range(n_blocks):
    raw_b = raw_blocks[:, b, :]
    all_raw_norms.append(raw_b.norm(dim=-1))
all_raw_norms = torch.cat(all_raw_norms)
print(f"Raw block norms: mean={all_raw_norms.mean():.4f}, std={all_raw_norms.std():.4f}")

