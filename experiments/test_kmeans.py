#!/usr/bin/env python3
"""Minimal test of pt_kmeans with cosine distance."""

import torch
import torch.nn.functional as F
from pt_kmeans import kmeans

# Synthetic data: 1000 points in 64-d
n_points, dim = 1000, 64
data = torch.randn(n_points, dim)
data_unit = F.normalize(data, dim=-1)

# Run k-means with cosine distance
K = 128
codebook, labels = kmeans(
    data_unit,
    n_clusters=K,
    max_iters=100,
    distance_metric="cosine",
    init_method="kmeans++",
    x_pre_normalized=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    chunk_size=1000,
)

# Normalize centroids
centroids = F.normalize(codebook, dim=-1)

# Check cosine similarity
assigned = centroids[labels]
cos_sim = (data_unit * assigned).sum(dim=-1)
print(f"Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}, min={cos_sim.min():.4f}")

# Check codebook norms
print(f"Codebook norms: mean={centroids.norm(dim=-1).mean():.4f}, std={centroids.norm(dim=-1).std():.4f}")

# Reconstruction error
recon = cos_sim.unsqueeze(-1) * assigned
err = torch.norm(data_unit - recon) / torch.norm(data_unit)
print(f"Reconstruction error: {err:.4f}")
