"""Standalone k-means clustering for PQ + residual codebooks.

No skcq dependencies — only torch + pt_kmeans. Both the ROCm and CUDA sides
import this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from pt_kmeans import kmeans
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodebookParams(BaseModel):
    k_gate: int = Field(default=4096, description="Codebook size for gate projection")
    k_up: int = Field(default=4096, description="Codebook size for up projection")
    k_down: int = Field(default=4096, description="Codebook size for down projection")
    n_blocks_gate_up: int = Field(
        default=1, description="Number of sub-blocks for gate/up input dim (PQ)"
    )
    n_blocks_down: int = Field(
        default=1, description="Number of sub-blocks for down input dim (PQ)"
    )
    n_codebooks: int = Field(
        default=2, description="Number of codebooks (1 = no residual, 2 = primary + 1 residual)"
    )
    max_iters: int = Field(default=100, description="Max k-means iterations")
    norm_threshold: float = Field(
        default=0.001,
        description="Rows with norm below this are treated as zeros, skipped from clustering",
    )
    skip_zeros: bool = Field(
        default=True,
        description="Whether to exclude near-zero rows from codebook building",
    )
    k_residual_mult: float = Field(
        default=1.0,
        description="K multiplier for residual codebooks. K_c = K_0 * k_residual_mult^c",
    )


class LayerOverride(BaseModel):
    """Override codebook params for a specific layer. None means use defaults."""

    k_gate: int | None = None
    k_up: int | None = None
    k_down: int | None = None
    n_blocks_gate_up: int | None = None
    n_blocks_down: int | None = None
    n_codebooks: int | None = None
    max_iters: int | None = None
    norm_threshold: float | None = None
    skip_zeros: bool | None = None
    k_residual_mult: float | None = None


@dataclass
class CodebookResult:
    """Unit-sphere residual codebook result for one projection.

    All codebooks operate in unit-sphere space: cb0 clusters the normalized
    directions (spherical k-means), cb1+ cluster the residual in unit-sphere
    space (euclidean k-means). A single scale per (expert, block, out_idx) is
    re-fit to the final reconstructed direction.

    Shapes:
        codebooks:   list of (n_blocks, block_size, K_c) — BMM-ready, one per codebook
        assignments: list of (num_experts, n_blocks, out_dim) int64 — one per codebook
        scales:      (num_experts, n_blocks, out_dim) — single scale (from cb0, re-fit)
        zero_mask:   (num_experts * out_dim,) — bool, full-row near-zero flag (metadata)
    """

    codebooks: list[torch.Tensor]
    assignments: list[torch.Tensor]
    scales: torch.Tensor
    zero_mask: torch.Tensor
    n_blocks: int
    n_codebooks: int


DistanceMetric = Literal["cosine", "euclidean"]


def _cluster_block(
    block_data: torch.Tensor,
    k: int,
    max_iters: int,
    norm_threshold: float,
    skip_zeros: bool,
    device: torch.device,
    name: str,
    distance_metric: DistanceMetric = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster one block's sub-vectors.

    Args:
        block_data: (n_rows, block_size)
        distance_metric: "cosine" (spherical k-means, centroids re-normalized,
            scale = dot(data, centroid)) or "euclidean" (l2 k-means, centroids
            are the raw reconstruction, scale unused/zero).

    Returns:
        codebook_b: (block_size, K) — BMM-ready (transposed from kmeans output)
        labels_full: (n_rows,) int64 — 0 for zero blocks
        scales_full: (n_rows,) — 0 for zero blocks (zeros for euclidean)
        recon_b: (n_rows, block_size) — scale * centroid[assign] (cosine) or
            centroid[assign] (euclidean); 0 for zero blocks
    """
    n_rows, block_size = block_data.shape
    block_norms = block_data.norm(dim=-1)
    block_zero = block_norms < norm_threshold

    if skip_zeros:
        non_zero = block_data[~block_zero].float()
        logger.info(
            "[%s] %d/%d block-rows below norm threshold %s",
            name,
            block_zero.sum().item(),
            n_rows,
            norm_threshold,
        )
    else:
        non_zero = block_data.float()
        block_zero = torch.zeros(n_rows, dtype=torch.bool, device=block_data.device)

    if non_zero.shape[0] == 0:
        codebook_b = torch.zeros(block_size, k, dtype=torch.float32, device=device)
        labels_full = torch.zeros(n_rows, dtype=torch.long, device=block_data.device)
        scales_full = torch.zeros(n_rows, dtype=torch.float32, device=block_data.device)
        recon_b = torch.zeros(n_rows, block_size, dtype=torch.float32, device=device)
        return codebook_b, labels_full, scales_full, recon_b

    logger.info(
        "[%s] clustering %d block-rows, k=%d, metric=%s",
        name,
        non_zero.shape[0],
        k,
        distance_metric,
    )

    k_eff = min(k, non_zero.shape[0])
    chunk_size = max(1, (2 * 1024**3) // (k_eff * 4))
    chunk_size = min(chunk_size, non_zero.shape[0])

    if distance_metric == "cosine":
        unit = F.normalize(non_zero, dim=-1)
        codebook_kmeans, labels_nz = kmeans(
            unit,
            n_clusters=k_eff,
            max_iters=max_iters,
            distance_metric="cosine",
            init_method="kmeans++",
            x_pre_normalized=True,
            device=device,
            chunk_size=chunk_size,
        )
        centroids_kbd = F.normalize(codebook_kmeans, dim=-1).to(block_data.device)
        labels_nz = labels_nz.to(block_data.device)
        assigned_centroids = centroids_kbd[labels_nz]
        scales_nz = (non_zero * assigned_centroids).sum(dim=-1)

        labels_full = torch.zeros(n_rows, dtype=torch.long, device=block_data.device)
        scales_full = torch.zeros(n_rows, dtype=torch.float32, device=block_data.device)
        labels_full[~block_zero] = labels_nz
        scales_full[~block_zero] = scales_nz
        recon_b = scales_full.unsqueeze(-1) * centroids_kbd[labels_full]
    else:  # euclidean (l2): centroids are the direct reconstruction, no scale
        codebook_kmeans, labels_nz = kmeans(
            non_zero,
            n_clusters=k_eff,
            max_iters=max_iters,
            distance_metric="l2",
            init_method="kmeans++",
            x_pre_normalized=False,
            device=device,
            chunk_size=chunk_size,
        )
        centroids_kbd = codebook_kmeans.to(block_data.device)
        labels_nz = labels_nz.to(block_data.device)
        labels_full = torch.zeros(n_rows, dtype=torch.long, device=block_data.device)
        labels_full[~block_zero] = labels_nz
        scales_full = torch.zeros(n_rows, dtype=torch.float32, device=block_data.device)
        recon_b = centroids_kbd[labels_full]
        recon_b[block_zero] = 0.0

    unique, counts = torch.unique(labels_nz, return_counts=True)
    logger.info(
        "[%s] non-empty clusters: %d/%d, sizes: min=%d, max=%d, mean=%.1f",
        name,
        len(unique),
        k,
        counts.min().item(),
        counts.max().item(),
        counts.float().mean().item(),
    )

    codebook_b = centroids_kbd.t().contiguous()
    return codebook_b, labels_full, scales_full, recon_b


def build_codebook(
    rows: torch.Tensor,
    params: CodebookParams,
    k: int,
    n_blocks: int,
    n_codebooks: int,
    num_experts: int,
    out_dim: int,
    device: torch.device | None = None,
    name: str = "",
) -> CodebookResult:
    """Build a unit-sphere residual PQ codebook for one projection.

    All codebooks operate in unit-sphere space. The full weight rows are first
    normalized to the unit sphere; cb0 clusters these directions with spherical
    k-means (cosine). cb1+ cluster the residual in unit-sphere space with
    euclidean (l2) k-means. A single scale per (expert, block, out_idx) is
    re-fit to the final reconstructed direction:
        scale = dot(raw_block, sum_c centroid_c[assign_c]) / ||final_direction||^2

    Args:
        rows: (num_experts * out_dim, in_dim) — weight rows, expert-major
        k: base codebook size (K_0); K_c = K_0 * k_residual_mult^c
        n_blocks: number of input-dim sub-blocks (PQ)
        n_codebooks: number of codebooks (1 = no residual, 2+ = primary + residuals)
        num_experts, out_dim: for reshaping assignments/scales
    """
    if device is None:
        device = rows.device

    n_rows, in_dim = rows.shape
    if in_dim % n_blocks != 0:
        raise ValueError(f"in_dim={in_dim} not divisible by n_blocks={n_blocks}")
    block_size = in_dim // n_blocks

    row_norms = rows.norm(dim=-1)
    zero_mask = row_norms < params.norm_threshold
    logger.info(
        "[%s] %d/%d full rows below norm threshold %s",
        name,
        zero_mask.sum().item(),
        n_rows,
        params.norm_threshold,
    )

    raw = rows.float().clone()
    unit = F.normalize(raw, dim=-1)
    unit_blocks = unit.reshape(n_rows, n_blocks, block_size)
    raw_blocks = raw.reshape(n_rows, n_blocks, block_size)

    k_per_codebook = [max(1, int(k * params.k_residual_mult**c)) for c in range(n_codebooks)]

    cb_codebooks: list[torch.Tensor] = []
    cb_assignments: list[torch.Tensor] = []

    unit_residual = unit_blocks.clone()

    for c in range(n_codebooks):
        k_c = k_per_codebook[c]
        metric: DistanceMetric = "cosine" if c == 0 else "euclidean"
        block_codebooks: list[torch.Tensor] = []
        block_assigns: list[torch.Tensor] = []
        for b in range(n_blocks):
            block_data = unit_residual[:, b, :]
            codebook_b, labels_b, _scales_b, _recon_b = _cluster_block(
                block_data,
                k=k_c,
                max_iters=params.max_iters,
                norm_threshold=params.norm_threshold,
                skip_zeros=params.skip_zeros,
                device=device,
                name=f"{name} cb={c}/{n_codebooks} blk={b}/{n_blocks}",
                distance_metric=metric,
            )
            block_codebooks.append(codebook_b)
            block_assigns.append(labels_b)
            subtract = codebook_b.t()[labels_b]
            subtract[zero_mask] = 0.0
            unit_residual[:, b, :] = unit_residual[:, b, :] - subtract
        cb_codebooks.append(torch.stack(block_codebooks, dim=0))
        cb_assignments.append(torch.stack(block_assigns, dim=0))

    # Re-fit a single scale per (row, block) to the final reconstructed direction.
    scales = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.float32, device=device)
    for b in range(n_blocks):
        final_direction = torch.zeros(n_rows, block_size, dtype=torch.float32, device=device)
        for c in range(n_codebooks):
            cb_b = cb_codebooks[c][b]
            asg_b = cb_assignments[c][b]
            final_direction = final_direction + cb_b.t()[asg_b]
        raw_block = raw_blocks[:, b, :]
        scale = (raw_block * final_direction).sum(dim=-1) / (
            final_direction.norm(dim=-1) ** 2 + 1e-10
        )
        scales[:, b, :] = scale.reshape(num_experts, out_dim)

    assignments: list[torch.Tensor] = []
    for c in range(n_codebooks):
        a = (
            cb_assignments[c]
            .reshape(n_blocks, num_experts, out_dim)
            .permute(1, 0, 2)
            .contiguous()
        )
        assignments.append(a)

    codebooks_out = [cb.to(rows.dtype).cpu() for cb in cb_codebooks]

    return CodebookResult(
        codebooks=codebooks_out,
        assignments=[a.cpu() for a in assignments],
        scales=scales.to(rows.dtype).cpu(),
        zero_mask=zero_mask.cpu(),
        n_blocks=n_blocks,
        n_codebooks=n_codebooks,
    )
