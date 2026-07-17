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
from pydantic import BaseModel, Field
from tqdm import tqdm

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
        description="K_0/K_r ratio: primary codebook size divided by residual codebook size. K_c = K_0 / k_residual_mult^c (e.g. 32 means K_r = K/32)",
    )
    chunk_budget_mb: int = Field(
        default=2048,
        description="Memory budget (MB) for k-means chunking — reduce for low-VRAM GPUs",
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
    chunk_budget_mb: int | None = None


@dataclass
class BlockClusterResult:
    """Result of clustering one block's sub-vectors.

    For cosine (spherical): codebook holds unit-norm directions, scale = dot(data, centroid).
    For euclidean (l2): codebook holds raw centroids (direct reconstruction), scale = 0.

    Shapes:
        codebook:  (d, K) — BMM-ready (transposed); unit-norm centroids (cosine) or raw (euclidean)
        labels:    (n_rows,) int64 — 0 for zero blocks
        scales:    (n_rows,) — 0 for zero blocks (zeros for euclidean)
    """

    codebook: torch.Tensor
    labels: torch.Tensor
    scales: torch.Tensor


@dataclass
class CodebookResult:
    """Unit-sphere residual codebook result for one projection.

    All codebooks operate in unit-sphere space: cb0 clusters the normalized
    directions (spherical k-means), cb1+ cluster the residual in unit-sphere
    space (euclidean k-means). A single scale per (row, block) is
    re-fit to the final reconstructed direction.

    Shapes:
        codebooks:   list of (n_blocks, block_size, K_c) — BMM-ready, one per codebook
                     When shared_codebook=True, shape is (1, block_size, K_c) —
                     one codebook reused across all blocks.
        assignments: list of (n_blocks, n_rows) int64 — one per codebook
        scales:      (n_rows, n_blocks) — single scale (from cb0, re-fit)
        zero_mask:   (n_rows,) — bool, full-row near-zero flag (metadata)
    """

    codebooks: list[torch.Tensor]
    assignments: list[torch.Tensor]
    scales: torch.Tensor
    zero_mask: torch.Tensor
    n_blocks: int
    n_codebooks: int
    shared_codebook: bool = False
    sign_bits: torch.Tensor | None = None  # (n_rows, n_blocks, block_size) if sign_split


DistanceMetric = Literal["cosine", "euclidean"]


def _assign_to_centroids(
    data: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign each data point to its nearest centroid (cosine: max dot product).

    Data stays on its original device; chunks are moved to centroids' device
    for the matmul, results moved back.
    """
    if centroids.shape[0] == data.shape[1]:
        cb = centroids.t()
    else:
        cb = centroids
    n = data.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=data.device)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = data[i:end].to(cb.device)
        dists = chunk @ cb.t()
        labels[i:end] = dists.argmax(dim=-1).to(data.device)
    return labels


def _assign_to_centroids_l2(
    data: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign each data point to its nearest centroid (l2: min squared distance).

    Data stays on its original device; chunks are moved to centroids' device
    for the matmul, results moved back.
    """
    if centroids.shape[0] == data.shape[1]:
        cb = centroids.t()
    else:
        cb = centroids
    n = data.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=data.device)
    cb_sq = cb.square().sum(dim=-1).unsqueeze(0)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = data[i:end].to(cb.device)
        dots = chunk @ cb.t()
        dists = -2 * dots + cb_sq
        labels[i:end] = dists.argmin(dim=-1).to(data.device)
    return labels


def _sobol_first_orthant(k: int, d: int, device: torch.device) -> torch.Tensor:
    """Generate k unit-norm points on the first orthant using a Sobol sequence.

    1. Generate k points in [0,1]^d via Sobol (low-discrepancy, space-filling)
    2. Map to first orthant of unit sphere via inverse transform:
       - Treat each coordinate as a direction sample
       - Normalize to unit sphere (all coordinates positive since Sobol ∈ [0,1])
    """
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=42)
    points = sobol.draw(k).to(device)  # (k, d) in [0,1]
    # Avoid exact zeros (would give zero norm after normalization)
    points = points.clamp(min=1e-6)
    # Normalize to unit sphere — all positive → first orthant
    return F.normalize(points, dim=-1)


def _sobol_unit_cube(k: int, d: int, device: torch.device) -> torch.Tensor:
    """Generate k points in [0,1]^d via Sobol sequence (for euclidean k-means init).

    Unlike _sobol_first_orthant (which normalizes to unit sphere), this keeps
    points in the cube — appropriate for euclidean data that isn't on a sphere.
    Scales to [-1, 1] to be roughly centered around zero.
    """
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=42)
    points = sobol.draw(k).to(device)  # (k, d) in [0,1]
    points = points * 2 - 1  # [-1, 1]
    return points


def _euclidean_kmeans(
    data: torch.Tensor,
    k: int,
    max_iters: int,
    device: torch.device,
    name: str,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard euclidean (l2) k-means with Sobol init and tqdm progress bar.

    Assignment: argmin ||x_i - centroid_c||^2
    Update:     centroid_c = mean(x_i for i in c)

    Args:
        data: (n, d) — raw data on CPU
    Returns:
        centroids: (K, d) — on device
        labels: (n,) — on CPU
    """
    n, d = data.shape
    k_eff = min(k, n)

    # Sobol init in [-1,1]^d — O(k), no data dependency
    logger.info("[%s] Sobol init (k=%d, d=%d, euclidean)...", name, k_eff, d)
    centroids = _sobol_unit_cube(k_eff, d, device)  # (k, d)

    pbar = tqdm(range(max_iters), desc=name, leave=False)
    for it in pbar:
        # Assign: argmin ||x - c||^2 = argmin(-2*x·c + ||c||^2)
        labels = _assign_to_centroids_l2(data, centroids.t().contiguous(), chunk_size, device)

        # Update: centroid_c = mean(x_i for i in c)
        new_centroids = torch.zeros(k_eff, d, dtype=torch.float32, device=device)
        counts = torch.zeros(k_eff, device=device)
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = data[i:end].to(device)
            chunk_labels = labels[i:end].to(device)
            new_centroids.index_add_(0, chunk_labels, chunk)
            counts.index_add_(0, chunk_labels, torch.ones(end - i, device=device))

        # Handle empty clusters: re-init from Sobol
        empty = counts == 0
        n_empty = empty.sum().item()
        if n_empty > 0:
            new_centroids[empty] = _sobol_unit_cube(n_empty, d, device)
        new_centroids[~empty] = new_centroids[~empty] / counts[~empty].unsqueeze(-1)

        # Check convergence
        moved = (new_centroids - centroids).norm().item()
        centroids = new_centroids

        pbar.set_postfix(moved=f"{moved:.6f}", empty=n_empty)

        if moved < 1e-6:
            logger.info("[%s] converged at iter %d (moved=%.6f)", name, it, moved)
            break
    pbar.close()

    # Final assignment
    labels = _assign_to_centroids_l2(data, centroids.t().contiguous(), chunk_size, device)

    logger.info("[%s] k-means done after %d iters", name, it + 1)
    return centroids, labels.cpu()  # (K, d), (n,)


def _norm_weighted_spherical_kmeans(
    data: torch.Tensor,
    raw_data: torch.Tensor,
    k: int,
    max_iters: int,
    device: torch.device,
    name: str,
    chunk_size: int,
    first_orthant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Norm-weighted spherical k-means.

    Assignment: argmax(unit_i · centroid_c)  (cosine similarity)
    Update:     centroid_c = normalize(Σ_{i∈c} raw_data_i)  (norm-weighted mean)

    This minimizes Σ ||x_i||² × (1 - cos²(x_i, centroid)) instead of
    the unweighted Σ (1 - cos²(x_i, centroid)).

    Args:
        data: (n, d) — unit-normalized data (for assignment)
        raw_data: (n, d) — raw data (for norm-weighted centroid update)
    Returns:
        centroids: (K, d) — unit-norm
        labels: (n,) — on CPU
    """
    n, d = data.shape
    k_eff = min(k, n)

    unit_data = F.normalize(data, dim=-1)

    # Deterministic Sobol init — space-filling, O(k), no data dependency
    logger.info("[%s] Sobol init (k=%d, d=%d, first_orthant=%s)...", name, k_eff, d, first_orthant)
    centroids = _sobol_first_orthant(k_eff, d, device)  # (k, d)
    centroids_t = centroids.t().contiguous()  # (d, k)

    pbar = tqdm(range(max_iters), desc=name, leave=False)
    for it in pbar:
        # Assign: argmax(unit_i · centroid_c)
        labels = _assign_to_centroids(unit_data, centroids_t, chunk_size, device)

        # Update: centroid_c = normalize(Σ_{i∈c} raw_data_i) — norm-weighted!
        new_centroids = torch.zeros(k_eff, d, dtype=torch.float32, device=device)
        counts = torch.zeros(k_eff, device=device)
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = raw_data[i:end].to(device)
            chunk_labels = labels[i:end].to(device)
            new_centroids.index_add_(0, chunk_labels, chunk)
            counts.index_add_(0, chunk_labels, torch.ones(end - i, device=device))

        # Handle empty clusters: re-init from Sobol
        empty = counts == 0
        n_empty = empty.sum().item()
        if n_empty > 0:
            new_centroids[empty] = _sobol_first_orthant(n_empty, d, device)

        # Normalize to unit sphere
        new_centroids = F.normalize(new_centroids, dim=-1)

        # Check convergence
        moved = (new_centroids - centroids).norm().item()
        centroids = new_centroids
        centroids_t = centroids.t().contiguous()

        pbar.set_postfix(moved=f"{moved:.6f}", empty=n_empty)

        if moved < 1e-6:
            logger.info("[%s] converged at iter %d (moved=%.6f)", name, it, moved)
            break
    pbar.close()

    # Final assignment
    labels = _assign_to_centroids(unit_data, centroids_t, chunk_size, device)

    logger.info("[%s] k-means done after %d iters", name, it + 1)
    return centroids, labels.cpu()  # (K, d), (n,)


def _cluster_block(
    block_data: torch.Tensor,
    k: int,
    max_iters: int,
    norm_threshold: float,
    skip_zeros: bool,
    device: torch.device,
    name: str,
    distance_metric: DistanceMetric = "cosine",
    chunk_budget_mb: int = 2048,
    max_train_samples: int = 0,
    first_orthant: bool = False,
    raw_data: torch.Tensor | None = None,
) -> BlockClusterResult:
    """Cluster one block's sub-vectors.

    Args:
        block_data: (n_rows, block_size) — unit-normalized for cosine
        raw_data: (n_rows, block_size) — raw data for norm-weighted centroid update.
            If None, falls back to block_data (no norm-weighting).
        distance_metric: "cosine" (spherical k-means, centroids re-normalized,
            scale = dot(data, centroid)) or "euclidean" (l2 k-means, centroids
            are the raw reconstruction, scale unused/zero).
        max_train_samples: if >0, sub-sample to this many points for k-means
            training (init + iterations), then re-assign all points. Needed
            because torch.multinomial (k-means++ init) is limited to 2^24.
    """
    n_rows, block_size = block_data.shape
    # Keep data on CPU — only k-means chunks and centroids use GPU
    block_data = block_data.cpu()
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
        return BlockClusterResult(
            codebook=torch.zeros(block_size, k, dtype=torch.float32, device=device),
            labels=torch.zeros(n_rows, dtype=torch.long, device=block_data.device),
            scales=torch.zeros(n_rows, dtype=torch.float32, device=block_data.device),
        )

    logger.info(
        "[%s] clustering %d block-rows, k=%d, metric=%s",
        name,
        non_zero.shape[0],
        k,
        distance_metric,
    )

    k_eff = min(k, non_zero.shape[0])
    budget_bytes = chunk_budget_mb * 1024 * 1024
    chunk_size = max(1, budget_bytes // (k_eff * 4))
    chunk_size = min(chunk_size, non_zero.shape[0])

    # Sub-sample for k-means training if dataset is too large (torch.multinomial
    # in k-means++ init is limited to 2^24 categories)
    MAX_MULTINOMIAL = (2 ** 24) - 1
    train_data = non_zero
    train_raw = raw_data if raw_data is not None else non_zero
    if max_train_samples > 0 and non_zero.shape[0] > max_train_samples:
        perm = torch.randperm(non_zero.shape[0], device=non_zero.device)[:max_train_samples]
        train_data = non_zero[perm]
        train_raw = train_raw[perm]
        logger.info("[%s] sub-sampled %d -> %d for k-means training", name, non_zero.shape[0], train_data.shape[0])
    elif non_zero.shape[0] > MAX_MULTINOMIAL:
        perm = torch.randperm(non_zero.shape[0], device=non_zero.device)[:MAX_MULTINOMIAL]
        train_data = non_zero[perm]
        train_raw = train_raw[perm]
        logger.info("[%s] sub-sampled %d -> %d for k-means training (multinomial limit)", name, non_zero.shape[0], train_data.shape[0])

    train_chunk_size = min(chunk_size, train_data.shape[0])

    if distance_metric == "cosine":
        unit = F.normalize(non_zero, dim=-1)

        # Norm-weighted spherical k-means: centroid update uses raw (unnormalized) data
        # so high-norm blocks pull centroids toward their direction.
        centroids_kbd, labels_nz = _norm_weighted_spherical_kmeans(
            train_data,
            raw_data=train_raw,
            k=k_eff,
            max_iters=max_iters,
            device=device,
            name=name,
            chunk_size=train_chunk_size,
            first_orthant=first_orthant,
        )
        # centroids_kbd is (K, d) on GPU — keep on GPU for fast assignment
        # _assign_to_centroids handles cross-device (CPU data, GPU centroids)

        # Re-assign ALL points to the learned centroids
        labels_nz = _assign_to_centroids(unit, centroids_kbd, chunk_size, device)
        assigned_centroids = centroids_kbd[labels_nz.to(centroids_kbd.device)].to(unit.device)
        scales_nz = torch.einsum('nd,nd->n', non_zero, assigned_centroids)

        cos_sim = torch.einsum('nd,nd->n', unit, assigned_centroids)
        logger.info(
            "[%s] training cos(raw_unit, centroid): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            name, cos_sim.mean().item(), cos_sim.std().item(),
            cos_sim.min().item(), cos_sim.max().item(),
        )

        # Flip centroids with negative dot products to ensure positive scales
        neg_mask = scales_nz < 0
        if neg_mask.any():
            neg_labels = labels_nz[neg_mask].unique().to(centroids_kbd.device)
            centroids_kbd[:, neg_labels] = -centroids_kbd[:, neg_labels]
            assigned_centroids = centroids_kbd[labels_nz.to(centroids_kbd.device)].to(unit.device)
            scales_nz = torch.einsum('nd,nd->n', non_zero, assigned_centroids)
            logger.info(
                "[%s] flipped %d/%d centroids with negative dot products",
                name, neg_mask.sum().item(), non_zero.shape[0],
            )

        labels_full = torch.zeros(n_rows, dtype=torch.long, device=block_data.device)
        scales_full = torch.zeros(n_rows, dtype=torch.float32, device=block_data.device)
        labels_full[~block_zero] = labels_nz.to(block_data.device)
        scales_full[~block_zero] = scales_nz
    else:  # euclidean (l2): centroids are the direct reconstruction, no scale
        centroids_kbd, labels_nz = _euclidean_kmeans(
            train_data,
            k=k_eff,
            max_iters=max_iters,
            device=device,
            name=name,
            chunk_size=train_chunk_size,
        )

        # Re-assign ALL points to the learned centroids (l2: argmin distance)
        labels_nz = _assign_to_centroids_l2(non_zero, centroids_kbd.t().contiguous(), chunk_size, device)
        labels_full = torch.zeros(n_rows, dtype=torch.long, device=block_data.device)
        labels_full[~block_zero] = labels_nz
        scales_full = torch.zeros(n_rows, dtype=torch.float32, device=block_data.device)

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
    return BlockClusterResult(
        codebook=codebook_b,
        labels=labels_full,
        scales=scales_full,
    )


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
    distance_metric: DistanceMetric = "cosine",
    shared_codebook: bool = False,
    sign_split: bool = False,
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
        k: base codebook size (K_0); K_c = K_0 / k_residual_mult^c
        n_blocks: number of input-dim sub-blocks (PQ)
        n_codebooks: number of codebooks (1 = no residual, 2+ = primary + residuals)
        num_experts, out_dim: for reshaping assignments/scales
        shared_codebook: if True, pool all blocks' sub-vectors into a single
            shared codebook (one k-means for all blocks). Reduces codebook
            storage by n_blocks and gives n_blocks× more samples per centroid.
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

    # Sign-splitting: extract signs, fold to first orthant for clustering
    if sign_split:
        signs = torch.sign(unit_blocks)  # (n_rows, n_blocks, block_size) — ±1
        signs[signs == 0] = 1.0  # handle exact zeros
        unit_blocks = unit_blocks * signs  # all-positive (first orthant)
        logger.info("[%s] sign-split enabled: clustering on first orthant", name)
    else:
        signs = None

    k_per_codebook = [max(1, int(k / params.k_residual_mult**c)) for c in range(n_codebooks)]

    cb_codebooks: list[torch.Tensor] = []
    cb_assignments: list[torch.Tensor] = []

    unit_residual = unit_blocks.clone()
    # raw_residual mirrors unit_residual but with original magnitudes (folded to first orthant if sign_split)
    if sign_split:
        raw_residual = raw_blocks * signs  # first-orthant raw data
    else:
        raw_residual = raw_blocks.clone()

    for c in range(n_codebooks):
        k_c = k_per_codebook[c]
        metric: DistanceMetric = distance_metric if c == 0 else "euclidean"

        if shared_codebook:
            # Pool all blocks' sub-vectors into one (n_rows * n_blocks, block_size) tensor
            pooled = unit_residual.reshape(n_rows * n_blocks, block_size)
            pooled_raw = raw_residual.reshape(n_rows * n_blocks, block_size)
            result_pool = _cluster_block(
                pooled,
                k=k_c,
                max_iters=params.max_iters,
                norm_threshold=params.norm_threshold,
                skip_zeros=params.skip_zeros if c == 0 else False,
                device=device,
                name=f"{name} cb={c}/{n_codebooks} (shared)",
                distance_metric=metric,
                chunk_budget_mb=params.chunk_budget_mb,
                max_train_samples=2**23,  # 8M samples for training, rest re-assigned
                first_orthant=sign_split,
                raw_data=pooled_raw,
            )
            # codebook: (block_size, K_c) — single shared codebook
            # labels: (n_rows * n_blocks,) → reshape to (n_rows, n_blocks) → transpose to (n_blocks, n_rows)
            labels_2d = result_pool.labels.reshape(n_rows, n_blocks)
            cb_codebooks.append(result_pool.codebook.unsqueeze(0))  # (1, block_size, K_c)
            cb_assignments.append(labels_2d.t().contiguous())  # (n_blocks, n_rows)

            # Residual subtraction
            assigned = result_pool.codebook.t()[result_pool.labels]  # (n_rows*n_blocks, block_size)
            assigned = assigned.reshape(n_rows, n_blocks, block_size)
            assigned[zero_mask] = 0.0
            dot = torch.einsum('nbd,nbd->nb', unit_residual, assigned).unsqueeze(-1)
            unit_residual = unit_residual - dot * assigned
        else:
            block_codebooks: list[torch.Tensor] = []
            block_assigns: list[torch.Tensor] = []
            for b in range(n_blocks):
                block_data = unit_residual[:, b, :]
                block_raw = raw_residual[:, b, :]
                result_b = _cluster_block(
                    block_data,
                    k=k_c,
                    max_iters=params.max_iters,
                    norm_threshold=params.norm_threshold,
                    skip_zeros=params.skip_zeros if c == 0 else False,
                    device=device,
                    name=f"{name} cb={c}/{n_codebooks} blk={b}/{n_blocks}",
                    distance_metric=metric,
                    chunk_budget_mb=params.chunk_budget_mb,
                    first_orthant=sign_split,
                    raw_data=block_raw,
                )
                block_codebooks.append(result_b.codebook)
                block_assigns.append(result_b.labels)

                subtract = result_b.codebook.t()[result_b.labels]
                subtract[zero_mask] = 0.0
                dot = torch.einsum('nd,nd->n', unit_residual[:, b, :], subtract).unsqueeze(-1)
                unit_residual[:, b, :] = unit_residual[:, b, :] - dot * subtract
            cb_codebooks.append(torch.stack(block_codebooks, dim=0))
            cb_assignments.append(torch.stack(block_assigns, dim=0))

    # Re-fit a single scale per (row, block) to the final reconstructed direction.
    # Keep scales as (n_rows, n_blocks) to avoid reshape corruption.
    scales_flat = torch.zeros(n_rows, n_blocks, dtype=torch.float32, device=device)
    n_flipped = 0
    for b in range(n_blocks):
        final_direction = torch.zeros(n_rows, block_size, dtype=torch.float32, device=device)
        for c in range(n_codebooks):
            if shared_codebook:
                cb_b = cb_codebooks[c][0]  # (block_size, K_c)
            else:
                cb_b = cb_codebooks[c][b]  # (block_size, K_c)
            asg_b = cb_assignments[c][b]
            final_direction = final_direction + cb_b.t()[asg_b]
        raw_block = raw_blocks[:, b, :]
        # When sign-split is active, centroids are in the first orthant.
        # Fold raw_block to match: use |raw_block| for scale computation.
        if sign_split and signs is not None:
            raw_block = raw_block * signs[:, b, :]
        dot = torch.einsum('nd,nd->n', raw_block, final_direction)
        # Flip direction where dot < 0 to ensure positive scale
        neg = dot < 0
        if neg.any():
            n_flipped += neg.sum().item()
            final_direction[neg] = -final_direction[neg]
            dot[neg] = -dot[neg]
        scale = dot / (final_direction.norm(dim=-1) ** 2 + 1e-10)
        scales_flat[:, b] = scale
    if n_flipped > 0:
        logger.info("[%s] flipped %d/%d block-rows with negative dot(raw, direction)", name, n_flipped, n_rows * n_blocks)

    # Keep assignments as list of (n_blocks, n_rows) - no reshape/permute
    assignments: list[torch.Tensor] = [a.cpu() for a in cb_assignments]

    codebooks_out = [cb.cpu() for cb in cb_codebooks]

    return CodebookResult(
        codebooks=codebooks_out,
        assignments=assignments,
        scales=scales_flat.cpu(),
        zero_mask=zero_mask.cpu(),
        n_blocks=n_blocks,
        n_codebooks=n_codebooks,
        shared_codebook=shared_codebook,
        sign_bits=signs.cpu() if signs is not None else None,
    )
