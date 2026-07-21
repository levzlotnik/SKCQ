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
    residual_k: int | list[int] | None = Field(
        default=None,
        description="K for residual codebooks (c>=1). If int, all residuals use that K. "
        "If list, residual_k[c-1] is used for codebook c. If None, same as primary K.",
    )
    residual_block_sizes: int | list[int] | None = Field(
        default=None,
        description="Block size for residual codebooks (c>=1). Fully independent of "
        "the primary block size (non-commensurate allowed); leftover columns are "
        "stored raw (bf16). If int, all residuals use that size. If list, "
        "residual_block_sizes[c-1] for codebook c. If None, same as primary.",
    )
    residual_sign_split: bool | list[bool] | None = Field(
        default=None,
        description="SSVQ sign-split for residual codebooks (c>=1). If bool, applies "
        "to all residuals. If list, residual_sign_split[c-1] for codebook c. If None, "
        "no residual sign-split. Note: build_codebook reads sign_split from its own "
        "arg (a bool|list[bool] spanning all codebooks); this field is only an "
        "informational params channel for callers that route it into that arg.",
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
    residual_k: int | list[int] | None = None
    residual_block_sizes: int | list[int] | None = None
    residual_sign_split: bool | list[bool] | None = None
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


def reconstruct_codebooks(
    n_rows: int,
    in_dim: int,
    codebooks: list[torch.Tensor],
    assignments: list[torch.Tensor],
    scales: torch.Tensor,
    block_sizes: list[int],
    shared_codebook: bool,
    sign_bits: list[torch.Tensor | None] | None,
    remainders: list[torch.Tensor | None] | None,
    device: torch.device | None = None,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Shared reconstruction: decode (n_rows, in_dim) from codebooks + remainders.

    This is THE single source of truth for reconstruction, used by both
    ``build_codebook`` (for scale re-fit / error reporting) and
    ``CodebookResult.reconstruct``.

    Scheme (SSVQ is a PER-CODEBOOK concern):
      - Primary (c=0): recon over covered_0 = signs_0 ⊙ (scale_0 ⊙ dir_0), where
        dir_0 are unit-norm direction centroids at block size bs_0. The primary
        remainder (bf16, raw) is added over [covered_0:in_dim].
      - Residual (c>=1): centroids added over covered_c (euclidean, no scale),
        optionally masked by its OWN signs_c, plus its own raw remainder over
        [covered_c:in_dim].

    ``sign_bits`` is a per-codebook list (one entry per codebook); each entry is
    either ``(n_rows, cov_c)`` (sign-split enabled for that codebook) or None.
    The remainder region carries no sign handling (already signed / raw).

    Every codebook partitions in_dim into its OWN blocks of size bs_c; block
    sizes are fully independent (non-commensurate allowed). Leftover columns
    (in_dim % bs_c) are the remainder region for that codebook.

    Memory: the only full-size (n_rows, in_dim) allocation is the output. All
    per-codebook working buffers are materialized one ROW-CHUNK at a time so peak
    VRAM stays bounded (fits low-VRAM GPUs like the 12 GB ``serval``). Set
    ``chunk_rows`` to override the auto ~128 MiB working-set size.
    """
    if device is None:
        device = codebooks[0].device
    recon = torch.zeros(n_rows, in_dim, dtype=torch.float32, device=device)
    n_codebooks = len(codebooks)

    bs_0 = block_sizes[0]
    n_blocks_0 = in_dim // bs_0
    cov_0 = n_blocks_0 * bs_0

    if chunk_rows is None:
        # ~128 MiB fp32 working set per chunk (independent of expert count).
        chunk_rows = max(1, (128 * 1024 * 1024) // (max(1, in_dim) * 4))
    chunk_rows = min(chunk_rows, n_rows)

    # Move the (small) codebooks to the device once; assignments/signs/remainders
    # are sliced per chunk to keep the working set bounded.
    cbs_dev = [cb.float().to(device) for cb in codebooks]

    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        rc = r1 - r0

        # Primary: scale ⊙ dir over covered_0 (first-orthant if sign_split), signs.
        prim = torch.empty(rc, cov_0, dtype=torch.float32, device=device)
        for b in range(n_blocks_0):
            cb = cbs_dev[0][0] if shared_codebook else cbs_dev[0][b]
            d = cb.t()[assignments[0][b][r0:r1].to(device)]  # (rc, bs_0)
            sc_b = scales[r0:r1, b].float().to(device).unsqueeze(-1)
            prim[:, b * bs_0 : (b + 1) * bs_0] = sc_b * d
        if sign_bits is not None and sign_bits[0] is not None:
            prim *= sign_bits[0][r0:r1].reshape(rc, cov_0).float().to(device)
        recon[r0:r1, :cov_0] += prim
        if remainders is not None and remainders[0] is not None:
            recon[r0:r1, cov_0:] += remainders[0][r0:r1].float().to(device)

        # Residuals: centroids (magnitude included), masked by own signs.
        for c in range(1, n_codebooks):
            bs_c = block_sizes[c]
            n_blocks_c = in_dim // bs_c
            cov_c = n_blocks_c * bs_c
            buf = torch.empty(rc, cov_c, dtype=torch.float32, device=device)
            for b in range(n_blocks_c):
                cb = cbs_dev[c][0] if shared_codebook else cbs_dev[c][b]
                buf[:, b * bs_c : (b + 1) * bs_c] = cb.t()[assignments[c][b][r0:r1].to(device)]
            if sign_bits is not None and sign_bits[c] is not None:
                sc = sign_bits[c]
                assert sc is not None
                buf *= sc[r0:r1].reshape(rc, cov_c).float().to(device)
            recon[r0:r1, :cov_c] += buf
            if remainders is not None and remainders[c] is not None:
                rem_c_t = remainders[c]
                assert rem_c_t is not None
                recon[r0:r1, cov_c:] += rem_c_t[r0:r1].float().to(device)

    return recon


@dataclass
class CodebookResult:
    """Real-error residual codebook result for one projection.

    Primary (cb0) clusters unit directions (spherical k-means, cosine) and
    stores a per-(row, primary-block) scale. Residuals (cb1+) cluster the real
    reconstruction error with euclidean k-means (magnitude included, no scale
    of their own). Each codebook partitions in_dim into its own independent
    blocks; leftover (remainder) columns are stored raw (bf16) and reconstructed
    exactly.

    SSVQ (sign-split) is a PER-CODEBOOK concern: any codebook (primary or
    residual) may fold its own input to the first orthant, cluster the absolute
    values, and store its own per-element signs over its covered region.

    Shapes:
        codebooks:   list of (n_blocks_c, bs_c, K_c) — one per codebook.
                     When shared_codebook=True, shape is (1, bs_c, K_c).
        assignments: list of (n_blocks_c, n_rows) int64 — one per codebook
        scales:      (n_rows, n_blocks_0) — primary scale (re-fit)
        zero_mask:   (n_rows,) — bool, full-row near-zero flag (metadata)
        sign_bits:   per-codebook list; entry c is (n_rows, cov_c) if codebook c
                     used sign-split, else None. Whole list is None if no
                     codebook used sign-split.
        remainders:  list of (n_rows, rem_c) bf16 or None — one per codebook
        block_sizes: list of bs_c — per-codebook block size
    """

    codebooks: list[torch.Tensor]
    assignments: list[torch.Tensor]
    scales: torch.Tensor
    zero_mask: torch.Tensor
    n_blocks: int
    n_codebooks: int
    shared_codebook: bool = False
    # per-codebook signs: entry c is (n_rows, cov_c) or None; whole list may be None
    sign_bits: list[torch.Tensor | None] | None = None
    residual_block_sizes: list[int] | None = None  # bs for c>=1 (None/empty = same as primary)
    remainders: list[torch.Tensor | None] | None = None  # per-codebook raw remainder (bf16)
    block_sizes: list[int] | None = None  # per-codebook block size
    num_experts: int | None = None  # for deriving out_dim = n_rows // num_experts

    def block_size(self) -> int:
        """Primary sub-vector block size (bs_0)."""
        if self.block_sizes:
            return self.block_sizes[0]
        return self.codebooks[0].shape[1]

    def bs_per_codebook(self) -> list[int]:
        """Block size for each codebook: [primary_bs, residual_bs_1, ...]."""
        if self.block_sizes:
            return list(self.block_sizes)
        bs_p = self.block_size()
        if not self.residual_block_sizes:
            return [bs_p] * self.n_codebooks
        return [bs_p] + list(self.residual_block_sizes)

    def in_dim(self) -> int:
        """Full input dimension = covered_0 + primary remainder."""
        bs_0 = self.block_size()
        cov_0 = self.n_blocks * bs_0
        rem_0 = (
            self.remainders[0].shape[1]
            if (self.remainders and self.remainders[0] is not None)
            else 0
        )
        return cov_0 + rem_0

    def reconstruct(self) -> torch.Tensor:
        """Decode the quantized weight matrix (n_rows, in_dim) via the shared helper."""
        return reconstruct_codebooks(
            n_rows=self.scales.shape[0],
            in_dim=self.in_dim(),
            codebooks=self.codebooks,
            assignments=self.assignments,
            scales=self.scales,
            block_sizes=self.bs_per_codebook(),
            shared_codebook=self.shared_codebook,
            sign_bits=self.sign_bits,
            remainders=self.remainders,
        )


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
    cb = centroids.t() if centroids.shape[0] == data.shape[1] else centroids
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
    for the matmul, results moved back. In-place ops to avoid keeping both
    `dots` and `dists` alive simultaneously.
    """
    cb = centroids.t() if centroids.shape[0] == data.shape[1] else centroids
    n = data.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=data.device)
    cb_sq = cb.square().sum(dim=-1).unsqueeze(0)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = data[i:end].to(cb.device)
        # ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2
        # ||x||^2 is constant across centroids, drop it for argmin.
        # In-place: reuse the dots buffer to avoid 2x VRAM.
        dists = chunk @ cb.t()
        dists.mul_(-2).add_(cb_sq)
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
    """Standard euclidean (l2) k-means with random data-point init and tqdm.

    Assignment: argmin ||x_i - centroid_c||^2
    Update:     centroid_c = mean(x_i for i in c)

    Init: k random data points (forgy init). Unlike Sobol in [-1,1]^d,
    this adapts to the data's actual scale and location — critical for
    residual codebooks where data lives in a tiny region near the origin.

    Args:
        data: (n, d) — raw data on CPU
    Returns:
        centroids: (K, d) — on device
        labels: (n,) — on CPU
    """
    n, d = data.shape
    k_eff = min(k, n)

    # Forgy init: k random data points — adapts to data scale/location
    logger.info("[%s] Forgy init (k=%d, d=%d, euclidean)...", name, k_eff, d)
    perm = torch.randperm(n, device=data.device)[:k_eff]
    centroids = data[perm].to(device).clone()  # (k, d)

    pbar = tqdm(range(max_iters), desc=name, leave=True)
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

        # Handle empty clusters: re-init from worst-fit data points
        # (points furthest from their assigned centroid)
        empty = counts == 0
        n_empty = empty.sum().item()
        if n_empty > 0:
            # Compute distance[i] = ||data[i] - centroids[labels[i]]||^2
            dists = torch.empty(n, device=data.device)
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk = data[i:end].to(device)
                chunk_labels = labels[i:end].to(device)
                assigned = centroids[chunk_labels]  # (chunk, d)
                dists[i:end] = ((chunk - assigned) ** 2).sum(dim=-1).to(data.device)
            # n_empty worst-fit points → re-init from those
            worst_idx = dists.argsort(descending=True)[:n_empty]
            new_centroids[empty] = data[worst_idx].to(device)
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

    pbar = tqdm(range(max_iters), desc=name, leave=True)
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

        # Handle empty clusters: re-init from worst-fit data points
        # (points with lowest cosine sim to their assigned centroid)
        empty = counts == 0
        n_empty = empty.sum().item()
        if n_empty > 0:
            # Compute sim[i] = dot(unit_data[i], centroids[labels[i]])
            sims = torch.empty(n, device=unit_data.device)
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk = unit_data[i:end].to(device)
                chunk_labels = labels[i:end].to(device)
                assigned = centroids[chunk_labels]  # (chunk, d)
                sims[i:end] = torch.einsum("nd,nd->n", chunk, assigned).to(unit_data.device)
            # n_empty worst-fit points → re-init from those directions
            worst_idx = sims.argsort()[:n_empty]
            new_centroids[empty] = unit_data[worst_idx].to(device)

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
    d = non_zero.shape[1]
    budget_bytes = chunk_budget_mb * 1024 * 1024
    # Each chunk row needs: d*4 bytes (chunk data) + k_eff*4 bytes (matmul output)
    # Plus cb_sq (k_eff*4) and dists (k_eff*4) — but those are small vs chunk.
    # Use 2x safety factor for temporaries.
    bytes_per_row = (d + k_eff) * 4 * 2
    chunk_size = max(1, budget_bytes // bytes_per_row)
    chunk_size = min(chunk_size, non_zero.shape[0])

    # Sub-sample for k-means training if dataset is too large (torch.multinomial
    # in k-means++ init is limited to 2^24 categories)
    max_multinomial = (2**24) - 1
    train_data = non_zero
    train_raw = raw_data if raw_data is not None else non_zero
    if max_train_samples > 0 and non_zero.shape[0] > max_train_samples:
        perm = torch.randperm(non_zero.shape[0], device=non_zero.device)[:max_train_samples]
        train_data = non_zero[perm]
        train_raw = train_raw[perm]
        logger.info(
            "[%s] sub-sampled %d -> %d for k-means training",
            name,
            non_zero.shape[0],
            train_data.shape[0],
        )
    elif non_zero.shape[0] > max_multinomial:
        perm = torch.randperm(non_zero.shape[0], device=non_zero.device)[:max_multinomial]
        train_data = non_zero[perm]
        train_raw = train_raw[perm]
        logger.info(
            "[%s] sub-sampled %d -> %d for k-means training (multinomial limit)",
            name,
            non_zero.shape[0],
            train_data.shape[0],
        )

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
        scales_nz = torch.einsum("nd,nd->n", non_zero, assigned_centroids)

        cos_sim = torch.einsum("nd,nd->n", unit, assigned_centroids)
        logger.info(
            "[%s] training cos(raw_unit, centroid): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            name,
            cos_sim.mean().item(),
            cos_sim.std().item(),
            cos_sim.min().item(),
            cos_sim.max().item(),
        )

        # Flip centroids with negative dot products to ensure positive scales
        neg_mask = scales_nz < 0
        if neg_mask.any():
            neg_labels = labels_nz[neg_mask].unique().to(centroids_kbd.device)
            centroids_kbd[:, neg_labels] = -centroids_kbd[:, neg_labels]
            assigned_centroids = centroids_kbd[labels_nz.to(centroids_kbd.device)].to(unit.device)
            scales_nz = torch.einsum("nd,nd->n", non_zero, assigned_centroids)
            logger.info(
                "[%s] flipped %d/%d centroids with negative dot products",
                name,
                neg_mask.sum().item(),
                non_zero.shape[0],
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
        labels_nz = _assign_to_centroids_l2(
            non_zero,
            centroids_kbd.t().contiguous(),
            chunk_size,
            device,
        )
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


def _do_cluster(
    data: torch.Tensor,
    raw_data: torch.Tensor,
    k_c: int,
    metric: DistanceMetric,
    shared: bool,
    params: CodebookParams,
    device: torch.device,
    name: str,
    skip_zeros: bool,
    first_orthant: bool,
    cache: object | None,
    cache_key: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster one codebook's block-partitioned data.

    Args:
        data: (n_rows, n_blocks_c, bs_c) — data used for cosine assignment /
            euclidean clustering.
        raw_data: (n_rows, n_blocks_c, bs_c) — raw magnitudes for norm-weighted
            centroid update (cosine). Same as data for euclidean.

    Returns:
        codebook:   (n_blocks_c, bs_c, K) or (1, bs_c, K) if shared
        assignments:(n_blocks_c, n_rows) int64
        assigned:   (n_rows, n_blocks_c, bs_c) — the gathered centroids per row
    """
    n_rows, n_blocks_c, bs_c = data.shape

    if shared:
        pooled = data.reshape(n_rows * n_blocks_c, bs_c)
        pooled_raw = raw_data.reshape(n_rows * n_blocks_c, bs_c)

        cached_codebook = None
        if cache is not None and cache_key is not None:
            cached_codebook = cache.get(cache_key)  # type: ignore[attr-defined]

        if cached_codebook is not None:
            # Cache hit: re-run assignment only (much cheaper than full k-means).
            codebook = cached_codebook.to(device)
            _budget_bytes = params.chunk_budget_mb * 1024 * 1024
            _k_eff = max(1, codebook.shape[-1])
            _bytes_per_row = (bs_c + _k_eff) * 4 * 2
            _chunk = max(1, _budget_bytes // _bytes_per_row)
            if metric == "cosine":
                labels = _assign_to_centroids(pooled, codebook, _chunk, device)
            else:
                labels = _assign_to_centroids_l2(pooled, codebook, _chunk, device)
            logger.info("[%s] cache HIT (skipped k-means)", name)
        else:
            res = _cluster_block(
                pooled,
                k=k_c,
                max_iters=params.max_iters,
                norm_threshold=params.norm_threshold,
                skip_zeros=skip_zeros,
                device=device,
                name=name,
                distance_metric=metric,
                chunk_budget_mb=params.chunk_budget_mb,
                max_train_samples=2**23,
                first_orthant=first_orthant,
                raw_data=pooled_raw,
            )
            codebook = res.codebook
            labels = res.labels
            if cache is not None and cache_key is not None:
                cache.put(cache_key, codebook)  # type: ignore[attr-defined]
                logger.info("[%s] cached primary codebook", name)

        cb = codebook.unsqueeze(0)  # (1, bs_c, K)
        asg = labels.reshape(n_rows, n_blocks_c).t().contiguous()  # (n_blocks_c, n_rows)
        assigned = codebook.t()[labels.to(codebook.device)].reshape(n_rows, n_blocks_c, bs_c)
        return cb, asg, assigned.to(device)

    block_codebooks: list[torch.Tensor] = []
    block_assigns: list[torch.Tensor] = []
    assigned = torch.zeros(n_rows, n_blocks_c, bs_c, dtype=torch.float32, device=device)
    for b in range(n_blocks_c):
        res = _cluster_block(
            data[:, b, :],
            k=k_c,
            max_iters=params.max_iters,
            norm_threshold=params.norm_threshold,
            skip_zeros=skip_zeros,
            device=device,
            name=f"{name} blk={b}/{n_blocks_c}",
            distance_metric=metric,
            chunk_budget_mb=params.chunk_budget_mb,
            first_orthant=first_orthant,
            raw_data=raw_data[:, b, :],
        )
        block_codebooks.append(res.codebook)
        block_assigns.append(res.labels)
        assigned[:, b, :] = res.codebook.t()[res.labels.to(res.codebook.device)].to(device)
    cb = torch.stack(block_codebooks, dim=0)  # (n_blocks_c, bs_c, K)
    asg = torch.stack(block_assigns, dim=0)  # (n_blocks_c, n_rows)
    return cb, asg, assigned


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
    sign_split: bool | list[bool] = False,
    residual_block_sizes: int | list[int] | None = None,
    codebook_bits: int = 16,
    primary_codebook_cache: object | None = None,
    cache_key_str: str | None = None,
    primary_block_size: int | None = None,
) -> CodebookResult:
    """Build a real-error residual PQ codebook for one projection.

    Primary (cb0) clusters unit directions with spherical k-means (cosine) and
    stores a per-(row, primary-block) scale. Residuals (cb1+) cluster the REAL
    reconstruction error ``E_c = W - sum_{c'<c} recon_{c'}`` with euclidean
    k-means over the whole in_dim (magnitude included, no scale of their own).

    Every codebook partitions in_dim into its OWN block size ``bs_c``; block
    sizes are fully independent (non-commensurate allowed). Leftover columns
    (``in_dim % bs_c``) are the remainder region — stored raw (bf16) and
    reconstructed exactly.

    A single final re-fit of the primary block-wise scale is done against the
    full cumulative reconstruction.

    Args:
        rows: (num_experts * out_dim, in_dim) — weight rows, expert-major
        k: base codebook size (K_0); K_c = residual_k for c>=1 (or K if residual_k is None)
        n_blocks: number of primary sub-blocks — only used when
            ``primary_block_size`` is None (then bs_0 = in_dim // n_blocks,
            requires exact division i.e. rem_0 = 0).
        primary_block_size: if set, the primary block size bs_0 (n_blocks_0 =
            in_dim // bs_0, may leave a remainder). Independent of ``n_blocks``.
        n_codebooks: number of codebooks (1 = no residual, 2+ = primary + residuals)
        num_experts, out_dim: for reshaping assignments/scales
        shared_codebook: if True, pool all blocks' sub-vectors into a single
            shared codebook (one k-means for all blocks).
    """
    if device is None:
        device = rows.device

    n_rows, in_dim = rows.shape

    # Primary block size: explicit primary_block_size wins; else in_dim // n_blocks.
    if primary_block_size is not None:
        bs_0 = primary_block_size
    else:
        if n_blocks <= 0:
            raise ValueError(f"n_blocks={n_blocks} must be positive")
        bs_0 = in_dim // n_blocks
    if bs_0 <= 0 or bs_0 > in_dim:
        raise ValueError(f"primary block_size={bs_0} invalid for in_dim={in_dim}")
    n_blocks_0 = in_dim // bs_0
    cov_0 = n_blocks_0 * bs_0
    rem_0 = in_dim - cov_0

    # Residual block sizes -> bs_per_codebook = [bs_0, residual_bs_1, ...]
    rbs_list_raw = params.residual_block_sizes
    if rbs_list_raw is None:
        rbs_list = [bs_0] * (n_codebooks - 1)
    elif isinstance(rbs_list_raw, int):
        rbs_list = [rbs_list_raw] * (n_codebooks - 1)
    else:  # list
        if len(rbs_list_raw) < n_codebooks - 1:
            raise ValueError(
                f"residual_block_sizes has {len(rbs_list_raw)} values, need {n_codebooks - 1}"
            )
        rbs_list = list(rbs_list_raw[: n_codebooks - 1])
    for i, rbs in enumerate(rbs_list):
        if rbs <= 0:
            raise ValueError(f"residual_block_sizes[{i}]={rbs} must be positive")
    bs_per_codebook = [bs_0] + rbs_list

    raw = rows.float().clone()
    row_norms = raw.norm(dim=-1)
    zero_mask = row_norms < params.norm_threshold
    logger.info(
        "[%s] %d/%d full rows below norm threshold %s",
        name,
        zero_mask.sum().item(),
        n_rows,
        params.norm_threshold,
    )

    # K per codebook: c=0 uses primary k, c>=1 uses residual_k
    rk = params.residual_k
    if rk is None:
        k_per_codebook = [k] * n_codebooks
    elif isinstance(rk, int):
        k_per_codebook = [k if c == 0 else rk for c in range(n_codebooks)]
    else:  # list[int]
        if len(rk) < n_codebooks - 1:
            raise ValueError(f"residual_k list has {len(rk)} values, need {n_codebooks - 1}")
        k_per_codebook = [k if c == 0 else rk[c - 1] for c in range(n_codebooks)]

    cb_codebooks: list[torch.Tensor] = []
    cb_assignments: list[torch.Tensor] = []
    remainders_list: list[torch.Tensor | None] = [None] * n_codebooks
    scales_flat = torch.zeros(n_rows, n_blocks_0, dtype=torch.float32, device=device)

    # Normalize sign_split into a per-codebook list. A bare bool is interpreted
    # as PRIMARY-ONLY (back-compat): [bool, False, False, ...]. A list is used
    # as-is (must have length == n_codebooks).
    if isinstance(sign_split, bool):
        sign_split_list: list[bool] = [sign_split] + [False] * (n_codebooks - 1)
    else:
        if len(sign_split) != n_codebooks:
            raise ValueError(f"sign_split list has {len(sign_split)} values, need {n_codebooks}")
        sign_split_list = list(sign_split)
    signs_list: list[torch.Tensor | None] = [None] * n_codebooks

    recon_total = torch.zeros(n_rows, in_dim, dtype=torch.float32, device=device)
    error = raw.clone()  # E_0 = W

    for c in range(n_codebooks):
        k_c = k_per_codebook[c]
        metric: DistanceMetric = distance_metric if c == 0 else "euclidean"
        bs_c = bs_per_codebook[c]
        n_blocks_c = in_dim // bs_c
        cov_c = n_blocks_c * bs_c
        rem_c = in_dim - cov_c
        ss_c = sign_split_list[c]

        if c == 0:
            # Primary: cluster unit directions on the covered region of W.
            data = raw[:, :cov_0].reshape(n_rows, n_blocks_0, bs_0)
        else:
            # Residual: cluster the REAL error over the covered region (euclidean).
            data = error[:, :cov_c].reshape(n_rows, n_blocks_c, bs_c)

        if ss_c:
            # SSVQ: fold this codebook's own input to the first orthant, cluster
            # the absolute values, and store its own per-element signs.
            signs_c = torch.sign(data)
            signs_c[signs_c == 0] = 1.0
            data = data * signs_c
            signs_list[c] = signs_c.reshape(n_rows, cov_c)
            logger.info("[%s] cb=%d sign-split: clustering on first orthant", name, c)
        raw_for_update = data

        cb, asg, assigned = _do_cluster(
            data,
            raw_for_update,
            k_c=k_c,
            metric=metric,
            shared=shared_codebook,
            params=params,
            device=device,
            name=f"{name} cb={c}/{n_codebooks}",
            skip_zeros=params.skip_zeros if c == 0 else False,
            first_orthant=ss_c,
            cache=primary_codebook_cache if c == 0 else None,
            cache_key=cache_key_str if c == 0 else None,
        )
        # Zero rows contribute nothing from the codebooks.
        assigned = assigned.clone()
        assigned[zero_mask] = 0.0
        cb_codebooks.append(cb)
        cb_assignments.append(asg)

        if c == 0:
            # scale_0 = dot(folded_block, unit_dir) per (row, block)
            dot = torch.einsum("nbd,nbd->nb", data, assigned)  # (n_rows, n_blocks_0)
            dot[zero_mask] = 0.0
            scales_flat = dot
            prim = (scales_flat.unsqueeze(-1) * assigned).reshape(n_rows, cov_0)
            if signs_list[0] is not None:
                prim = prim * signs_list[0]
            recon_total[:, :cov_0] += prim
            if rem_0 > 0:
                remainders_list[0] = raw[:, cov_0:].clone()
                recon_total[:, cov_0:] += remainders_list[0]
        else:
            res = assigned.reshape(n_rows, cov_c)
            if signs_list[c] is not None:
                res = res * signs_list[c]
            recon_total[:, :cov_c] += res
            if rem_c > 0:
                remainders_list[c] = error[:, cov_c:].clone()
                recon_total[:, cov_c:] += remainders_list[c]

        error = raw - recon_total

    # Quantize codebook centroids (per-centroid symmetric scale). The residual
    # chain above used unquantized centroids; we rebuild recon_total from the
    # QUANTIZED centroids before the scale re-fit.
    if codebook_bits < 16:
        levels = 2 ** (codebook_bits - 1) - 1
        for c in range(n_codebooks):
            cb = cb_codebooks[c]  # (n_blocks_or_1, bs, K_c)
            flat = cb.reshape(-1, cb.shape[-1])  # (n_blocks_or_1 * bs, K_c)
            cb_max = flat.abs().max(dim=0).values.clamp(min=1e-10)  # (K_c,)
            q_scale = cb_max / levels
            q = torch.round(flat / q_scale).clamp(-levels, levels)
            cb_codebooks[c] = (q * q_scale).reshape(cb.shape).to(cb.dtype)
            logger.info(
                "[%s] codebook[%d] quantized to int%d (per-centroid scale)",
                name,
                c,
                codebook_bits,
            )

    # Rebuild the full reconstruction from the (possibly quantized) centroids +
    # remainders via the shared helper (identical to CodebookResult.reconstruct).
    # Free the residual/error working buffer first so the reconstruction (which
    # needs a full (n_rows, in_dim) output) has headroom on low-VRAM GPUs.
    del error
    if device.type == "cuda":
        torch.cuda.empty_cache()
    recon_total = reconstruct_codebooks(
        n_rows=n_rows,
        in_dim=in_dim,
        codebooks=cb_codebooks,
        assignments=cb_assignments,
        scales=scales_flat,
        block_sizes=bs_per_codebook,
        shared_codebook=shared_codebook,
        sign_bits=signs_list,
        remainders=remainders_list,
        device=device,
    )

    # Final single re-fit of the primary block-wise scale against the full
    # cumulative reconstruction (least-squares optimum given everything else):
    #   other_b   = recon_total[:, block_b] - scale_old_b ⊙ pdir_b
    #   scale_new = dot(W_b - other_b, pdir_b) / (||pdir_b||^2 + eps)
    # where pdir_b is the (signed) primary direction. We store the SIGNED scale
    # (equivalent to the "flip dir when dot<0" trick, but consistent with the
    # shared unflipped-centroid helper).
    for b in range(n_blocks_0):
        cols = slice(b * bs_0, (b + 1) * bs_0)
        cb0 = cb_codebooks[0][0] if shared_codebook else cb_codebooks[0][b]
        d = cb0.float().t()[cb_assignments[0][b].to(cb0.device)].to(device)  # (n_rows, bs_0)
        d = d.clone()
        d[zero_mask] = 0.0
        if signs_list[0] is not None:
            s0 = signs_list[0]
            assert s0 is not None
            pdir = d * s0[:, b * bs_0 : (b + 1) * bs_0]
        else:
            pdir = d
        scale_old = scales_flat[:, b]
        other_b = recon_total[:, cols] - scale_old.unsqueeze(-1) * pdir
        target = raw[:, cols] - other_b
        dot = torch.einsum("nd,nd->n", target, pdir)
        scale_new = dot / (pdir.norm(dim=-1) ** 2 + 1e-10)
        scale_new[zero_mask] = 0.0
        scales_flat[:, b] = scale_new

    assignments_out: list[torch.Tensor] = [a.cpu() for a in cb_assignments]
    codebooks_out = [cb.cpu() for cb in cb_codebooks]
    remainders_list_out: list[torch.Tensor | None] = [
        (r.to(torch.bfloat16).cpu() if r is not None else None) for r in remainders_list
    ]
    remainders_out: list[torch.Tensor | None] | None = (
        None if all(r is None for r in remainders_list_out) else remainders_list_out
    )

    # Store residual_block_sizes only if any differ from primary (back-compat).
    rbs_out = [bs for bs in bs_per_codebook[1:] if bs != bs_0]
    signs_out_list: list[torch.Tensor | None] = [
        (s.cpu() if s is not None else None) for s in signs_list
    ]
    signs_out: list[torch.Tensor | None] | None = (
        None if all(s is None for s in signs_out_list) else signs_out_list
    )
    return CodebookResult(
        codebooks=codebooks_out,
        assignments=assignments_out,
        scales=scales_flat.cpu(),
        zero_mask=zero_mask.cpu(),
        n_blocks=n_blocks_0,
        n_codebooks=n_codebooks,
        shared_codebook=shared_codebook,
        sign_bits=signs_out,
        residual_block_sizes=rbs_out if rbs_out else None,
        remainders=remainders_out,
        block_sizes=list(bs_per_codebook),
        num_experts=num_experts,
    )
