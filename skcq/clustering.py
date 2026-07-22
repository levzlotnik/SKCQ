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

    # Euclidean primary carries no scale (centroids ARE the reconstruction);
    # multiplying them by the all-zero scales would zero out the output. Detect
    # once and skip the multiply in the per-chunk loop below. Cheaper and
    # more robust than threading a `primary_metric` flag through every caller.
    primary_uses_scale = bool(scales.any().item())

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
        # When the primary metric is euclidean, scales are all zero (centroids
        # ARE the reconstruction) and the scale-multiply would zero it out —
        # skip it. Detect this once by checking if scales is identically zero;
        # cheaper than threading the metric flag through the call signature and
        # correct for both build_codebook and CodebookResult.reconstruct.
        prim = torch.empty(rc, cov_0, dtype=torch.float32, device=device)
        for b in range(n_blocks_0):
            cb = cbs_dev[0][0] if shared_codebook else cbs_dev[0][b]
            d = cb.t()[assignments[0][b][r0:r1].to(device)]  # (rc, bs_0)
            sc_b = scales[r0:r1, b].float().to(device).unsqueeze(-1)
            prim[:, b * bs_0 : (b + 1) * bs_0] = sc_b * d if primary_uses_scale else d
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
