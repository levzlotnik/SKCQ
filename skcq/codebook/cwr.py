from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Base CWR — all compressed-weight representations inherit from this.
# Pure dataclass, no methods. Each (CB, CWR) pair decides its own fields.
# ---------------------------------------------------------------------------


@dataclass
class CompressedWeightsRepresentation:
    """Per-row compressed data produced by ``WeightsCodebookBase.fit``.

    The row dimension (``n_rows``) is whatever ``fit`` received — a subset of
    the full weight matrix (e.g. one expert sliced out of the consolidated MoE
    weights the codebook was trained on). ``reconstruct`` returns a tensor
    shaped ``[n_rows, in_dim]`` matching this CWR's row count, NOT the
    training row count.
    """


@dataclass
class EuclideanCWR(CompressedWeightsRepresentation):
    """Compressed data for a euclidean (l2) codebook.

    Centroids carry magnitude — no separate scale needed.
    """

    idxs: torch.Tensor  # (n_blocks, n_rows) int64 — indices into the codebook
    remainder: torch.Tensor | None  # (n_rows, rem) bf16 — raw uncovered columns, or None


@dataclass
class SphericalCWR(CompressedWeightsRepresentation):
    """Compressed data for a spherical (cosine) codebook.

    Centroids are unit-norm directions; a per-(row, block) scale carries magnitude.
    The ``scales`` tensor may be quantized (int8, fp8, etc.) — the codebook
    instance holds the dequantization state (e.g. the global abs-max for int8).
    """

    idxs: torch.Tensor  # (n_blocks, n_rows) int64
    scales: torch.Tensor  # (n_rows, n_blocks) — bf16, int8, fp8, etc.
    remainder: torch.Tensor | None  # (n_rows, rem) bf16, or None


@dataclass
class SSVQCWR[CWR: CompressedWeightsRepresentation](CompressedWeightsRepresentation):
    """Combinator: wraps any inner CWR with per-element signs (±1).

    The signs tensor covers the codebook's covered region (``cov`` columns):
    ``recon = signs ⊙ inner_recon``. Applied per-codebook independently.
    """

    inner: CWR
    signs: torch.Tensor  # (n_rows, cov) int8 — ±1


@dataclass
class AQLMCWR(CompressedWeightsRepresentation):
    """Composite: primary + residual chain (AQLM-style).

    ``primary`` is typically a SphericalCWR (or SSVQCWR wrapping one).
    ``residuals`` is a heterogeneous list — each can be any CWR type
    (EuclideanCWR, SSVQCWR, etc.), allowing independent SSVQ per residual.
    """

    primary: CompressedWeightsRepresentation
    residuals: list[CompressedWeightsRepresentation]
