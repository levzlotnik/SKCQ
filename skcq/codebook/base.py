from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch

from skcq.codebook.cwr import CompressedWeightsRepresentation
from skcq.codebook.events import Experiment

DistanceMetric = Literal["cosine", "euclidean"]


class WeightsCodebookBase[CWR: CompressedWeightsRepresentation](Experiment, ABC):
    """Base class for all codebook quantization schemes.

    Each ``(CB, CWR)`` pair owns its own fields — the base class declares no
    attributes. The codebook instance is the *codec* (holds learned centroids
    and any shared dequantization state); the CWR is the *compressed payload*
    (per-row indices, scales, signs, etc.).

    ``fit(w)`` trains the centroids on ``w`` and returns a CWR carrying the
    per-row compressed data. ``reconstruct(cwr)`` decodes a CWR back to a
    weight tensor — the row dimension matches the CWR's row dimension, which
    may be a subset of the training weights (e.g. one expert sliced out of
    the consolidated MoE matrix).
    """

    @abstractmethod
    def fit(self, w: torch.Tensor) -> CWR:
        """Train centroids on ``w`` and return the compressed per-row data.

        Args:
            w: weight matrix ``(n_rows, in_dim)``. ``n_rows`` may be the
               consolidated count across all experts in an MoE layer.
        Returns:
            A CWR whose row dimension equals ``n_rows``.
        """

    @abstractmethod
    def reconstruct(self, cwr: CWR) -> torch.Tensor:
        """Decode a CWR into a weight tensor.

        The returned tensor has shape ``[n_rows, in_dim]`` where ``n_rows``
        is the CWR's row dimension — NOT the training row count. Passing a
        CWR carrying a subset of rows (e.g. one expert) reconstructs just
        that subset.

        No streaming / chunking — runs fully on GPU.
        """

    def quantize_centroids(self, bits: int) -> None:
        """Quantize centroids to ``int<bits>`` (symmetric per-centroid-scale).

        Override in subclasses that carry centroids. Default: no-op.
        """

    @property
    def centroids_tensor(self) -> torch.Tensor:
        """The learned centroids: ``(n_blocks, bs, K)`` or ``(1, bs, K)`` if shared."""
        raise NotImplementedError
