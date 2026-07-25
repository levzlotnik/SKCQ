from __future__ import annotations

import logging

import torch

from skcq.codebook.base import WeightsCodebookBase
from skcq.codebook.cwr import SSVQCWR
from skcq.codebook.events import Experiment

logger = logging.getLogger(__name__)


class SSVQCodebook(WeightsCodebookBase[SSVQCWR], Experiment):
    """Sign-split VQ combinator: wraps any inner codebook with per-element signs.

    During ``fit``, the input is folded to the first orthant (``signs = sign(w)``,
    ``w_folded = |w|``), the inner codebook clusters the folded data, and the
    signs are stored on the CWR. ``reconstruct`` applies the signs to the inner
    reconstruction: ``recon = signs ⊙ inner_recon``.

    This doubles the effective codebook resolution (all centroids are in the
    first orthant, signs recover the original quadrant). Applied per-codebook
    independently — a primary may use SSVQ while residuals don't, or vice versa.
    """

    def __init__(self, inner: WeightsCodebookBase) -> None:
        Experiment.__init__(self)
        self.inner = inner
        # Forward all events from the inner codebook
        from skcq.codebook.events import (
            KmeansDoneEvent,
            KmeansIterEvent,
            KmeansStartEvent,
        )

        self._forward_from(inner, KmeansStartEvent)
        self._forward_from(inner, KmeansIterEvent)
        self._forward_from(inner, KmeansDoneEvent)

    def fit(self, w: torch.Tensor) -> SSVQCWR:
        # Fold to first orthant
        signs = torch.sign(w)
        signs[signs == 0] = 1.0
        w_folded = w * signs  # = |w| (first orthant)

        inner_cwr = self.inner.fit(w_folded)

        # signs shape: (n_rows, in_dim) — covers the full input dim
        # But we only need signs over the covered region (cov columns)
        # The inner codebook's block structure determines cov.
        # We store the full signs and let reconstruct slice appropriately.
        signs_int8 = signs.to(torch.int8)

        return SSVQCWR(inner=inner_cwr, signs=signs_int8)

    def reconstruct(self, cwr: SSVQCWR) -> torch.Tensor:
        recon = self.inner.reconstruct(cwr.inner)
        # Apply signs over the covered region only
        # The covered region is the first cov columns, where cov = recon.shape[1] - rem
        # But signs covers the full in_dim. We slice signs[:, :cov].
        cov = recon.shape[1]
        if cwr.signs.shape[1] < cov:
            cov = cwr.signs.shape[1]
        recon[:, :cov] = recon[:, :cov] * cwr.signs[:, :cov].to(recon.device).float()
        return recon

    def quantize_centroids(self, bits: int) -> None:
        self.inner.quantize_centroids(bits)

    @property
    def centroids_tensor(self) -> torch.Tensor:
        return self.inner.centroids_tensor
