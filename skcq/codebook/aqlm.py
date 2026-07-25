from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from skcq.codebook.base import WeightsCodebookBase
from skcq.codebook.cwr import AQLMCWR, CompressedWeightsRepresentation, SphericalCWR
from skcq.codebook.events import (
    CodebookDoneEvent,
    CodebookIterEvent,
    CodebookStartEvent,
    KmeansDoneEvent,
    KmeansIterEvent,
    KmeansStartEvent,
)
from skcq.codebook.spherical import SphericalCodebook
from skcq.codebook.ssvq import SSVQCodebook

logger = logging.getLogger(__name__)


@dataclass
class AQLMCodebookConfig:
    """Composite-level config for AQLM (primary + residual chain).

    The primary and residual codebook *instances* are passed to the
    ``AQLMCodebook`` constructor, not to this config. This config holds
    only the composite-level parameters.
    """

    codebook_bits: int = 16
    norm_threshold: float = 0.001
    device: torch.device | None = None
    out_dim: int | None = None
    num_experts: int | None = None


class AQLMCodebook(WeightsCodebookBase[AQLMCWR]):
    """AQLM-style composite: primary (spherical) + residual chain (euclidean).

    Fit pipeline:
      1. Fit primary on raw weights → ``primary_cwr``
      2. Compute ``error = w − primary.reconstruct(primary_cwr)``
      3. For each residual: fit on ``error`` → ``res_cwr``; update error
      4. Quantize centroids (``codebook_bits < 16``)
      5. Rebuild full reconstruction with quantized centroids
      6. Re-fit primary scales (LS optimal, cosine primary only)

    Each codebook partitions ``in_dim`` into its OWN blocks; block sizes
    are fully independent (non-commensurate allowed). Leftover columns
    are stored raw (bf16) per codebook.
    """

    def __init__(
        self,
        primary: WeightsCodebookBase,
        residuals: list[WeightsCodebookBase],
        config: AQLMCodebookConfig | None = None,
    ) -> None:
        super().__init__()
        self.primary = primary
        self.residuals = residuals
        self.config = config or AQLMCodebookConfig()

    def fit(self, w: torch.Tensor) -> AQLMCWR:
        cfg = self.config
        device = cfg.device if cfg.device is not None else w.device
        n_rows, in_dim = w.shape

        raw = w.float().to(device)
        row_norms = raw.norm(dim=-1)
        zero_mask = row_norms < cfg.norm_threshold
        logger.info(
            "AQLM: %d/%d rows below norm threshold %s",
            zero_mask.sum().item(),
            n_rows,
            cfg.norm_threshold,
        )

        n_codebooks = 1 + len(self.residuals)

        # Determine primary metric for the start event
        primary_metric = "euclidean"
        sph_cb, _, _ = self._unwrap_spherical_primary()
        if sph_cb is not None:
            primary_metric = "cosine"

        self._emit(
            CodebookStartEvent(
                n_codebooks=n_codebooks,
                primary_block_size=getattr(self.primary, "block_size", 0),
                primary_k=getattr(self.primary, "K", 0),
                metric=primary_metric,
            )
        )

        # 1. Fit primary
        self._wire_child_events(self.primary, 0, n_codebooks)
        primary_cwr = self.primary.fit(raw)
        self._emit(CodebookDoneEvent(codebook_idx=0, n_codebooks=n_codebooks))

        # 2. Compute error and fit residuals
        recon = self.primary.reconstruct(primary_cwr).to(device)
        error = raw - recon

        res_cwrs: list[CompressedWeightsRepresentation] = []
        for i, res_cb in enumerate(self.residuals):
            self._wire_child_events(res_cb, i + 1, n_codebooks)
            res_cwr = res_cb.fit(error)
            res_cwrs.append(res_cwr)
            self._emit(CodebookDoneEvent(codebook_idx=i + 1, n_codebooks=n_codebooks))
            recon_c = res_cb.reconstruct(res_cwr).to(device)
            recon = recon + recon_c
            error = raw - recon

        # 3. Quantize centroids
        if cfg.codebook_bits < 16:
            self.primary.quantize_centroids(cfg.codebook_bits)
            for res_cb in self.residuals:
                res_cb.quantize_centroids(cfg.codebook_bits)
            logger.info("AQLM: quantized centroids to int%d", cfg.codebook_bits)

        # 4. Rebuild full reconstruction with quantized centroids
        recon = self.primary.reconstruct(primary_cwr).to(device)
        for i, res_cb in enumerate(self.residuals):
            recon = recon + res_cb.reconstruct(res_cwrs[i]).to(device)

        # 5. Scale re-fit (cosine primary only)
        sph_cb, sph_cwr, signs = self._unwrap_spherical_primary_from_cwr(primary_cwr)
        if sph_cb is not None and sph_cwr is not None:
            self._refit_primary_scale(sph_cb, sph_cwr, signs, raw, zero_mask, recon, device)

        return AQLMCWR(primary=primary_cwr, residuals=res_cwrs)

    def reconstruct(self, cwr: AQLMCWR) -> torch.Tensor:
        recon = self.primary.reconstruct(cwr.primary)
        for i, res_cb in enumerate(self.residuals):
            recon = recon + res_cb.reconstruct(cwr.residuals[i])
        return recon

    # -----------------------------------------------------------------------
    # Scale re-fit (LS optimal, cosine primary only)
    # -----------------------------------------------------------------------

    def _refit_primary_scale(
        self,
        sph_cb: SphericalCodebook,
        sph_cwr: SphericalCWR,
        signs: torch.Tensor | None,
        raw: torch.Tensor,
        zero_mask: torch.Tensor,
        recon_total: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Re-fit primary scales against the full reconstruction.

        LS optimal: ``scale = (target · pdir) / ||pdir||²`` where
        ``target = raw − (recon_total − scale_old * pdir)``.
        """
        assert sph_cb.directions is not None
        bs = sph_cb.block_size
        n_blocks = sph_cwr.idxs.shape[0]

        for b in range(n_blocks):
            cols = slice(b * bs, (b + 1) * bs)
            cb = sph_cb.directions[0] if sph_cb.shared else sph_cb.directions[b]
            d = cb.float().t()[sph_cwr.idxs[b].to(device)].to(device)
            d = d.clone()
            d[zero_mask] = 0.0
            pdir = d * signs[:, b * bs : (b + 1) * bs].to(device) if signs is not None else d
            scale_old = sph_cwr.scales[:, b].to(device)
            other_b = recon_total[:, cols] - scale_old.unsqueeze(-1) * pdir
            target = raw[:, cols] - other_b
            dot = torch.einsum("nd,nd->n", target, pdir)
            scale_new = dot / (pdir.norm(dim=-1) ** 2 + 1e-10)
            scale_new[zero_mask] = 0.0
            sph_cwr.scales[:, b] = scale_new.cpu()

    # -----------------------------------------------------------------------
    # SSVQ unwrapping helpers
    # -----------------------------------------------------------------------

    def _unwrap_spherical_primary(
        self,
    ) -> tuple[SphericalCodebook | None, SphericalCWR | None, torch.Tensor | None]:
        """Check if the primary is Spherical (possibly SSVQ-wrapped).

        Returns (spherical_cb, None, None) — CWR not available before fit.
        """
        if isinstance(self.primary, SSVQCodebook):
            inner = self.primary.inner
            if isinstance(inner, SphericalCodebook):
                return inner, None, None
        elif isinstance(self.primary, SphericalCodebook):
            return self.primary, None, None
        return None, None, None

    def _unwrap_spherical_primary_from_cwr(
        self, primary_cwr: CompressedWeightsRepresentation
    ) -> tuple[SphericalCodebook | None, SphericalCWR | None, torch.Tensor | None]:
        """Same as _unwrap_spherical_primary but also extracts the CWR + signs."""
        from skcq.codebook.cwr import SSVQCWR

        if isinstance(self.primary, SSVQCodebook):
            inner = self.primary.inner
            if isinstance(inner, SphericalCodebook) and isinstance(primary_cwr, SSVQCWR):
                return inner, primary_cwr.inner, primary_cwr.signs
        elif isinstance(self.primary, SphericalCodebook) and isinstance(primary_cwr, SphericalCWR):
            return self.primary, primary_cwr, None
        return None, None, None

    # -----------------------------------------------------------------------
    # Event wiring
    # -----------------------------------------------------------------------

    def _wire_child_events(
        self, child: WeightsCodebookBase, codebook_idx: int, n_codebooks: int
    ) -> None:
        """Forward child k-means events, wrapping KmeansIterEvent with codebook context."""
        # Forward raw events for tqdm
        self._forward_from(child, KmeansStartEvent)
        self._forward_from(child, KmeansDoneEvent)

        # Wrap KmeansIterEvent into CodebookIterEvent with codebook context
        def wrap_iter(e: KmeansIterEvent) -> None:
            self._emit(
                CodebookIterEvent(
                    codebook_idx=codebook_idx,
                    n_codebooks=n_codebooks,
                    block_idx=0,
                    n_blocks=1,
                    iter=e.iter,
                    max_iters=e.max_iters,
                    moved=e.moved,
                    n_empty=e.n_empty,
                    metric="cosine"
                    if isinstance(child, (SphericalCodebook, SSVQCodebook))
                    else "euclidean",
                )
            )

        child.on(KmeansIterEvent, wrap_iter)
        # Also forward raw KmeansIterEvent for tqdm listeners
        self._forward_from(child, KmeansIterEvent)
