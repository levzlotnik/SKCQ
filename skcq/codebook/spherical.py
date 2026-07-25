from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from skcq.codebook.base import WeightsCodebookBase
from skcq.codebook.cwr import SphericalCWR
from skcq.codebook.events import (
    Experiment,
    KmeansDoneEvent,
    KmeansIterEvent,
    KmeansStartEvent,
)
from skcq.codebook.kmeans import KmeansConfig, KmeansExperiment, assign_cosine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale quantization state — lives on the codebook instance (the codec)
# ---------------------------------------------------------------------------


_FP8_MAP = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


@dataclass
class ScaleQuantState:
    """Dequantization state for quantized scales.

    For fp16/bf16/fp8: just a cast (no extra state needed).
    For int<N>: ``q_scale`` is the global abs-max / (2^(N-1) - 1).
    """

    scale_dtype: str = "bf16"
    q_scale: torch.Tensor | None = None


def quantize_spherical_scales(cwr: SphericalCWR, cb: SphericalCodebook, scale_dtype: str) -> None:
    """Quantize the CWR's scales in-place and set the codebook's ScaleQuantState."""
    scales = cwr.scales.float()
    if scale_dtype == "bf16":
        cwr.scales = scales.to(torch.bfloat16)
        cb.scale_quant = ScaleQuantState(scale_dtype="bf16")
    elif scale_dtype == "fp16":
        cwr.scales = scales.to(torch.float16)
        cb.scale_quant = ScaleQuantState(scale_dtype="fp16")
    elif scale_dtype in _FP8_MAP:
        cwr.scales = scales.to(_FP8_MAP[scale_dtype])
        cb.scale_quant = ScaleQuantState(scale_dtype=scale_dtype)
    elif scale_dtype.startswith("int"):
        bits = int(scale_dtype[3:])
        levels = 2 ** (bits - 1) - 1
        abs_max = scales.abs().max()
        if abs_max == 0:
            cb.scale_quant = ScaleQuantState(scale_dtype=scale_dtype, q_scale=torch.tensor(0.0))
            return
        q_scale = abs_max / levels
        q = torch.round(scales / q_scale).clamp(-levels, levels)
        cwr.scales = q.to(torch.int8 if bits <= 8 else torch.int16)
        cb.scale_quant = ScaleQuantState(scale_dtype=scale_dtype, q_scale=q_scale)
    else:
        raise ValueError(f"Unknown scale dtype: {scale_dtype}")


def scale_bits_per_elem(dtype: str) -> int:
    if dtype.startswith("int"):
        return int(dtype[3:])
    if dtype in _FP8_MAP:
        return 8
    if dtype in ("fp16", "bf16"):
        return 16
    raise ValueError(f"Unknown scale dtype: {dtype}")


def parse_scale_dtype(s: str) -> str:
    s = s.strip().lower()
    if s.startswith("int"):
        bits = int(s[3:])
        if bits < 2 or bits > 16:
            raise ValueError(f"int bits must be 2-16, got {bits}")
        return s
    if s in ("fp16", "bf16"):
        return s
    if s in _FP8_MAP:
        return s
    raise ValueError(f"Unknown scale dtype '{s}'")


# ---------------------------------------------------------------------------
# SphericalCodebookConfig + SphericalCodebook
# ---------------------------------------------------------------------------


@dataclass
class SphericalCodebookConfig:
    """Configuration for a spherical (cosine) codebook.

    Centroids are unit-norm directions; a per-(row, block) scale carries
    magnitude. Scales may be quantized post-fit (see ``scale_dtype``).
    """

    block_size: int
    K: int
    shared: bool = False
    max_iters: int = 100
    norm_threshold: float = 0.001
    skip_zeros: bool = True
    chunk_budget_mb: int = 2048
    device: torch.device | None = None
    max_train_samples: int = 0
    scale_dtype: str = "bf16"  # post-fit scale quantization dtype


class SphericalCodebook(WeightsCodebookBase[SphericalCWR], Experiment):
    """Spherical (cosine) codebook with PQ blocks.

    Centroids are unit-norm directions; a per-(row, block) scale carries
    magnitude. Scales may be quantized — the ``ScaleQuantState`` on this
    instance holds the dequantization info.
    """

    def __init__(self, config: SphericalCodebookConfig) -> None:
        Experiment.__init__(self)
        self.config = config
        self.directions: torch.Tensor | None = None  # (n_blocks, bs, K) or (1, bs, K)
        self.block_size = config.block_size
        self.shared = config.shared
        self.K = config.K
        self.scale_quant = ScaleQuantState(scale_dtype="bf16")

    def fit(self, w: torch.Tensor) -> SphericalCWR:
        cfg = self.config
        device = cfg.device if cfg.device is not None else w.device
        n_rows, in_dim = w.shape
        bs = cfg.block_size
        n_blocks = in_dim // bs
        cov = n_blocks * bs
        rem = in_dim - cov

        raw = w.float().to(device)
        row_norms = raw.norm(dim=-1)
        zero_mask = row_norms < cfg.norm_threshold

        if cfg.shared:
            pooled = raw[:, :cov].reshape(n_rows * n_blocks, bs)
            pooled_zero = zero_mask.tile(n_blocks)
            labels, centroids_kd = self._cluster_block(
                pooled, pooled_zero, device, name="spherical"
            )
            assignments = labels.reshape(n_rows, n_blocks).t().contiguous()
            self.directions = centroids_kd.t().contiguous().unsqueeze(0)  # (1, bs, K)
        else:
            block_labels: list[torch.Tensor] = []
            block_centroids: list[torch.Tensor] = []
            for b in range(n_blocks):
                labels, centroids_kd = self._cluster_block(
                    raw[:, b * bs : (b + 1) * bs], zero_mask, device, name=f"spherical blk={b}"
                )
                block_labels.append(labels)
                block_centroids.append(centroids_kd.t().contiguous())
            assignments = torch.stack(block_labels, dim=0)
            self.directions = torch.stack(block_centroids, dim=0)  # (n_blocks, bs, K)

        # Compute scales: dot(raw_data, assigned_direction)
        scales = torch.zeros(n_rows, n_blocks, dtype=torch.float32, device=device)
        for b in range(n_blocks):
            cb = self.directions[0] if self.shared else self.directions[b]
            d = cb.float().t()[assignments[b].to(device)]  # (n_rows, bs)
            d[zero_mask] = 0.0
            scales[:, b] = torch.einsum("nd,nd->n", raw[:, b * bs : (b + 1) * bs], d)

        remainder = raw[:, cov:].to(torch.bfloat16).cpu() if rem > 0 else None

        self.directions = self.directions.to(torch.bfloat16).cpu()
        assignments = assignments.cpu()
        scales = scales.cpu()

        return SphericalCWR(idxs=assignments, scales=scales, remainder=remainder)

    def reconstruct(self, cwr: SphericalCWR) -> torch.Tensor:
        assert self.directions is not None, "fit() not called yet"
        n_rows = cwr.idxs.shape[1]
        bs = self.block_size
        n_blocks = cwr.idxs.shape[0]
        cov = n_blocks * bs
        rem = cwr.remainder.shape[1] if cwr.remainder is not None else 0
        in_dim = cov + rem
        device = cwr.idxs.device

        recon = torch.zeros(n_rows, in_dim, dtype=torch.float32, device=device)
        dirs = self.directions.float().to(device)
        scales = self._dequant_scales(cwr.scales, device)
        for b in range(n_blocks):
            cb = dirs[0] if self.shared else dirs[b]
            d = cb.t()[cwr.idxs[b].to(device)]  # (n_rows, bs)
            recon[:, b * bs : (b + 1) * bs] = scales[:, b].unsqueeze(-1) * d
        if cwr.remainder is not None:
            recon[:, cov:] = cwr.remainder.float().to(device)
        return recon

    def _dequant_scales(self, scales: torch.Tensor, device: torch.device) -> torch.Tensor:
        sq = self.scale_quant
        s = scales.to(device)
        if sq.scale_dtype in ("bf16", "fp16"):
            return s.float()
        if sq.scale_dtype in _FP8_MAP:
            return s.to(torch.float32)
        if sq.scale_dtype.startswith("int"):
            assert sq.q_scale is not None
            return s.float() * sq.q_scale.to(device)
        return s.float()

    def _cluster_block(
        self,
        data: torch.Tensor,
        zero_mask: torch.Tensor,
        device: torch.device,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cosine k-means on one block's data.

        Returns (labels, centroids) where:
          labels: (n_rows,) int64
          centroids: (k_eff, d) float32 — unit-norm directions
        """
        cfg = self.config
        n_rows = data.shape[0]

        block_zero = data.norm(dim=-1) < cfg.norm_threshold
        if cfg.skip_zeros:
            non_zero = data[~block_zero]
        else:
            non_zero = data
            block_zero = torch.zeros(n_rows, dtype=torch.bool, device=data.device)

        if non_zero.shape[0] == 0:
            return (
                torch.zeros(n_rows, dtype=torch.long, device=data.device),
                torch.zeros(min(cfg.K, 1), data.shape[1], dtype=torch.float32, device=device),
            )

        k_eff = min(cfg.K, non_zero.shape[0])
        d = non_zero.shape[1]
        budget_bytes = cfg.chunk_budget_mb * 1024 * 1024
        chunk_size = max(1, min(budget_bytes // ((d + k_eff) * 4 * 2), non_zero.shape[0]))

        train_data = non_zero
        train_raw = non_zero
        if cfg.max_train_samples > 0 and non_zero.shape[0] > cfg.max_train_samples:
            perm = torch.randperm(non_zero.shape[0], device=non_zero.device)[
                : cfg.max_train_samples
            ]
            train_data = non_zero[perm]
            train_raw = train_data

        km = KmeansExperiment(
            KmeansConfig(
                metric="cosine",
                k=k_eff,
                max_iters=cfg.max_iters,
                device=device,
                name=name,
                chunk_size=min(chunk_size, train_data.shape[0]),
                raw_data=train_raw,
            )
        )
        self._forward_from(km, KmeansStartEvent)
        self._forward_from(km, KmeansIterEvent)
        self._forward_from(km, KmeansDoneEvent)
        centroids, _ = km.fit(train_data)

        # Re-assign ALL points
        unit = F.normalize(non_zero, dim=-1)
        labels_nz = assign_cosine(unit, centroids.t().contiguous(), chunk_size, device)
        assigned = centroids[labels_nz.to(centroids.device)].to(unit.device)
        scales_nz = torch.einsum("nd,nd->n", non_zero, assigned)

        # Flip centroids with negative dot products
        neg_mask = scales_nz < 0
        if neg_mask.any():
            neg_labels = labels_nz[neg_mask].unique().to(centroids.device)
            centroids[:, neg_labels] = -centroids[:, neg_labels]
            logger.info("[%s] flipped %d/%d centroids", name, neg_labels.shape[0], k_eff)

        labels_full = torch.zeros(n_rows, dtype=torch.long, device=data.device)
        labels_full[~block_zero] = labels_nz.to(data.device)

        return labels_full, centroids

    def quantize_centroids(self, bits: int) -> None:
        if bits >= 16 or self.directions is None:
            return
        levels = 2 ** (bits - 1) - 1
        cb = self.directions.float()
        flat = cb.reshape(-1, cb.shape[-1])
        cb_max = flat.abs().max(dim=0).values.clamp(min=1e-10)
        q_scale = cb_max / levels
        q = torch.round(flat / q_scale).clamp(-levels, levels)
        self.directions = (q * q_scale).reshape(cb.shape).to(torch.bfloat16)

    @property
    def centroids_tensor(self) -> torch.Tensor:
        assert self.directions is not None, "fit() not called yet"
        return self.directions
