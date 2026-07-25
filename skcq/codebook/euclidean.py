from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from skcq.codebook.base import WeightsCodebookBase
from skcq.codebook.cwr import EuclideanCWR
from skcq.codebook.events import (
    Experiment,
    KmeansDoneEvent,
    KmeansIterEvent,
    KmeansStartEvent,
)
from skcq.codebook.kmeans import KmeansConfig, KmeansExperiment, assign_l2

logger = logging.getLogger(__name__)


@dataclass
class EuclideanCodebookConfig:
    """Configuration for a euclidean (l2) codebook.

    Centroids carry magnitude — no separate scale.
    """

    block_size: int
    K: int
    shared: bool = False
    max_iters: int = 100
    norm_threshold: float = 0.001
    skip_zeros: bool = True
    chunk_budget_mb: int = 2048
    device: torch.device | None = None
    max_train_samples: int = 0  # 0 = no subsampling; 2**23 for shared


class EuclideanCodebook(WeightsCodebookBase[EuclideanCWR], Experiment):
    """Euclidean (l2) codebook with PQ blocks.

    Centroids carry magnitude — reconstruction is a direct lookup.
    Each codebook partitions ``in_dim`` into blocks of ``block_size``;
    leftover columns (``in_dim % block_size``) are stored raw (bf16).
    """

    def __init__(self, config: EuclideanCodebookConfig) -> None:
        Experiment.__init__(self)
        self.config = config
        self.centroids: torch.Tensor | None = None  # (n_blocks, bs, K) or (1, bs, K)
        self.block_size = config.block_size
        self.shared = config.shared
        self.K = config.K

    def fit(self, w: torch.Tensor) -> EuclideanCWR:
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
            labels = self._cluster_block(pooled, zero_mask.tile(n_blocks), device, name="euclidean")
            # labels: (n_rows * n_blocks,) → reshape to (n_blocks, n_rows)
            assignments = labels.reshape(n_rows, n_blocks).t().contiguous()
            # centroids: (bs, K) → (1, bs, K)
            self.centroids = self._last_centroids.unsqueeze(0)
        else:
            block_labels: list[torch.Tensor] = []
            block_centroids: list[torch.Tensor] = []
            for b in range(n_blocks):
                labels = self._cluster_block(
                    raw[:, b * bs : (b + 1) * bs], zero_mask, device, name=f"euclidean blk={b}"
                )
                block_labels.append(labels)
                block_centroids.append(self._last_centroids)
            assignments = torch.stack(block_labels, dim=0)  # (n_blocks, n_rows)
            self.centroids = torch.stack(block_centroids, dim=0)  # (n_blocks, bs, K)

        remainder = raw[:, cov:].to(torch.bfloat16).cpu() if rem > 0 else None

        # Move to CPU (callers reconstruct on CPU by default)
        self.centroids = self.centroids.to(torch.bfloat16).cpu()
        assignments = assignments.cpu()

        return EuclideanCWR(idxs=assignments, remainder=remainder)

    def reconstruct(self, cwr: EuclideanCWR) -> torch.Tensor:
        assert self.centroids is not None, "fit() not called yet"
        n_rows = cwr.idxs.shape[1]
        bs = self.block_size
        n_blocks = cwr.idxs.shape[0]
        cov = n_blocks * bs
        rem = cwr.remainder.shape[1] if cwr.remainder is not None else 0
        in_dim = cov + rem
        device = cwr.idxs.device

        recon = torch.zeros(n_rows, in_dim, dtype=torch.float32, device=device)
        cbs = self.centroids.float().to(device)
        for b in range(n_blocks):
            cb = cbs[0] if self.shared else cbs[b]
            recon[:, b * bs : (b + 1) * bs] = cb.t()[cwr.idxs[b].to(device)]
        if cwr.remainder is not None:
            recon[:, cov:] = cwr.remainder.float().to(device)
        return recon

    def _cluster_block(
        self,
        data: torch.Tensor,
        zero_mask: torch.Tensor,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """Run euclidean k-means on one block's data. Returns labels (n_rows,)."""
        cfg = self.config
        n_rows = data.shape[0]

        block_zero = data.norm(dim=-1) < cfg.norm_threshold
        if cfg.skip_zeros:
            non_zero = data[~block_zero]
        else:
            non_zero = data
            block_zero = torch.zeros(n_rows, dtype=torch.bool, device=data.device)

        if non_zero.shape[0] == 0:
            self._last_centroids = torch.zeros(
                data.shape[1], cfg.K, dtype=torch.float32, device=device
            )
            return torch.zeros(n_rows, dtype=torch.long, device=data.device)

        k_eff = min(cfg.K, non_zero.shape[0])
        d = non_zero.shape[1]
        budget_bytes = cfg.chunk_budget_mb * 1024 * 1024
        chunk_size = max(1, min(budget_bytes // ((d + k_eff) * 4 * 2), non_zero.shape[0]))

        train_data = non_zero
        if cfg.max_train_samples > 0 and non_zero.shape[0] > cfg.max_train_samples:
            perm = torch.randperm(non_zero.shape[0], device=non_zero.device)[
                : cfg.max_train_samples
            ]
            train_data = non_zero[perm]

        km = KmeansExperiment(
            KmeansConfig(
                metric="euclidean",
                k=k_eff,
                max_iters=cfg.max_iters,
                device=device,
                name=name,
                chunk_size=min(chunk_size, train_data.shape[0]),
            )
        )
        self._forward_from(km, KmeansStartEvent)
        self._forward_from(km, KmeansIterEvent)
        self._forward_from(km, KmeansDoneEvent)
        centroids, _ = km.fit(train_data)

        # Re-assign ALL points (not just training subset)
        labels_nz = assign_l2(non_zero, centroids.t().contiguous(), chunk_size, device)

        labels_full = torch.zeros(n_rows, dtype=torch.long, device=data.device)
        labels_full[~block_zero] = labels_nz.to(data.device)

        self._last_centroids = centroids.t().contiguous()  # (bs, K)
        return labels_full

    def quantize_centroids(self, bits: int) -> None:
        if bits >= 16 or self.centroids is None:
            return
        levels = 2 ** (bits - 1) - 1
        cb = self.centroids.float()
        flat = cb.reshape(-1, cb.shape[-1])
        cb_max = flat.abs().max(dim=0).values.clamp(min=1e-10)
        q_scale = cb_max / levels
        q = torch.round(flat / q_scale).clamp(-levels, levels)
        self.centroids = (q * q_scale).reshape(cb.shape).to(torch.bfloat16)

    @property
    def centroids_tensor(self) -> torch.Tensor:
        assert self.centroids is not None, "fit() not called yet"
        return self.centroids
