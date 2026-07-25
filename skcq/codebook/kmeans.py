from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from skcq.codebook.base import DistanceMetric
from skcq.codebook.events import (
    Experiment,
    KmeansDoneEvent,
    KmeansIterEvent,
    KmeansStartEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assignment + init helpers (ported from clustering.py)
# ---------------------------------------------------------------------------


def assign_cosine(
    data: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign each data point to its nearest centroid (cosine: max dot product).

    ``centroids`` is ``(d, k)`` (transposed for matmul). Data stays on its
    original device; chunks are moved to ``centroids.device`` for the matmul.
    """
    n = data.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=data.device)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = data[i:end].to(device)
        dists = chunk @ centroids
        labels[i:end] = dists.argmax(dim=-1).to(data.device)
    return labels


def assign_l2(
    data: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign each data point to its nearest centroid (l2: min squared distance).

    ``centroids`` is ``(d, k)`` (transposed). In-place ops to avoid keeping
    both ``dots`` and ``dists`` alive simultaneously.
    """
    n = data.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=data.device)
    cb_sq = centroids.square().sum(dim=0).unsqueeze(0)  # (1, k)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = data[i:end].to(device)
        dists = chunk @ centroids
        dists.mul_(-2).add_(cb_sq)
        labels[i:end] = dists.argmin(dim=-1).to(data.device)
    return labels


def sobol_first_orthant(k: int, d: int, device: torch.device) -> torch.Tensor:
    """Generate k unit-norm points on the first orthant via Sobol sequence."""
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=42)
    points = sobol.draw(k).to(device)
    points = points.clamp(min=1e-6)
    return F.normalize(points, dim=-1)


# ---------------------------------------------------------------------------
# KmeansConfig + KmeansExperiment
# ---------------------------------------------------------------------------


@dataclass
class KmeansConfig:
    metric: DistanceMetric
    k: int
    max_iters: int
    device: torch.device
    name: str
    chunk_size: int
    raw_data: torch.Tensor | None = None  # cosine: raw (unnormalized) for norm-weighted update


class KmeansExperiment(Experiment):
    """One k-means run — cosine (spherical) or euclidean (l2).

    ``fit(data)`` returns ``(centroids, labels)``. For cosine, ``data`` should
    be unit-normalized; ``config.raw_data`` provides raw magnitudes for
    norm-weighted centroid updates.
    """

    def __init__(self, config: KmeansConfig) -> None:
        super().__init__()
        self.config = config
        self._centroids: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        n, d = data.shape
        k_eff = min(cfg.k, n)
        raw_data = cfg.raw_data if cfg.raw_data is not None else data

        self._emit(
            KmeansStartEvent(
                name=cfg.name, metric=cfg.metric, k=k_eff, max_iters=cfg.max_iters, n_points=n
            )
        )

        if cfg.metric == "cosine":
            centroids, labels = self._fit_cosine(data, raw_data, k_eff, n, d)
        else:
            centroids, labels = self._fit_euclidean(data, k_eff, n, d)

        self._centroids = centroids
        self._labels = labels
        return centroids, labels

    def _fit_cosine(
        self, data: torch.Tensor, raw_data: torch.Tensor, k_eff: int, n: int, d: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        device = cfg.device

        unit_data = F.normalize(data, dim=-1)
        centroids = sobol_first_orthant(k_eff, d, device)  # (k, d)
        centroids_t = centroids.t().contiguous()  # (d, k)

        final_moved = 0.0
        converged = False
        it = -1
        for it in range(cfg.max_iters):
            labels = assign_cosine(unit_data, centroids_t, cfg.chunk_size, device)

            new_centroids = torch.zeros(k_eff, d, dtype=torch.float32, device=device)
            counts = torch.zeros(k_eff, device=device)
            for i in range(0, n, cfg.chunk_size):
                end = min(i + cfg.chunk_size, n)
                chunk = raw_data[i:end].to(device)
                chunk_labels = labels[i:end].to(device)
                new_centroids.index_add_(0, chunk_labels, chunk)
                counts.index_add_(0, chunk_labels, torch.ones(end - i, device=device))

            empty = counts == 0
            n_empty = empty.sum().item()
            if n_empty > 0:
                sims = torch.empty(n, device=unit_data.device)
                for i in range(0, n, cfg.chunk_size):
                    end = min(i + cfg.chunk_size, n)
                    chunk = unit_data[i:end].to(device)
                    chunk_labels = labels[i:end].to(device)
                    assigned = centroids[chunk_labels]
                    sims[i:end] = torch.einsum("nd,nd->n", chunk, assigned).to(unit_data.device)
                worst_idx = sims.argsort()[:n_empty]
                new_centroids[empty] = unit_data[worst_idx].to(device)

            new_centroids = F.normalize(new_centroids, dim=-1)

            moved = (new_centroids - centroids).norm().item()
            centroids = new_centroids
            centroids_t = centroids.t().contiguous()

            self._emit(
                KmeansIterEvent(iter=it, max_iters=cfg.max_iters, moved=moved, n_empty=n_empty)
            )

            if moved < 1e-6:
                converged = True
                final_moved = moved
                break
            final_moved = moved

        labels = assign_cosine(unit_data, centroids_t, cfg.chunk_size, device)

        self._emit(KmeansDoneEvent(iters_run=it + 1, final_moved=final_moved, converged=converged))
        return centroids, labels

    def _fit_euclidean(
        self, data: torch.Tensor, k_eff: int, n: int, d: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        device = cfg.device

        perm = torch.randperm(n, device=data.device)[:k_eff]
        centroids = data[perm].to(device).clone()  # (k, d)

        final_moved = 0.0
        converged = False
        it = -1
        for it in range(cfg.max_iters):
            labels = assign_l2(data, centroids.t().contiguous(), cfg.chunk_size, device)

            new_centroids = torch.zeros(k_eff, d, dtype=torch.float32, device=device)
            counts = torch.zeros(k_eff, device=device)
            for i in range(0, n, cfg.chunk_size):
                end = min(i + cfg.chunk_size, n)
                chunk = data[i:end].to(device)
                chunk_labels = labels[i:end].to(device)
                new_centroids.index_add_(0, chunk_labels, chunk)
                counts.index_add_(0, chunk_labels, torch.ones(end - i, device=device))

            empty = counts == 0
            n_empty = empty.sum().item()
            if n_empty > 0:
                dists = torch.empty(n, device=data.device)
                for i in range(0, n, cfg.chunk_size):
                    end = min(i + cfg.chunk_size, n)
                    chunk = data[i:end].to(device)
                    chunk_labels = labels[i:end].to(device)
                    assigned = centroids[chunk_labels]
                    dists[i:end] = ((chunk - assigned) ** 2).sum(dim=-1).to(data.device)
                worst_idx = dists.argsort(descending=True)[:n_empty]
                new_centroids[empty] = data[worst_idx].to(device)
            new_centroids[~empty] = new_centroids[~empty] / counts[~empty].unsqueeze(-1)

            moved = (new_centroids - centroids).norm().item()
            centroids = new_centroids

            self._emit(
                KmeansIterEvent(iter=it, max_iters=cfg.max_iters, moved=moved, n_empty=n_empty)
            )

            if moved < 1e-6:
                converged = True
                final_moved = moved
                break
            final_moved = moved

        labels = assign_l2(data, centroids.t().contiguous(), cfg.chunk_size, device)

        self._emit(KmeansDoneEvent(iters_run=it + 1, final_moved=final_moved, converged=converged))
        return centroids, labels

    @property
    def centroids(self) -> torch.Tensor:
        assert self._centroids is not None, "fit() not called yet"
        return self._centroids

    @property
    def labels(self) -> torch.Tensor:
        assert self._labels is not None, "fit() not called yet"
        return self._labels
