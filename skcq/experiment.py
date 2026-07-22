"""Experiment classes for k-means codebook quantization.

Provides a ``.fit()``-style API with typed event callbacks for progress monitoring.
Three levels of composition:

  ``KmeansExperiment``   — one k-means run (cosine or euclidean)
  ``CodebookExperiment`` — full codebook build (primary + residuals, cache, scale re-fit)

Events flow up the composition chain::

    KmeansExperiment emits KmeansIterEvent
      → CodebookExperiment wraps it as CodebookIterEvent (adds c/b context)

Listeners subscribe via ``experiment.on(EventType, handler)`` where ``handler``
receives a single typed event object. Type-safe; mypy-checked.

Usage (standalone)::

    exp = CodebookExperiment(config)
    exp.on(KmeansIterEvent, TqdmListener())
    result = exp.fit(rows)

Usage (VQ sweep worker — wire listener instead of tqdm, see skcq.vq.experiment).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.nn.functional as F
from tqdm import tqdm

from skcq.clustering import (
    CodebookParams,
    CodebookResult,
    DistanceMetric,
    _assign_to_centroids,
    _assign_to_centroids_l2,
    _sobol_first_orthant,
    reconstruct_codebooks,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Event types (typed dataclasses — listeners receive a single event object)
# ---------------------------------------------------------------------------


# K-means level (emitted by KmeansExperiment)
@dataclass
class KmeansStartEvent:
    name: str
    metric: DistanceMetric
    k: int
    max_iters: int
    n_points: int


@dataclass
class KmeansIterEvent:
    iter: int
    max_iters: int
    moved: float
    n_empty: int


@dataclass
class KmeansDoneEvent:
    iters_run: int
    final_moved: float
    converged: bool


# Codebook level (emitted by CodebookExperiment, wrapping child events with context)
@dataclass
class CodebookStartEvent:
    n_codebooks: int
    primary_block_size: int
    primary_k: int  # noqa: N815
    metric: DistanceMetric


@dataclass
class CodebookIterEvent:
    codebook_idx: int
    n_codebooks: int
    block_idx: int
    n_blocks: int
    iter: int
    max_iters: int
    moved: float
    n_empty: int
    metric: DistanceMetric


@dataclass
class CodebookDoneEvent:
    codebook_idx: int
    n_codebooks: int


# ---------------------------------------------------------------------------
# Experiment base class
# ---------------------------------------------------------------------------


class Experiment:
    """Base class — ``on(EventType, callback)`` + internal ``_emit(event)``.

    Subscription is by event type: ``exp.on(KmeansIterEvent, handler)``.
    The handler receives a single typed event object. Multiple handlers can
    subscribe to the same event type; all are called in subscription order.
    """

    def __init__(self) -> None:
        self._listeners: dict[type, list[Callable[..., None]]] = defaultdict(list)

    def on(self, event_type: type[T], callback: Callable[[T], None]) -> None:
        """Subscribe to events of ``event_type``. ``callback`` receives the event."""
        self._listeners[event_type].append(callback)

    def _emit(self, event: object) -> None:
        """Notify all subscribers of the event's type."""
        for cb in self._listeners.get(type(event), []):
            cb(event)

    def _forward_from(self, child: Experiment, event_type: type[T]) -> None:
        """Forward events of ``event_type`` from ``child`` to self's listeners."""

        def handler(e: T) -> None:
            self._emit(e)

        child.on(event_type, handler)


# ---------------------------------------------------------------------------
# BlockClusterResult (internal to CodebookExperiment — moved from clustering.py)
# ---------------------------------------------------------------------------


@dataclass
class BlockClusterResult:
    """Result of clustering one block's sub-vectors."""

    codebook: torch.Tensor  # (bs, K) — BMM-ready (transposed)
    labels: torch.Tensor  # (n_rows,) — full-size, 0 for zero rows
    scales: torch.Tensor  # (n_rows,) — 0 for zero rows and euclidean


# ---------------------------------------------------------------------------
# KmeansExperiment — one k-means run (replaces _euclidean_kmeans + _norm_weighted_spherical_kmeans)
# ---------------------------------------------------------------------------


@dataclass
class KmeansConfig:
    """Configuration for one k-means run."""

    metric: DistanceMetric
    k: int
    max_iters: int
    device: torch.device
    name: str
    chunk_size: int
    first_orthant: bool = False
    raw_data: torch.Tensor | None = None  # cosine: raw (unnormalized) for norm-weighted update


class KmeansExperiment(Experiment):
    """One k-means run — cosine (spherical) or euclidean (l2).

    ``fit(data)`` returns ``(centroids, labels)`` — same as the original
    ``_norm_weighted_spherical_kmeans`` / ``_euclidean_kmeans`` functions.
    tqdm progress bars are replaced by ``KmeansIterEvent`` emissions.
    """

    def __init__(self, config: KmeansConfig) -> None:
        super().__init__()
        self.config = config
        self._centroids: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run k-means on ``data``. Returns (centroids, labels).

        For cosine: ``data`` should be unit-normalized (assignment uses cosine
        sim); ``config.raw_data`` provides raw magnitudes for norm-weighted
        centroid updates. For euclidean: ``data`` is raw; ``raw_data`` unused.
        """
        cfg = self.config
        n, d = data.shape
        k_eff = min(cfg.k, n)
        raw_data = cfg.raw_data if cfg.raw_data is not None else data

        self._emit(
            KmeansStartEvent(
                name=cfg.name,
                metric=cfg.metric,
                k=k_eff,
                max_iters=cfg.max_iters,
                n_points=n,
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
        """Body of _norm_weighted_spherical_kmeans."""
        cfg = self.config
        device = cfg.device

        unit_data = F.normalize(data, dim=-1)

        # Deterministic Sobol init — space-filling, O(k), no data dependency
        logger.info(
            "[%s] Sobol init (k=%d, d=%d, first_orthant=%s)...",
            cfg.name,
            k_eff,
            d,
            cfg.first_orthant,
        )
        centroids = _sobol_first_orthant(k_eff, d, device)  # (k, d)
        centroids_t = centroids.t().contiguous()  # (d, k)

        final_moved = 0.0
        converged = False
        it = -1
        for it in range(cfg.max_iters):
            # Assign: argmax(unit_i · centroid_c)
            labels = _assign_to_centroids(unit_data, centroids_t, cfg.chunk_size, device)

            # Update: centroid_c = normalize(Σ_{i∈c} raw_data_i) — norm-weighted!
            new_centroids = torch.zeros(k_eff, d, dtype=torch.float32, device=device)
            counts = torch.zeros(k_eff, device=device)
            for i in range(0, n, cfg.chunk_size):
                end = min(i + cfg.chunk_size, n)
                chunk = raw_data[i:end].to(device)
                chunk_labels = labels[i:end].to(device)
                new_centroids.index_add_(0, chunk_labels, chunk)
                counts.index_add_(0, chunk_labels, torch.ones(end - i, device=device))

            # Handle empty clusters: re-init from worst-fit data points
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

            # Normalize to unit sphere
            new_centroids = F.normalize(new_centroids, dim=-1)

            # Check convergence
            moved = (new_centroids - centroids).norm().item()
            centroids = new_centroids
            centroids_t = centroids.t().contiguous()

            self._emit(
                KmeansIterEvent(iter=it, max_iters=cfg.max_iters, moved=moved, n_empty=n_empty)
            )

            if moved < 1e-6:
                logger.info("[%s] converged at iter %d (moved=%.6f)", cfg.name, it, moved)
                converged = True
                final_moved = moved
                break
            final_moved = moved

        # Final assignment
        labels = _assign_to_centroids(unit_data, centroids_t, cfg.chunk_size, device)

        logger.info("[%s] k-means done after %d iters", cfg.name, it + 1)
        self._emit(
            KmeansDoneEvent(
                iters_run=it + 1,
                final_moved=final_moved,
                converged=converged,
            )
        )
        return centroids, labels

    def _fit_euclidean(
        self, data: torch.Tensor, k_eff: int, n: int, d: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Body of _euclidean_kmeans."""
        cfg = self.config
        device = cfg.device

        # Forgy init: k random data points — adapts to data scale/location
        logger.info("[%s] Forgy init (k=%d, d=%d, euclidean)...", cfg.name, k_eff, d)
        perm = torch.randperm(n, device=data.device)[:k_eff]
        centroids = data[perm].to(device).clone()  # (k, d)

        final_moved = 0.0
        converged = False
        it = -1
        for it in range(cfg.max_iters):
            # Assign: argmin ||x - c||^2 = argmin(-2*x·c + ||c||^2)
            labels = _assign_to_centroids_l2(
                data, centroids.t().contiguous(), cfg.chunk_size, device
            )

            # Update: centroid_c = mean(x_i for i in c)
            new_centroids = torch.zeros(k_eff, d, dtype=torch.float32, device=device)
            counts = torch.zeros(k_eff, device=device)
            for i in range(0, n, cfg.chunk_size):
                end = min(i + cfg.chunk_size, n)
                chunk = data[i:end].to(device)
                chunk_labels = labels[i:end].to(device)
                new_centroids.index_add_(0, chunk_labels, chunk)
                counts.index_add_(0, chunk_labels, torch.ones(end - i, device=device))

            # Handle empty clusters: re-init from worst-fit data points
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

            # Check convergence
            moved = (new_centroids - centroids).norm().item()
            centroids = new_centroids

            self._emit(
                KmeansIterEvent(iter=it, max_iters=cfg.max_iters, moved=moved, n_empty=n_empty)
            )

            if moved < 1e-6:
                logger.info("[%s] converged at iter %d (moved=%.6f)", cfg.name, it, moved)
                converged = True
                final_moved = moved
                break
            final_moved = moved

        # Final assignment
        labels = _assign_to_centroids_l2(data, centroids.t().contiguous(), cfg.chunk_size, device)

        logger.info("[%s] k-means done after %d iters", cfg.name, it + 1)
        self._emit(
            KmeansDoneEvent(
                iters_run=it + 1,
                final_moved=final_moved,
                converged=converged,
            )
        )
        return centroids, labels

    @property
    def centroids(self) -> torch.Tensor:
        assert self._centroids is not None, "fit() not called yet"
        return self._centroids

    @property
    def labels(self) -> torch.Tensor:
        assert self._labels is not None, "fit() not called yet"
        return self._labels


# ---------------------------------------------------------------------------
# CodebookExperiment — full codebook build (replaces build_codebook + _do_cluster + _cluster_block)
# ---------------------------------------------------------------------------


@dataclass
class CodebookConfig:
    """Unified configuration for a codebook build.

    Covers all parameters from the original ``build_codebook`` function.
    The build.py / quantize.py / cuda_server.py / ray_worker.py path uses
    defaults for VQ-specific fields (cosine, non-shared, no sign-split, etc.).
    The VQ sweep path sets all fields explicitly.
    """

    params: CodebookParams
    k: int
    n_blocks: int
    n_codebooks: int
    num_experts: int
    out_dim: int
    device: torch.device | None = None
    name: str = ""
    metric: DistanceMetric = "cosine"
    shared_codebook: bool = False
    sign_split: bool | list[bool] = False
    residual_block_sizes: int | list[int] | None = None
    codebook_bits: int = 16
    primary_codebook_cache: object | None = None
    cache_key_str: str | None = None
    primary_block_size: int | None = None


class CodebookExperiment(Experiment):
    """Build a real-error residual PQ codebook for one projection.

    Primary (cb0) clusters unit directions with spherical k-means (cosine) and
    stores a per-(row, primary-block) scale. Residuals (cb1+) cluster the REAL
    reconstruction error with euclidean k-means (magnitude included, no scale).

    Every codebook partitions in_dim into its OWN block size; block sizes are
    fully independent (non-commensurate allowed). Leftover columns are the
    remainder region — stored raw (bf16) and reconstructed exactly.

    A single final re-fit of the primary block-wise scale is done against the
    full cumulative reconstruction (cosine only; euclidean has no scale).

    Composes ``KmeansExperiment`` internally for each k-means run, forwarding
    events with codebook/block context as ``CodebookIterEvent``.
    """

    def __init__(self, config: CodebookConfig) -> None:
        super().__init__()
        self.config = config
        self._result: CodebookResult | None = None

    def fit(self, rows: torch.Tensor) -> CodebookResult:
        """Build the codebook. Returns a ``CodebookResult``."""
        cfg = self.config
        device = cfg.device if cfg.device is not None else rows.device

        n_rows, in_dim = rows.shape

        # Primary block size
        if cfg.primary_block_size is not None:
            bs_0 = cfg.primary_block_size
        else:
            if cfg.n_blocks <= 0:
                raise ValueError(f"n_blocks={cfg.n_blocks} must be positive")
            bs_0 = in_dim // cfg.n_blocks
        if bs_0 <= 0 or bs_0 > in_dim:
            raise ValueError(f"primary block_size={bs_0} invalid for in_dim={in_dim}")
        n_blocks_0 = in_dim // bs_0
        cov_0 = n_blocks_0 * bs_0
        rem_0 = in_dim - cov_0

        # Residual block sizes
        rbs_list_raw = cfg.params.residual_block_sizes
        if rbs_list_raw is None:
            rbs_list = [bs_0] * (cfg.n_codebooks - 1)
        elif isinstance(rbs_list_raw, int):
            rbs_list = [rbs_list_raw] * (cfg.n_codebooks - 1)
        else:
            if len(rbs_list_raw) < cfg.n_codebooks - 1:
                raise ValueError(
                    f"residual_block_sizes has {len(rbs_list_raw)} values, "
                    f"need {cfg.n_codebooks - 1}"
                )
            rbs_list = list(rbs_list_raw[: cfg.n_codebooks - 1])
        for i, rbs in enumerate(rbs_list):
            if rbs <= 0:
                raise ValueError(f"residual_block_sizes[{i}]={rbs} must be positive")
        bs_per_codebook = [bs_0] + rbs_list

        raw = rows.float().clone()
        row_norms = raw.norm(dim=-1)
        zero_mask = row_norms < cfg.params.norm_threshold
        logger.info(
            "[%s] %d/%d full rows below norm threshold %s",
            cfg.name,
            zero_mask.sum().item(),
            n_rows,
            cfg.params.norm_threshold,
        )

        # K per codebook
        rk = cfg.params.residual_k
        if rk is None:
            k_per_codebook = [cfg.k] * cfg.n_codebooks
        elif isinstance(rk, int):
            k_per_codebook = [cfg.k if c == 0 else rk for c in range(cfg.n_codebooks)]
        else:
            if len(rk) < cfg.n_codebooks - 1:
                raise ValueError(
                    f"residual_k list has {len(rk)} values, need {cfg.n_codebooks - 1}"
                )
            k_per_codebook = [cfg.k if c == 0 else rk[c - 1] for c in range(cfg.n_codebooks)]

        cb_codebooks: list[torch.Tensor] = []
        cb_assignments: list[torch.Tensor] = []
        remainders_list: list[torch.Tensor | None] = [None] * cfg.n_codebooks
        scales_flat = torch.zeros(n_rows, n_blocks_0, dtype=torch.float32, device=device)

        # Normalize sign_split
        if isinstance(cfg.sign_split, bool):
            sign_split_list: list[bool] = [cfg.sign_split] + [False] * (cfg.n_codebooks - 1)
        else:
            if len(cfg.sign_split) != cfg.n_codebooks:
                raise ValueError(
                    f"sign_split list has {len(cfg.sign_split)} values, need {cfg.n_codebooks}"
                )
            sign_split_list = list(cfg.sign_split)
        signs_list: list[torch.Tensor | None] = [None] * cfg.n_codebooks

        recon_total = torch.zeros(n_rows, in_dim, dtype=torch.float32, device=device)
        error = raw.clone()  # E_0 = W

        self._emit(
            CodebookStartEvent(
                n_codebooks=cfg.n_codebooks,
                primary_block_size=bs_0,
                primary_k=cfg.k,
                metric=cfg.metric,
            )
        )

        for c in range(cfg.n_codebooks):
            k_c = k_per_codebook[c]
            metric_c: DistanceMetric = cfg.metric if c == 0 else "euclidean"
            bs_c = bs_per_codebook[c]
            n_blocks_c = in_dim // bs_c
            cov_c = n_blocks_c * bs_c
            rem_c = in_dim - cov_c
            ss_c = sign_split_list[c]

            if c == 0:
                data = raw[:, :cov_0].reshape(n_rows, n_blocks_0, bs_0)
            else:
                data = error[:, :cov_c].reshape(n_rows, n_blocks_c, bs_c)

            if ss_c:
                signs_c = torch.sign(data)
                signs_c[signs_c == 0] = 1.0
                data = data * signs_c
                signs_list[c] = signs_c.reshape(n_rows, cov_c)
                logger.info("[%s] cb=%d sign-split: clustering on first orthant", cfg.name, c)
            raw_for_update = data

            cb, asg, assigned = self._do_cluster(
                data=data,
                raw_data=raw_for_update,
                k_c=k_c,
                metric=metric_c,
                shared=cfg.shared_codebook,
                device=device,
                name=f"{cfg.name} cb={c}/{cfg.n_codebooks}",
                skip_zeros=cfg.params.skip_zeros if c == 0 else False,
                first_orthant=ss_c,
                cache=cfg.primary_codebook_cache if c == 0 else None,
                cache_key=cfg.cache_key_str if c == 0 else None,
                codebook_idx=c,
                n_codebooks=cfg.n_codebooks,
            )
            assigned = assigned.clone()
            assigned[zero_mask] = 0.0
            cb_codebooks.append(cb)
            cb_assignments.append(asg)

            if c == 0:
                if cfg.metric == "cosine":
                    dot = torch.einsum("nbd,nbd->nb", data, assigned)
                    dot[zero_mask] = 0.0
                    scales_flat = dot
                    prim = (scales_flat.unsqueeze(-1) * assigned).reshape(n_rows, cov_0)
                else:
                    scales_flat = torch.zeros(
                        n_rows, n_blocks_0, dtype=torch.float32, device=device
                    )
                    prim = assigned.reshape(n_rows, cov_0)
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
            self._emit(CodebookDoneEvent(codebook_idx=c, n_codebooks=cfg.n_codebooks))

        # Quantize codebook centroids
        if cfg.codebook_bits < 16:
            levels = 2 ** (cfg.codebook_bits - 1) - 1
            for c in range(cfg.n_codebooks):
                cb = cb_codebooks[c]
                flat = cb.reshape(-1, cb.shape[-1])
                cb_max = flat.abs().max(dim=0).values.clamp(min=1e-10)
                q_scale = cb_max / levels
                q = torch.round(flat / q_scale).clamp(-levels, levels)
                cb_codebooks[c] = (q * q_scale).reshape(cb.shape).to(cb.dtype)
                logger.info(
                    "[%s] codebook[%d] quantized to int%d (per-centroid scale)",
                    cfg.name,
                    c,
                    cfg.codebook_bits,
                )

        # Rebuild reconstruction from quantized centroids
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
            shared_codebook=cfg.shared_codebook,
            sign_bits=signs_list,
            remainders=remainders_list,
            device=device,
        )

        # Final scale re-fit (cosine only)
        if cfg.metric == "cosine":
            for b in range(n_blocks_0):
                cols = slice(b * bs_0, (b + 1) * bs_0)
                cb0 = cb_codebooks[0][0] if cfg.shared_codebook else cb_codebooks[0][b]
                d = cb0.float().t()[cb_assignments[0][b].to(cb0.device)].to(device)
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

        assignments_out = [a.cpu() for a in cb_assignments]
        codebooks_out = [cb.cpu() for cb in cb_codebooks]
        remainders_list_out = [
            (r.to(torch.bfloat16).cpu() if r is not None else None) for r in remainders_list
        ]
        remainders_out = (
            None if all(r is None for r in remainders_list_out) else remainders_list_out
        )
        rbs_out = [bs for bs in bs_per_codebook[1:] if bs != bs_0]
        signs_out_list = [(s.cpu() if s is not None else None) for s in signs_list]
        signs_out = None if all(s is None for s in signs_out_list) else signs_out_list

        result = CodebookResult(
            codebooks=codebooks_out,
            assignments=assignments_out,
            scales=scales_flat.cpu(),
            zero_mask=zero_mask.cpu(),
            n_blocks=n_blocks_0,
            n_codebooks=cfg.n_codebooks,
            shared_codebook=cfg.shared_codebook,
            sign_bits=signs_out,
            residual_block_sizes=rbs_out if rbs_out else None,
            remainders=remainders_out,
            block_sizes=list(bs_per_codebook),
            num_experts=cfg.num_experts,
        )
        self._result = result
        return result

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct weights from the built codebook."""
        assert self._result is not None, "fit() not called yet"
        return self._result.reconstruct()

    @property
    def result(self) -> CodebookResult:
        assert self._result is not None, "fit() not called yet"
        return self._result

    def _do_cluster(
        self,
        data: torch.Tensor,
        raw_data: torch.Tensor,
        k_c: int,
        metric: DistanceMetric,
        shared: bool,
        device: torch.device,
        name: str,
        skip_zeros: bool,
        first_orthant: bool,
        cache: object | None,
        cache_key: str | None,
        codebook_idx: int,
        n_codebooks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cluster one codebook's block-partitioned data (replaces _do_cluster).

        Returns (codebook, assignments, assigned) where:
          codebook:   (n_blocks_c, bs_c, K) or (1, bs_c, K) if shared
          assignments:(n_blocks_c, n_rows) int64
          assigned:   (n_rows, n_blocks_c, bs_c) — gathered centroids per row
        """
        n_rows, n_blocks_c, bs_c = data.shape
        params = self.config.params

        if shared:
            pooled = data.reshape(n_rows * n_blocks_c, bs_c)
            pooled_raw = raw_data.reshape(n_rows * n_blocks_c, bs_c)

            cached_codebook = None
            if cache is not None and cache_key is not None:
                cached_codebook = cache.get(cache_key)  # type: ignore[attr-defined]

            if cached_codebook is not None:
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
                res = self._cluster_block(
                    block_data=pooled,
                    k=k_c,
                    metric=metric,
                    skip_zeros=skip_zeros,
                    device=device,
                    name=name,
                    first_orthant=first_orthant,
                    raw_data=pooled_raw,
                    max_train_samples=2**23,
                    codebook_idx=codebook_idx,
                    n_codebooks=n_codebooks,
                    block_idx=0,
                    n_blocks=1,
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
            res = self._cluster_block(
                block_data=data[:, b, :],
                k=k_c,
                metric=metric,
                skip_zeros=skip_zeros,
                device=device,
                name=f"{name} blk={b}/{n_blocks_c}",
                first_orthant=first_orthant,
                raw_data=raw_data[:, b, :],
                max_train_samples=0,
                codebook_idx=codebook_idx,
                n_codebooks=n_codebooks,
                block_idx=b,
                n_blocks=n_blocks_c,
            )
            block_codebooks.append(res.codebook)
            block_assigns.append(res.labels)
            assigned[:, b, :] = res.codebook.t()[res.labels.to(res.codebook.device)].to(device)
        cb = torch.stack(block_codebooks, dim=0)
        asg = torch.stack(block_assigns, dim=0)
        return cb, asg, assigned

    def _cluster_block(
        self,
        block_data: torch.Tensor,
        k: int,
        metric: DistanceMetric,
        skip_zeros: bool,
        device: torch.device,
        name: str,
        first_orthant: bool,
        raw_data: torch.Tensor | None,
        max_train_samples: int,
        codebook_idx: int,
        n_codebooks: int,
        block_idx: int,
        n_blocks: int,
    ) -> BlockClusterResult:
        """Cluster one block's sub-vectors (replaces _cluster_block)."""
        params = self.config.params
        n_rows, block_size = block_data.shape

        block_data = block_data.to(device)
        block_norms = block_data.norm(dim=-1)
        block_zero = block_norms < params.norm_threshold

        if skip_zeros:
            non_zero = block_data[~block_zero].float()
            logger.info(
                "[%s] %d/%d block-rows below norm threshold %s",
                name,
                block_zero.sum().item(),
                n_rows,
                params.norm_threshold,
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
            metric,
        )

        k_eff = min(k, non_zero.shape[0])
        d = non_zero.shape[1]
        budget_bytes = params.chunk_budget_mb * 1024 * 1024
        bytes_per_row = (d + k_eff) * 4 * 2
        chunk_size = max(1, budget_bytes // bytes_per_row)
        chunk_size = min(chunk_size, non_zero.shape[0])

        # Sub-sample for k-means training
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

        # Create KmeansExperiment and subscribe to its events, wrapping with context
        km_config = KmeansConfig(
            metric=metric,
            k=k_eff,
            max_iters=params.max_iters,
            device=device,
            name=name,
            chunk_size=train_chunk_size,
            first_orthant=first_orthant,
            raw_data=train_raw if metric == "cosine" else None,
        )
        km_exp = KmeansExperiment(km_config)

        # Forward KmeansIterEvent as CodebookIterEvent with codebook/block context
        def wrap_iter(e: KmeansIterEvent) -> None:
            self._emit(
                CodebookIterEvent(
                    codebook_idx=codebook_idx,
                    n_codebooks=n_codebooks,
                    block_idx=block_idx,
                    n_blocks=n_blocks,
                    iter=e.iter,
                    max_iters=e.max_iters,
                    moved=e.moved,
                    n_empty=e.n_empty,
                    metric=metric,
                )
            )

        km_exp.on(KmeansIterEvent, wrap_iter)
        # Also forward raw KmeansIterEvent for tqdm listeners
        self._forward_from(km_exp, KmeansIterEvent)
        self._forward_from(km_exp, KmeansStartEvent)
        self._forward_from(km_exp, KmeansDoneEvent)

        centroids_kbd, labels_nz = km_exp.fit(train_data)

        if metric == "cosine":
            unit = F.normalize(non_zero, dim=-1)
            # Re-assign ALL points to the learned centroids
            labels_nz = _assign_to_centroids(unit, centroids_kbd, chunk_size, device)
            assigned_centroids = centroids_kbd[labels_nz.to(centroids_kbd.device)].to(unit.device)
            scales_nz = torch.einsum("nd,nd->n", non_zero, assigned_centroids)

            cos_sim = torch.einsum("nd,nd->n", unit, assigned_centroids)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[%s] training cos(raw_unit, centroid):"
                    " mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
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
                assigned_centroids = centroids_kbd[labels_nz.to(centroids_kbd.device)].to(
                    unit.device
                )
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
        else:
            # Re-assign ALL points (l2: argmin distance)
            labels_nz = _assign_to_centroids_l2(
                non_zero,
                centroids_kbd.t().contiguous(),
                chunk_size,
                device,
            )
            labels_full = torch.zeros(n_rows, dtype=torch.long, device=block_data.device)
            labels_full[~block_zero] = labels_nz
            scales_full = torch.zeros(n_rows, dtype=torch.float32, device=block_data.device)

        if logger.isEnabledFor(logging.DEBUG):
            unique, counts = torch.unique(labels_nz, return_counts=True)
            logger.debug(
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


# ---------------------------------------------------------------------------
# TqdmListener — renders KmeansStart/Iter/Done events to stderr (preserves
# the original tqdm output)
# ---------------------------------------------------------------------------


class TqdmListener:
    """Subscribe to KmeansExperiment events and render them via tqdm.

    Usage::

        listener = TqdmListener()
        exp.on(KmeansStartEvent, listener.on_start)
        exp.on(KmeansIterEvent, listener.on_iter)
        exp.on(KmeansDoneEvent, listener.on_done)
    """

    def __init__(self) -> None:
        self._pbar: tqdm | None = None

    def on_start(self, e: KmeansStartEvent) -> None:
        self._pbar = tqdm(total=e.max_iters, desc=e.name, leave=True)

    def on_iter(self, e: KmeansIterEvent) -> None:
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(moved=f"{e.moved:.6f}", empty=e.n_empty)

    def on_done(self, e: KmeansDoneEvent) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
