from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass

import torch

from skcq.codebook.aqlm import AQLMCodebook, AQLMCodebookConfig
from skcq.codebook.base import DistanceMetric
from skcq.codebook.cwr import SSVQCWR, CompressedWeightsRepresentation, SphericalCWR
from skcq.codebook.euclidean import EuclideanCodebook, EuclideanCodebookConfig
from skcq.codebook.events import (
    CodebookDoneEvent,
    CodebookIterEvent,
    CodebookStartEvent,
    Experiment,
    KmeansDoneEvent,
    KmeansIterEvent,
    KmeansStartEvent,
)
from skcq.codebook.spherical import (
    SphericalCodebook,
    SphericalCodebookConfig,
    quantize_spherical_scales,
    scale_bits_per_elem,
)
from skcq.codebook.ssvq import SSVQCodebook
from skcq.protocol import ProgressMessage, send_frame
from skcq.vq.bpw import bits_per_weight_kmeans

logger = logging.getLogger(__name__)


@dataclass
class VQStartEvent:
    config_id: str
    projection: str
    block_size: int
    K: int
    n_codebooks: int
    metric: DistanceMetric


@dataclass
class VQDoneEvent:
    config_id: str
    row: dict


@dataclass
class VQRunConfig:
    projection: str
    in_dim: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    block_size: int
    K: int
    n_codebooks: int
    metric: DistanceMetric
    residual_k: int | list[int] | None
    residual_sign_split: bool | list[bool] | None
    shared_codebook: bool
    sign_split: bool
    max_iters: int
    scale_dtype: str
    residual_block_sizes: int | list[int] | None
    codebook_bits: int
    layer_idx: int
    device: torch.device
    chunk_budget_mb: int
    primary_codebook_cache: object | None = None
    cache_key_str: str | None = None


def _resolve_list(val, n, default):
    if val is None:
        return [default] * n
    if isinstance(val, (int, bool)):
        return [val] * n
    return list(val[:n]) + [default] * (n - len(val))


class VQRunExperiment(Experiment):
    """One VQ config run. Composes AQLMCodebook internally."""

    def __init__(self, config: VQRunConfig) -> None:
        super().__init__()
        self.config = config
        self._result: dict | None = None

    def fit(self, W_raw: torch.Tensor) -> dict:  # noqa: N803
        cfg = self.config
        device = cfg.device
        n_rows = W_raw.shape[0]
        w_norm = W_raw.float().norm().item()
        n_blocks = cfg.in_dim // cfg.block_size
        remainder_dim = cfg.in_dim - n_blocks * cfg.block_size

        k_per_codebook = _resolve_list(cfg.residual_k, cfg.n_codebooks, cfg.K)
        if isinstance(cfg.residual_k, int | None):
            k_per_codebook = [
                cfg.K if c == 0 else (cfg.residual_k or cfg.K) for c in range(cfg.n_codebooks)
            ]
        elif cfg.residual_k is None:
            k_per_codebook = [cfg.K] * cfg.n_codebooks
        else:
            k_per_codebook = [
                cfg.K if c == 0 else cfg.residual_k[c - 1] for c in range(cfg.n_codebooks)
            ]

        if cfg.residual_block_sizes is None:
            bs_per_codebook = [cfg.block_size] * cfg.n_codebooks
        elif isinstance(cfg.residual_block_sizes, int):
            bs_per_codebook = [
                cfg.block_size if c == 0 else cfg.residual_block_sizes
                for c in range(cfg.n_codebooks)
            ]
        else:
            bs_per_codebook = [
                cfg.block_size if c == 0 else cfg.residual_block_sizes[c - 1]
                for c in range(cfg.n_codebooks)
            ]

        if cfg.residual_sign_split is None:
            res_ss = [False] * (cfg.n_codebooks - 1)
        elif isinstance(cfg.residual_sign_split, bool):
            res_ss = [cfg.residual_sign_split] * (cfg.n_codebooks - 1)
        else:
            res_ss = list(cfg.residual_sign_split[: cfg.n_codebooks - 1])
            res_ss += [False] * (cfg.n_codebooks - 1 - len(res_ss))
        sign_split_list = [cfg.sign_split, *res_ss]

        shared_tag = "shared" if cfg.shared_codebook else "perblock"
        ssvq_tag = "ssvq" if cfg.sign_split else "nosign"
        cb_parts = [
            f"cb{c}_b{bs_per_codebook[c]}k{k_per_codebook[c]}" for c in range(cfg.n_codebooks)
        ]
        cb_id = "-".join(cb_parts)
        scale_tag = f"_{cfg.scale_dtype}" if cfg.scale_dtype != "bf16" else ""
        cb_qtag = f"_cb{cfg.codebook_bits}" if cfg.codebook_bits < 16 else ""
        label = f"kmeans_{cb_id}_{cfg.metric[:3]}_{shared_tag}_{ssvq_tag}{scale_tag}{cb_qtag}"

        logger.info(
            "[%s] %s (n_blocks=%d, rem=%d, K=%d, K_r=%s, bs_r=%s, cb=%d, metric=%s, shared=%s)",
            cfg.projection,
            label,
            n_blocks,
            remainder_dim,
            cfg.K,
            cfg.residual_k,
            cfg.residual_block_sizes,
            cfg.n_codebooks,
            cfg.metric,
            cfg.shared_codebook,
        )

        out_dim = cfg.intermediate_size if cfg.projection != "down" else cfg.hidden_size
        config_id = f"L{cfg.layer_idx}.{cfg.projection}.{label}"

        self._emit(
            VQStartEvent(
                config_id=config_id,
                projection=cfg.projection,
                block_size=cfg.block_size,
                K=cfg.K,
                n_codebooks=cfg.n_codebooks,
                metric=cfg.metric,
            )
        )

        # Build primary codebook
        from skcq.codebook.base import WeightsCodebookBase

        if cfg.metric == "cosine":
            primary: WeightsCodebookBase = SphericalCodebook(
                SphericalCodebookConfig(
                    block_size=cfg.block_size,
                    K=cfg.K,
                    shared=cfg.shared_codebook,
                    max_iters=cfg.max_iters,
                    norm_threshold=0.001,
                    skip_zeros=True,
                    chunk_budget_mb=cfg.chunk_budget_mb,
                    device=device,
                    max_train_samples=2**23 if cfg.shared_codebook else 0,
                )
            )
            if cfg.sign_split:
                primary = SSVQCodebook(primary)
        else:
            primary = EuclideanCodebook(
                EuclideanCodebookConfig(
                    block_size=cfg.block_size,
                    K=cfg.K,
                    shared=cfg.shared_codebook,
                    max_iters=cfg.max_iters,
                    norm_threshold=0.001,
                    skip_zeros=True,
                    chunk_budget_mb=cfg.chunk_budget_mb,
                    device=device,
                    max_train_samples=2**23 if cfg.shared_codebook else 0,
                )
            )

        # Build residual codebooks
        residuals: list[WeightsCodebookBase] = []
        for c in range(1, cfg.n_codebooks):
            euc: WeightsCodebookBase = EuclideanCodebook(
                EuclideanCodebookConfig(
                    block_size=bs_per_codebook[c],
                    K=k_per_codebook[c],
                    shared=cfg.shared_codebook,
                    max_iters=cfg.max_iters,
                    norm_threshold=0.001,
                    skip_zeros=False,
                    chunk_budget_mb=cfg.chunk_budget_mb,
                    device=device,
                )
            )
            if sign_split_list[c]:
                euc = SSVQCodebook(euc)
            residuals.append(euc)

        aqlm = AQLMCodebook(
            primary=primary,
            residuals=residuals,
            config=AQLMCodebookConfig(
                codebook_bits=cfg.codebook_bits,
                norm_threshold=0.001,
                device=device,
                out_dim=out_dim,
                num_experts=cfg.num_experts,
            ),
        )

        self._forward_from(aqlm, CodebookStartEvent)
        self._forward_from(aqlm, CodebookIterEvent)
        self._forward_from(aqlm, CodebookDoneEvent)
        self._forward_from(aqlm, KmeansStartEvent)
        self._forward_from(aqlm, KmeansIterEvent)
        self._forward_from(aqlm, KmeansDoneEvent)

        cwr = aqlm.fit(W_raw.to(device))

        # Quantize scales post-hoc
        sc_bits = scale_bits_per_elem(cfg.scale_dtype)
        if cfg.metric == "cosine":
            cb = aqlm.primary
            cwr_p: CompressedWeightsRepresentation = cwr.primary
            if isinstance(cb, SSVQCodebook):
                cb = cb.inner
                assert isinstance(cwr_p, SSVQCWR)
                cwr_p = cwr_p.inner
            if isinstance(cb, SphericalCodebook) and isinstance(cwr_p, SphericalCWR):
                quantize_spherical_scales(cwr_p, cb, cfg.scale_dtype)
                logger.info(
                    "  [%s] scale quantized to %s (%d bits/elem)",
                    cfg.projection,
                    cfg.scale_dtype,
                    sc_bits,
                )

        w_recon = aqlm.reconstruct(cwr)
        err = torch.norm(W_raw.float() - w_recon).item() / w_norm

        bpw = bits_per_weight_kmeans(
            n_rows,
            cfg.in_dim,
            n_blocks,
            cfg.block_size,
            cfg.n_codebooks,
            k_per_codebook,
            shared_codebook=cfg.shared_codebook,
            sign_split=cfg.sign_split,
            scale_bits_per_elem=sc_bits,
            bs_per_codebook=bs_per_codebook,
            codebook_bits=cfg.codebook_bits,
            residual_sign_split=res_ss,
            primary_metric=cfg.metric,
        )
        comp_ratio = 16.0 / bpw

        logger.info("  [%s] err=%.6f bpw=%.3f cr=%.2f", cfg.projection, err, bpw, comp_ratio)

        del cwr, w_recon
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = {
            "projection": cfg.projection,
            "scheme": label,
            "block_size": cfg.block_size,
            "K": cfg.K,
            "n_codebooks": cfg.n_codebooks,
            "metric": cfg.metric,
            "shared": cfg.shared_codebook,
            "sign_split": cfg.sign_split,
            "scale_dtype": cfg.scale_dtype,
            "kmeans_iters": cfg.max_iters,
            "residual_block_sizes": (
                [cfg.block_size] + [bs for bs in bs_per_codebook[1:] if bs != cfg.block_size]
                if any(bs != cfg.block_size for bs in bs_per_codebook[1:])
                else []
            ),
            "rel_fro_err": err,
            "bits_per_weight": bpw,
            "compression_ratio": comp_ratio,
        }
        self._result = row
        self._emit(VQDoneEvent(config_id=config_id, row=row))
        return row

    @property
    def result(self) -> dict:
        assert self._result is not None, "fit() not called yet"
        return self._result


class WireListener:
    def __init__(self, sock: socket.socket, lock: threading.Lock, worker_name: str) -> None:
        self._sock = sock
        self._lock = lock
        self._worker_name = worker_name

    def on_iter(self, e: CodebookIterEvent) -> None:
        msg = ProgressMessage(
            worker_name=self._worker_name,
            codebook_idx=e.codebook_idx,
            n_codebooks=e.n_codebooks,
            block_idx=e.block_idx,
            n_blocks=e.n_blocks,
            iter=e.iter,
            max_iters=e.max_iters,
            moved=e.moved,
            n_empty=e.n_empty,
        )
        with self._lock:
            send_frame(self._sock, msg)
