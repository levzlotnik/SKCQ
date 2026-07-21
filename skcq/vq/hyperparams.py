"""VQ hyperparameter search space data structures.

Single source of truth for the hyperparameter range. The orchestrator holds a
VQHyperparamRange instance, iterates it to populate the job queue. The dashboard
edits the range via POST /api/range. No JSON files, no magic constants in
scripts.

Core types:
    VQCodebookSpec  — params for one codebook (primary or residual), frozen+hashable
    VQConfig        — one point in the search space (projection + primary + residuals)
    VQCodebookRange — tunable dimensions for one codebook slot (cross-producted)
    VQHyperparamRange — the full search space (iterable, yields VQConfig)
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model dimensions (Qwen/Qwen3.6-35B-A3B, layer 24)
# ---------------------------------------------------------------------------
# TODO: these should come from the model config at runtime, not hardcoded.
# For now they're here so the range can do divisibility filtering without
# loading the model. The orchestrator can override via VQHyperparamRange(model_dims=...).

MODEL_DIMS: dict[str, tuple[int, int]] = {
    # projection -> (n_rows, in_dim)
    "gate": (131072, 2048),
    "up": (131072, 2048),
    "down": (524288, 512),
}


# ---------------------------------------------------------------------------
# Codebook spec — one codebook's params (primary or residual)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VQCodebookSpec:
    """Params for one codebook (primary or residual).

    Frozen + hashable so it can be used as a dict key (for caching).
    """

    block_size: int
    K: int
    # Primary-only fields (None for residuals — residuals are always euclidean,
    # fp16 scales). Exception: `sign_split` may ALSO be set on a residual to
    # enable per-codebook SSVQ on that residual (its own fold/cluster/signs).
    metric: str | None = None  # "cosine" | "euclidean"
    sign_split: bool | None = None
    scale_dtype: str | None = None  # "int8", "fp16", "bf16", "fp8_e4m3", ...

    def cache_key(self, layer: int, projection: str) -> str:
        """Deterministic cache key for this codebook spec."""
        parts = [str(layer), projection, f"bs{self.block_size}", f"K{self.K}"]
        if self.metric is not None:
            parts.append(f"m{self.metric}")
        if self.sign_split is not None:
            parts.append(f"ss{int(self.sign_split)}")
        if self.scale_dtype is not None:
            parts.append(f"sd{self.scale_dtype}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# VQConfig — one point in the search space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VQConfig:
    """One configuration to evaluate."""

    projection: str
    primary: VQCodebookSpec
    residuals: tuple[VQCodebookSpec, ...]  # 0+ residual codebooks

    @property
    def n_codebooks(self) -> int:
        return 1 + len(self.residuals)

    @property
    def id(self) -> str:
        """Deterministic ID for this config (used as SQLite PK)."""
        parts = [self.projection, f"p_b{self.primary.block_size}_K{self.primary.K}"]
        if self.primary.metric:
            parts.append(f"m{self.primary.metric[:3]}")
        if self.primary.sign_split is not None:
            parts.append(f"ss{int(self.primary.sign_split)}")
        if self.primary.scale_dtype:
            parts.append(f"sd{self.primary.scale_dtype}")
        for i, r in enumerate(self.residuals):
            parts.append(f"r{i}_b{r.block_size}_K{r.K}")
            if r.sign_split:
                parts.append(f"r{i}ss{int(r.sign_split)}")
        return "_".join(parts)

    @property
    def est_bpw(self) -> float:
        """Estimated bits per weight (mirrors runner.bits_per_weight_kmeans)."""
        n_rows, in_dim = MODEL_DIMS[self.projection]
        block_size = self.primary.block_size
        n_blocks = in_dim // block_size

        # Scale bits: int8 = 8, fp16 = 16, bf16 = 16, fp8 = 8
        scale_bits_map = {"int8": 8, "fp16": 16, "bf16": 16, "fp8_e4m3": 8, "fp8_e5m2": 8}
        scale_bits_per_elem = scale_bits_map.get(self.primary.scale_dtype or "bf16", 16)

        codebook_bits_per_elem = 16  # codebook always stored in original dtype
        codebook_bits_total = 0
        assign_bits = 0
        remainder_bits = 0
        sign_bits = 0
        specs = [self.primary, *self.residuals]
        for spec in specs:
            bs_c = spec.block_size
            n_blocks_c = in_dim // bs_c
            cov_c = n_blocks_c * bs_c
            rem_c = in_dim - cov_c
            n_cb_c = 1  # always shared in the sweep
            codebook_bits_total += n_cb_c * spec.K * bs_c * codebook_bits_per_elem
            if spec.K <= 1:
                assign_bits += n_rows * n_blocks_c * 1
            else:
                assign_bits += n_rows * n_blocks_c * math.ceil(math.log2(spec.K))
            remainder_bits += 16 * n_rows * rem_c  # bf16 raw remainder
            if spec.sign_split:
                sign_bits += n_rows * cov_c  # 1 bit per covered element (this codebook)

        # Scales: one per primary block.
        scale_bits = n_rows * n_blocks * scale_bits_per_elem

        total_bits = codebook_bits_total + assign_bits + scale_bits + sign_bits + remainder_bits
        return total_bits / (n_rows * in_dim)

    @property
    def est_runtime_s(self) -> float:
        """Rough runtime estimate (seconds). Calibrated from empirical data."""
        n_rows, in_dim = MODEL_DIMS[self.projection]
        bs = self.primary.block_size
        n_blocks = in_dim // bs
        # Primary work
        total_work = n_rows * n_blocks * self.primary.K * bs
        # Residual work (euclidean, usually smaller K)
        for r in self.residuals:
            r_n_blocks = in_dim // r.block_size
            total_work += n_rows * r_n_blocks * r.K * r.block_size
        # Calibrate: bs=8 K=8192 n_rows=131072 → ~2.2e12 work → 30s @ 50 iters
        return max(5.0, total_work * 1.36e-11 * 50 / 100)


# ---------------------------------------------------------------------------
# Codebook range — tunable dimensions for one codebook slot
# ---------------------------------------------------------------------------


@dataclass
class VQCodebookRange:
    """Tunable dimensions for one codebook slot.

    Cross-producted to enumerate all VQCodebookSpec variants.
    For residual slots, metric/sign_split/scale_dtype are None (not applicable).
    """

    block_size: list[int] = field(default_factory=lambda: [8, 10, 12, 16, 24, 32, 64, 128])
    K: list[int] = field(
        default_factory=lambda: [
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
    )
    # Primary-only (None for residuals):
    metric: list[str] | None = None
    sign_split: list[bool] | None = None
    scale_dtype: list[str] | None = None

    def enumerate_specs(self) -> Iterator[VQCodebookSpec]:
        """Yield all VQCodebookSpec variants in this range."""
        for bs in self.block_size:
            for k_val in self.K:
                if (
                    self.metric is not None
                    and self.sign_split is not None
                    and self.scale_dtype is not None
                ):
                    for m in self.metric:
                        for ss in self.sign_split:
                            for sd in self.scale_dtype:
                                yield VQCodebookSpec(
                                    block_size=bs, K=k_val, metric=m, sign_split=ss, scale_dtype=sd
                                )
                else:
                    # Residual: no metric/sign_split/scale_dtype
                    yield VQCodebookSpec(block_size=bs, K=k_val)

    def count(self, in_dim: int) -> int:
        """Number of specs. Block sizes need not divide in_dim (remainder handled)."""
        return sum(1 for _ in self.enumerate_specs())


# ---------------------------------------------------------------------------
# VQHyperparamRange — the full search space
# ---------------------------------------------------------------------------


@dataclass
class VQHyperparamRange:
    """The full VQ hyperparameter search space.

    Iterable; yields VQConfig. The cross product is:
        projection × primary_specs × (residual_depth ∈ 0..len(residuals))
                    × (for each depth d, residuals[0:d] cross-producted)

    So if `residuals` has 2 slots, configs include:
        - primary only (depth 0)
        - primary + residuals[0] (depth 1)
        - primary + residuals[0] + residuals[1] (depth 2)

    Residuals are applied in order (slot 0 before slot 1) — this is the
    residual codebook hierarchy: each residual is trained on the residual
    after all previous codebooks.

    Filtering rules (applied in __iter__):
        - K <= n_rows (can't have more centroids than training points)
        - est_bpw must be in [bpw_min, bpw_max]

    Block sizes need NOT divide in_dim, and residual block sizes are fully
    independent of the primary (non-commensurate allowed) — leftover columns
    are stored raw (bf16) as a per-codebook remainder.
    """

    projection: list[str] = field(default_factory=lambda: ["gate", "down"])
    bpw_min: float = 1.0
    bpw_max: float = 6.0
    primary: VQCodebookRange = field(default_factory=VQCodebookRange)
    residuals: list[VQCodebookRange] = field(default_factory=list)
    model_dims: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(MODEL_DIMS))

    def __iter__(self) -> Iterator[VQConfig]:
        for proj in self.projection:
            if proj not in self.model_dims:
                continue
            n_rows, in_dim = self.model_dims[proj]
            for primary_spec in self._filtered_primary_specs(in_dim, n_rows):
                # Depth 0: primary only
                cfg = VQConfig(projection=proj, primary=primary_spec, residuals=())
                if self._passes_bpw(cfg):
                    yield cfg
                # Depth 1..N: primary + residuals[0:d]
                yield from self._enumerate_residual_depths(
                    proj, primary_spec, in_dim, n_rows, depth=0, chain=()
                )

    def _filtered_primary_specs(self, in_dim: int, n_rows: int) -> Iterator[VQCodebookSpec]:
        for spec in self.primary.enumerate_specs():
            if n_rows < spec.K:
                continue
            yield spec

    def _enumerate_residual_depths(
        self,
        proj: str,
        primary_spec: VQCodebookSpec,
        in_dim: int,
        n_rows: int,
        depth: int,
        chain: tuple[VQCodebookSpec, ...],
    ) -> Iterator[VQConfig]:
        """Recursively yield configs with increasing residual depth."""
        if depth >= len(self.residuals):
            return
        for r_spec in self._filtered_residual_specs(
            self.residuals[depth], primary_spec, in_dim, n_rows
        ):
            new_chain = chain + (r_spec,)
            cfg = VQConfig(projection=proj, primary=primary_spec, residuals=new_chain)
            if self._passes_bpw(cfg):
                yield cfg
            yield from self._enumerate_residual_depths(
                proj, primary_spec, in_dim, n_rows, depth + 1, new_chain
            )

    @staticmethod
    def _filtered_residual_specs(
        rrange: VQCodebookRange, primary_spec: VQCodebookSpec, in_dim: int, n_rows: int
    ) -> Iterator[VQCodebookSpec]:
        for spec in rrange.enumerate_specs():
            if n_rows < spec.K:
                continue
            yield spec

    def _passes_bpw(self, cfg: VQConfig) -> bool:
        bpw = cfg.est_bpw
        return self.bpw_min <= bpw <= self.bpw_max

    def __len__(self) -> int:
        """Total configs after filtering (forces full iteration)."""
        return sum(1 for _ in self)

    def total_runtime_s(self) -> float:
        """Estimated total runtime (serial)."""
        return sum(c.est_runtime_s for c in self)

    def to_dict(self) -> dict:
        """Serialize for the dashboard API."""
        return {
            "projection": self.projection,
            "bpw_min": self.bpw_min,
            "bpw_max": self.bpw_max,
            "primary": {
                "block_size": self.primary.block_size,
                "K": self.primary.K,
                "metric": self.primary.metric,
                "sign_split": self.primary.sign_split,
                "scale_dtype": self.primary.scale_dtype,
            },
            "residuals": [
                {
                    "block_size": r.block_size,
                    "K": r.K,
                }
                for r in self.residuals
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> VQHyperparamRange:
        """Deserialize from dashboard API JSON."""
        primary = VQCodebookRange(
            block_size=d["primary"]["block_size"],
            K=d["primary"]["K"],
            metric=d["primary"].get("metric"),
            sign_split=d["primary"].get("sign_split"),
            scale_dtype=d["primary"].get("scale_dtype"),
        )
        residuals = [
            VQCodebookRange(block_size=r["block_size"], K=r["K"]) for r in d.get("residuals", [])
        ]
        return cls(
            projection=d["projection"],
            bpw_min=d["bpw_min"],
            bpw_max=d["bpw_max"],
            primary=primary,
            residuals=residuals,
        )


# ---------------------------------------------------------------------------
# Default range (matches the previous gen_vq_hyperparams.py curated subset)
# ---------------------------------------------------------------------------


def default_range() -> VQHyperparamRange:
    """Curated default search space.

    Matches the previous gen_vq_hyperparams.py defaults: bs=[8,10,12,16,24,32,64,128],
    K=[256...65536] (not 1M), cosine + SSVQ + int8 scales, one residual slot
    with bs=[None,4,8,16,32] and K=[16...8192].
    """
    return VQHyperparamRange(
        projection=["gate", "down"],
        bpw_min=1.0,
        bpw_max=6.0,
        primary=VQCodebookRange(
            block_size=[8, 10, 12, 16, 24, 32, 64, 128],
            K=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
            metric=["cosine"],
            sign_split=[True],
            scale_dtype=["int8"],
        ),
        residuals=[
            VQCodebookRange(
                block_size=[8, 16, 32],
                K=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            )
        ],
    )
