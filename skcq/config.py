from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field

from skcq.codebook.aqlm import AQLMCodebook, AQLMCodebookConfig
from skcq.codebook.base import WeightsCodebookBase
from skcq.codebook.euclidean import EuclideanCodebook, EuclideanCodebookConfig
from skcq.codebook.spherical import SphericalCodebook, SphericalCodebookConfig
from skcq.codebook.ssvq import SSVQCodebook

__all__ = ["CodebookParams", "LayerOverride", "ExperimentConfig", "build_aqlm_codebook"]


class CodebookParams(BaseModel):
    k_gate: int = Field(default=4096, description="Codebook size for gate projection")
    k_up: int = Field(default=4096, description="Codebook size for up projection")
    k_down: int = Field(default=4096, description="Codebook size for down projection")
    n_blocks_gate_up: int = Field(
        default=1, description="Number of sub-blocks for gate/up input dim (PQ)"
    )
    n_blocks_down: int = Field(
        default=1, description="Number of sub-blocks for down input dim (PQ)"
    )
    n_codebooks: int = Field(
        default=2, description="Number of codebooks (1 = no residual, 2 = primary + 1 residual)"
    )
    max_iters: int = Field(default=100, description="Max k-means iterations")
    norm_threshold: float = Field(
        default=0.001,
        description="Rows with norm below this are treated as zeros, skipped from clustering",
    )
    skip_zeros: bool = Field(
        default=True,
        description="Whether to exclude near-zero rows from codebook building",
    )
    residual_k: int | list[int] | None = Field(
        default=None,
        description="K for residual codebooks (c>=1). If int, all residuals use that K. "
        "If list, residual_k[c-1] is used for codebook c. If None, same as primary K.",
    )
    residual_block_sizes: int | list[int] | None = Field(
        default=None,
        description="Block size for residual codebooks (c>=1). Fully independent of "
        "the primary block size (non-commensurate allowed); leftover columns are "
        "stored raw (bf16). If int, all residuals use that size. If list, "
        "residual_block_sizes[c-1] for codebook c. If None, same as primary.",
    )
    residual_sign_split: bool | list[bool] | None = Field(
        default=None,
        description="SSVQ sign-split for residual codebooks (c>=1). If bool, applies "
        "to all residuals. If list, residual_sign_split[c-1] for codebook c. If None, "
        "no residual sign-split.",
    )
    chunk_budget_mb: int = Field(
        default=2048,
        description="Memory budget (MB) for k-means chunking — reduce for low-VRAM GPUs",
    )


class LayerOverride(BaseModel):
    k_gate: int | None = None
    k_up: int | None = None
    k_down: int | None = None
    n_blocks_gate_up: int | None = None
    n_blocks_down: int | None = None
    n_codebooks: int | None = None
    max_iters: int | None = None
    norm_threshold: float | None = None
    skip_zeros: bool | None = None
    residual_k: int | list[int] | None = None
    residual_block_sizes: int | list[int] | None = None
    residual_sign_split: bool | list[bool] | None = None
    chunk_budget_mb: int | None = None


class ExperimentConfig(BaseModel):
    model_id: str = Field(default="Qwen/Qwen3.6-35B-A3B")
    defaults: CodebookParams = Field(default_factory=CodebookParams)
    layer_overrides: dict[int, LayerOverride] = Field(default_factory=dict)
    eval_samples: int = Field(
        default=1000, description="Number of C4 validation samples for perplexity"
    )
    output_dir: Path = Field(default=Path("codebooks"))

    def params_for_layer(self, layer_idx: int) -> CodebookParams:
        if layer_idx not in self.layer_overrides:
            return self.defaults
        override = self.layer_overrides[layer_idx]
        return self.defaults.model_copy(
            update={k: v for k, v in override.model_dump().items() if v is not None}
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def _resolve_residual_k(
    residual_k: int | list[int] | None, primary_k: int, n_codebooks: int
) -> list[int]:
    if residual_k is None:
        return [primary_k] * n_codebooks
    if isinstance(residual_k, int):
        return [primary_k if c == 0 else residual_k for c in range(n_codebooks)]
    return [primary_k if c == 0 else residual_k[c - 1] for c in range(n_codebooks)]


def _resolve_residual_bs(
    residual_bs: int | list[int] | None, primary_bs: int, n_codebooks: int
) -> list[int]:
    if residual_bs is None:
        return [primary_bs] * n_codebooks
    if isinstance(residual_bs, int):
        return [primary_bs if c == 0 else residual_bs for c in range(n_codebooks)]
    return [primary_bs if c == 0 else residual_bs[c - 1] for c in range(n_codebooks)]


def _resolve_residual_ss(residual_ss: bool | list[bool] | None, n_codebooks: int) -> list[bool]:
    if residual_ss is None:
        return [False] * n_codebooks
    if isinstance(residual_ss, bool):
        return [residual_ss] * n_codebooks
    return list(residual_ss[:n_codebooks]) + [False] * (n_codebooks - len(residual_ss))


def build_aqlm_codebook(
    params: CodebookParams,
    k: int,
    n_blocks: int,
    n_codebooks: int,
    in_dim: int,
    out_dim: int,
    num_experts: int = 0,
    device: torch.device | None = None,
    name: str = "",
) -> AQLMCodebook:
    """Construct an AQLMCodebook from CodebookParams for one projection.

    Primary is always spherical (cosine). Residuals (c>=1) are euclidean (l2),
    optionally SSVQ-wrapped when ``residual_sign_split`` is set.
    """
    primary_bs = in_dim // n_blocks if n_blocks > 0 else in_dim
    bs_list = _resolve_residual_bs(params.residual_block_sizes, primary_bs, n_codebooks)
    k_list = _resolve_residual_k(params.residual_k, k, n_codebooks)
    ss_list = _resolve_residual_ss(params.residual_sign_split, n_codebooks)

    primary = SphericalCodebook(
        SphericalCodebookConfig(
            block_size=primary_bs,
            K=k,
            max_iters=params.max_iters,
            norm_threshold=params.norm_threshold,
            skip_zeros=params.skip_zeros,
            chunk_budget_mb=params.chunk_budget_mb,
            device=device,
        )
    )

    residuals: list[WeightsCodebookBase] = []
    for c in range(1, n_codebooks):
        euc: WeightsCodebookBase = EuclideanCodebook(
            EuclideanCodebookConfig(
                block_size=bs_list[c],
                K=k_list[c],
                max_iters=params.max_iters,
                norm_threshold=params.norm_threshold,
                skip_zeros=False,
                chunk_budget_mb=params.chunk_budget_mb,
                device=device,
            )
        )
        if ss_list[c]:
            euc = SSVQCodebook(euc)
        residuals.append(euc)

    return AQLMCodebook(
        primary=primary,
        residuals=residuals,
        config=AQLMCodebookConfig(
            codebook_bits=16,
            norm_threshold=params.norm_threshold,
            device=device,
            out_dim=out_dim,
            num_experts=num_experts or None,
        ),
    )
