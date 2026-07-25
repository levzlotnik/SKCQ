from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from skcq.codebook.cwr import (
    AQLMCWR,
    SSVQCWR,
    CompressedWeightsRepresentation,
    EuclideanCWR,
    SphericalCWR,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from skcq.codebook.aqlm import AQLMCodebook
    from skcq.codebook.base import WeightsCodebookBase

ForwardMode = Literal["matmul_gather", "gather_matmul"]


def _choose_forward_mode(k: int, out_dim: int) -> ForwardMode:
    return "gather_matmul" if k > out_dim else "matmul_gather"


def _codebook_pass(
    codebook: torch.Tensor,
    assignments: torch.Tensor,
    hidden_states: torch.Tensor,
    n_blocks: int,
    block_size: int,
    expert_idx: int,
    forward_mode: ForwardMode,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    cov = n_blocks * block_size
    hidden_cov = hidden_states[:, :cov]
    expert_assign = assignments[expert_idx]
    hidden_blocked = hidden_cov.reshape(-1, n_blocks, block_size).permute(1, 0, 2)

    if signs is not None:
        forward_mode = "gather_matmul"

    if forward_mode == "gather_matmul":
        gathered_cb = codebook.gather(
            dim=2, index=expert_assign.unsqueeze(1).expand(-1, block_size, -1)
        )
        if signs is not None:
            out_dim = expert_assign.shape[1]
            signs_e = signs[expert_idx].reshape(out_dim, n_blocks, block_size).permute(1, 2, 0)
            gathered_cb = gathered_cb * signs_e.to(gathered_cb.dtype)
        logits = torch.bmm(hidden_blocked, gathered_cb)
    else:
        n_tokens = hidden_cov.shape[0]
        logits = torch.bmm(hidden_blocked, codebook)
        logits = logits.gather(dim=2, index=expert_assign.unsqueeze(1).expand(-1, n_tokens, -1))
    return logits


def _remainder_pass(
    remainder: torch.Tensor,
    hidden_states: torch.Tensor,
    n_blocks: int,
    block_size: int,
    expert_idx: int,
) -> torch.Tensor:
    cov = n_blocks * block_size
    rem_w = remainder[expert_idx]
    x_rem = hidden_states[:, cov : cov + rem_w.shape[1]]
    return x_rem @ rem_w.mT


class _CodebookLayer(nn.Module):
    """One codebook's GPU forward layer.

    Stores centroids (Parameter), assignments (buffer), optional scales
    (Parameter, spherical primary only), remainder (buffer), optional signs
    (buffer, SSVQ).
    """

    codebook: nn.Parameter
    assignments: torch.Tensor
    scales: nn.Parameter | None
    remainder: torch.Tensor
    signs: torch.Tensor | None

    def __init__(
        self,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k: int,
        remainder: torch.Tensor | None = None,
        signs: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.out_dim = out_dim
        self.k = k
        self.has_scale = scales is not None
        num_experts = assignments.shape[0]
        self.codebook = nn.Parameter(codebook)
        self.register_buffer("assignments", assignments)
        if scales is not None:
            self.scales = nn.Parameter(scales)
        else:
            self.scales = None
        if remainder is None:
            remainder = torch.zeros(num_experts, out_dim, 0, dtype=codebook.dtype)
        self.rem = remainder.shape[-1]
        self.register_buffer("remainder", remainder)
        self.sign_cov = signs.shape[-1] if signs is not None else 0
        self.register_buffer("signs", signs)
        self.forward_mode: ForwardMode = _choose_forward_mode(k, out_dim)

    def block_pass(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return _codebook_pass(
            self.codebook,
            self.assignments,
            hidden_states,
            self.n_blocks,
            self.block_size,
            expert_idx,
            self.forward_mode,
            signs=self.signs,
        )

    def remainder_pass(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return _remainder_pass(
            self.remainder, hidden_states, self.n_blocks, self.block_size, expert_idx
        )

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return self.block_pass(hidden_states, expert_idx)


class CodebookModule(nn.Module):
    """Composite GPU forward module: primary + residual codebook layers.

    Each layer carries its OWN block partition (n_blocks_c, block_size_c) and
    a raw remainder. Forward sums:
      - primary per-block pass scaled by the primary per-block scale (if present),
        plus the primary raw remainder;
      - each residual per-block pass (unscaled), plus its raw remainder.

    SSVQ sign bits are per-layer: any layer may carry a ``signs`` buffer.
    """

    def __init__(self, layers: nn.ModuleList, n_codebooks: int, out_dim: int):
        super().__init__()
        self.layers = layers
        self.n_codebooks = n_codebooks
        self.out_dim = out_dim
        # Introspection aliases
        self.n_blocks = layers[0].n_blocks
        self.block_size = layers[0].block_size

    @property
    def primary(self) -> _CodebookLayer:
        return self.layers[0]

    @property
    def additives(self) -> list[_CodebookLayer]:
        return list(self.layers[1:])

    @classmethod
    def from_cwr(
        cls,
        aqlm: AQLMCodebook,
        cwr: AQLMCWR,
        out_dim: int,
    ) -> CodebookModule:
        """Build a CodebookModule from a fitted AQLMCodebook + AQLMCWR.

        Per-codebook block sizes, n_blocks, K, remainders, and signs are all
        derived from the codebook instances + CWR tensors.
        """
        n_rows = _cwr_n_rows(cwr)
        num_experts = n_rows // out_dim

        # Flatten the AQLM hierarchy into (codebook, cwr) pairs
        pairs: list[tuple[WeightsCodebookBase, CompressedWeightsRepresentation]] = [
            (aqlm.primary, cwr.primary),
            *zip(aqlm.residuals, cwr.residuals, strict=True),
        ]

        layer_list: list[_CodebookLayer] = []
        for cb, layer_cwr in pairs:
            centroids, assignments_3d, scales_3d, remainder_3d, signs_3d = _extract_and_reshape(
                cb, layer_cwr, num_experts, out_dim
            )
            n_blocks = assignments_3d.shape[1]
            block_size = getattr(cb, "block_size", centroids.shape[1])
            k = centroids.shape[-1]
            layer_list.append(
                _CodebookLayer(
                    codebook=centroids,
                    assignments=assignments_3d,
                    n_blocks=n_blocks,
                    block_size=block_size,
                    out_dim=out_dim,
                    k=k,
                    remainder=remainder_3d,
                    signs=signs_3d,
                    scales=scales_3d,
                )
            )

        return cls(nn.ModuleList(layer_list), len(pairs), out_dim)

    @classmethod
    def empty(
        cls,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k_list: list[int],
        num_experts: int,
        n_blocks_list: list[int] | None = None,
        block_size_list: list[int] | None = None,
        rem_list: list[int] | None = None,
        sign_cov_list: list[int] | None = None,
        has_scale_list: list[bool] | None = None,
    ) -> CodebookModule:
        """Create an empty (zero-filled) module sized for load_state_dict."""
        n_cb = len(k_list)
        nbl = n_blocks_list or [n_blocks] * n_cb
        bsl = block_size_list or [block_size] * n_cb
        reml = rem_list or [0] * n_cb
        scl = sign_cov_list or [0] * n_cb
        hsl = has_scale_list or [True] + [False] * (n_cb - 1)

        layer_list: list[_CodebookLayer] = []
        for c in range(n_cb):
            codebook = torch.zeros(nbl[c], bsl[c], k_list[c], dtype=torch.bfloat16)
            assignments = torch.zeros(num_experts, nbl[c], out_dim, dtype=torch.long)
            remainder = torch.zeros(num_experts, out_dim, reml[c], dtype=torch.bfloat16)
            signs = (
                torch.zeros(num_experts, out_dim, scl[c], dtype=torch.int8) if scl[c] > 0 else None
            )
            scales = (
                torch.zeros(num_experts, nbl[c], out_dim, dtype=torch.bfloat16) if hsl[c] else None
            )
            layer_list.append(
                _CodebookLayer(
                    codebook=codebook,
                    assignments=assignments,
                    n_blocks=nbl[c],
                    block_size=bsl[c],
                    out_dim=out_dim,
                    k=k_list[c],
                    remainder=remainder,
                    signs=signs,
                    scales=scales,
                )
            )
        return cls(nn.ModuleList(layer_list), n_cb, out_dim)

    @classmethod
    def load(cls, path: str | Any) -> CodebookModule:
        data = torch.load(path, weights_only=True)
        meta = data["meta"]
        module = cls.empty(
            n_blocks=meta["n_blocks"],
            block_size=meta["block_size"],
            out_dim=meta["out_dim"],
            k_list=meta["k_list"],
            num_experts=meta["num_experts"],
            n_blocks_list=meta.get("n_blocks_list"),
            block_size_list=meta.get("block_size_list"),
            rem_list=meta.get("rem_list"),
            sign_cov_list=meta.get("sign_cov_list"),
            has_scale_list=meta.get("has_scale_list"),
        )
        module.load_state_dict(data["state_dict"])
        return module

    def _all_layers(self) -> list[_CodebookLayer]:
        return list(self.layers)

    def state_dict_with_meta(self) -> dict[str, Any]:
        cbs = self._all_layers()
        return {
            "state_dict": self.state_dict(),
            "meta": {
                "n_blocks": self.n_blocks,
                "block_size": self.block_size,
                "out_dim": self.out_dim,
                "n_codebooks": self.n_codebooks,
                "k_list": [cb.k for cb in cbs],
                "n_blocks_list": [cb.n_blocks for cb in cbs],
                "block_size_list": [cb.block_size for cb in cbs],
                "rem_list": [cb.rem for cb in cbs],
                "sign_cov_list": [cb.sign_cov for cb in cbs],
                "has_scale_list": [cb.has_scale for cb in cbs],
                "num_experts": self.primary.assignments.shape[0],
            },
        }

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        p = self.primary
        prim = p(hidden_states, expert_idx)
        if p.has_scale and p.scales is not None:
            scale = p.scales[expert_idx]
            out = (prim * scale.unsqueeze(1)).sum(dim=0)
        else:
            out = prim.sum(dim=0)
        out = out + p.remainder_pass(hidden_states, expert_idx)
        for cb in self.layers[1:]:
            out = out + cb(hidden_states, expert_idx).sum(dim=0)
            out = out + cb.remainder_pass(hidden_states, expert_idx)
        return out


# ---------------------------------------------------------------------------
# CWR extraction helpers
# ---------------------------------------------------------------------------


def _cwr_n_rows(cwr: CompressedWeightsRepresentation) -> int:
    """Get the number of rows from a CWR (unwrapping AQLM/SSVQ if needed)."""
    if isinstance(cwr, AQLMCWR):
        return _cwr_n_rows(cwr.primary)
    inner = cwr.inner if isinstance(cwr, SSVQCWR) else cwr
    if isinstance(inner, SphericalCWR):
        return inner.idxs.shape[1]
    if isinstance(inner, EuclideanCWR):
        return inner.idxs.shape[1]
    raise TypeError(f"Unknown CWR type: {type(inner)}")


def _extract_and_reshape(
    cb: WeightsCodebookBase,
    cwr: CompressedWeightsRepresentation,
    num_experts: int,
    out_dim: int,
) -> tuple[
    torch.Tensor,  # centroids
    torch.Tensor,  # assignments_3d (num_experts, n_blocks, out_dim)
    torch.Tensor | None,  # scales_3d
    torch.Tensor | None,  # remainder_3d
    torch.Tensor | None,  # signs_3d
]:
    """Extract tensors from a (codebook, CWR) pair and reshape to per-expert."""
    # Unwrap SSVQ
    if isinstance(cwr, SSVQCWR):
        inner_cwr = cwr.inner
        signs_flat = cwr.signs  # (n_rows, cov)
    else:
        inner_cwr = cwr
        signs_flat = None

    centroids = cb.centroids_tensor  # (n_blocks, bs, K) or (1, bs, K)

    if isinstance(inner_cwr, SphericalCWR):
        idxs = inner_cwr.idxs  # (n_blocks, n_rows)
        scales_flat = inner_cwr.scales  # (n_rows, n_blocks)
        remainder_flat = inner_cwr.remainder
    elif isinstance(inner_cwr, EuclideanCWR):
        idxs = inner_cwr.idxs
        scales_flat = None
        remainder_flat = inner_cwr.remainder
    else:
        raise TypeError(f"Unknown inner CWR type: {type(inner_cwr)}")

    n_blocks = idxs.shape[0]
    idxs.shape[1]

    # Reshape to per-expert:
    # (n_blocks, n_rows) → (n_blocks, num_experts, out_dim) → (num_experts, n_blocks, out_dim)
    assignments_3d = idxs.reshape(n_blocks, num_experts, out_dim).permute(1, 0, 2).contiguous()

    if scales_flat is not None:
        scales_3d = (
            scales_flat.reshape(num_experts, out_dim, n_blocks).permute(0, 2, 1).contiguous()
        )
    else:
        scales_3d = None

    if remainder_flat is not None:
        rem = remainder_flat.shape[1]
        remainder_3d = remainder_flat.reshape(num_experts, out_dim, rem).contiguous()
    else:
        remainder_3d = None

    if signs_flat is not None:
        cov = signs_flat.shape[1]
        signs_3d = signs_flat.reshape(num_experts, out_dim, cov).to(torch.int8).contiguous()
    else:
        signs_3d = None

    return centroids, assignments_3d, scales_3d, remainder_3d, signs_3d


# ---------------------------------------------------------------------------
# SwiGLU + CodebookExperts (unchanged from original)
# ---------------------------------------------------------------------------


class _SwiGLU(nn.Module):
    def __init__(self, act_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.act_fn(gate) * up


class CodebookExperts(nn.Module):
    """Drop-in replacement for Qwen3_5MoeExperts using codebook quantization."""

    def __init__(
        self,
        gate: CodebookModule,
        up: CodebookModule,
        down: CodebookModule,
        num_experts: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate = gate
        self.up = up
        self.intermediate = _SwiGLU(act_fn)
        self.down = down

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor.item()

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            token_hidden = hidden_states[token_idx]

            gate_out = self.gate(token_hidden, expert_idx)
            up_out = self.up(token_hidden, expert_idx)
            intermediate = self.intermediate(gate_out, up_out)

            down_out = self.down(intermediate, expert_idx)

            down_out = down_out * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, down_out.to(final_hidden_states.dtype))

        return final_hidden_states
