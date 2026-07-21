from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from skcq.clustering import CodebookResult

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

ForwardMode = Literal["matmul_gather", "gather_matmul"]


def _choose_forward_mode(k: int, out_dim: int) -> ForwardMode:
    """Pick the cheaper forward mode based on codebook K vs out_dim.

    matmul_gather costs tokens*block_size*K; gather_matmul costs tokens*block_size*out_dim.
    So gather_matmul wins when K > out_dim.
    """
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
    """Compute Q @ gathered_centroids for each block over the covered region.

    ``hidden_states`` is sliced to the covered region (n_blocks * block_size)
    before reshaping. Returns (n_blocks, tokens, out_dim) — NO scale applied.

    When ``signs`` is provided (SSVQ for this codebook), it is
    ``(num_experts, out_dim, cov)`` of ±1; the gather_matmul path is forced and
    the gathered centroids are multiplied by the per-(block, in-elem, out_idx)
    sign so that ``y[t,r] = Σ_d x[t,b,d]·signs[r,b,d]·cb[d,assign[r,b]]``.
    """
    cov = n_blocks * block_size
    hidden_cov = hidden_states[:, :cov]
    expert_assign = assignments[expert_idx]  # (n_blocks, out_dim)
    hidden_blocked = hidden_cov.reshape(-1, n_blocks, block_size).permute(1, 0, 2)

    if signs is not None:
        forward_mode = "gather_matmul"

    if forward_mode == "gather_matmul":
        gathered_cb = codebook.gather(
            dim=2,
            index=expert_assign.unsqueeze(1).expand(-1, block_size, -1),
        )  # (n_blocks, block_size, out_dim)
        if signs is not None:
            out_dim = expert_assign.shape[1]
            # (out_dim, cov) -> (out_dim, n_blocks, block_size) -> (n_blocks, block_size, out_dim)
            signs_e = signs[expert_idx].reshape(out_dim, n_blocks, block_size).permute(1, 2, 0)
            gathered_cb = gathered_cb * signs_e.to(gathered_cb.dtype)
        logits = torch.bmm(hidden_blocked, gathered_cb)
    else:
        n_tokens = hidden_cov.shape[0]
        logits = torch.bmm(hidden_blocked, codebook)
        logits = logits.gather(
            dim=2,
            index=expert_assign.unsqueeze(1).expand(-1, n_tokens, -1),
        )
    return logits  # (n_blocks, tokens, out_dim)


def _remainder_pass(
    remainder: torch.Tensor,
    hidden_states: torch.Tensor,
    n_blocks: int,
    block_size: int,
    expert_idx: int,
) -> torch.Tensor:
    """Contribution of the raw remainder columns: x[:, cov:] @ remainder.mT.

    ``remainder`` is (num_experts, out_dim, rem). When rem == 0 the matmul
    naturally yields a (tokens, out_dim) zero tensor.
    """
    cov = n_blocks * block_size
    rem_w = remainder[expert_idx]  # (out_dim, rem)
    x_rem = hidden_states[:, cov : cov + rem_w.shape[1]]  # (tokens, rem)
    return x_rem @ rem_w.mT  # (tokens, out_dim)


class _CodebookBase(nn.Module):
    """Shared machinery: per-block codebook + assignments + raw remainder.

    Optionally carries a per-codebook ``signs`` buffer (SSVQ) shaped
    ``(num_experts, out_dim, cov)`` of ±1. When present, the forward pass folds
    the signs into the gathered centroids (see ``_codebook_pass``).
    """

    codebook: nn.Parameter
    assignments: torch.Tensor
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
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.out_dim = out_dim
        self.k = k
        num_experts = assignments.shape[0]
        self.codebook = nn.Parameter(codebook)
        self.register_buffer("assignments", assignments)
        if remainder is None:
            remainder = torch.zeros(num_experts, out_dim, 0, dtype=codebook.dtype)
        self.rem = remainder.shape[-1]
        self.register_buffer("remainder", remainder)
        # Per-codebook SSVQ signs (±1) over the covered region, or None.
        self.sign_cov = signs.shape[-1] if signs is not None else 0
        self.register_buffer("signs", signs)
        self.forward_mode: ForwardMode = _choose_forward_mode(k, out_dim)

    def block_pass(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Per-block Q @ gathered_centroids over the covered region (no scale)."""
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
        """Raw-remainder contribution (tokens, out_dim)."""
        return _remainder_pass(
            self.remainder, hidden_states, self.n_blocks, self.block_size, expert_idx
        )


class SphericalCodebook(_CodebookBase):
    """Primary codebook (cb0). Spherical k-means. Direction codebook + per-block scale.

    The per-block scale multiplies ONLY the primary contribution (not residuals).
    """

    scales: nn.Parameter

    def __init__(
        self,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        scales: torch.Tensor,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k: int,
        remainder: torch.Tensor | None = None,
        signs: torch.Tensor | None = None,
    ):
        super().__init__(codebook, assignments, n_blocks, block_size, out_dim, k, remainder, signs)
        self.scales = nn.Parameter(scales)

    @classmethod
    def empty(
        cls,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k: int,
        num_experts: int,
        rem: int = 0,
        sign_cov: int = 0,
    ) -> SphericalCodebook:
        codebook = torch.zeros(n_blocks, block_size, k, dtype=torch.bfloat16)
        assignments = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.long)
        scales = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.bfloat16)
        remainder = torch.zeros(num_experts, out_dim, rem, dtype=torch.bfloat16)
        signs = (
            torch.zeros(num_experts, out_dim, sign_cov, dtype=torch.int8) if sign_cov > 0 else None
        )
        return cls(
            codebook, assignments, scales, n_blocks, block_size, out_dim, k, remainder, signs
        )

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Per-block contribution (n_blocks, tokens, out_dim). NO scale applied."""
        return self.block_pass(hidden_states, expert_idx)


class AdditiveCodebook(_CodebookBase):
    """Residual codebook (cb1+). Euclidean k-means on the real error. No scale."""

    @classmethod
    def empty(
        cls,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k: int,
        num_experts: int,
        rem: int = 0,
        sign_cov: int = 0,
    ) -> AdditiveCodebook:
        codebook = torch.zeros(n_blocks, block_size, k, dtype=torch.bfloat16)
        assignments = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.long)
        remainder = torch.zeros(num_experts, out_dim, rem, dtype=torch.bfloat16)
        signs = (
            torch.zeros(num_experts, out_dim, sign_cov, dtype=torch.int8) if sign_cov > 0 else None
        )
        return cls(codebook, assignments, n_blocks, block_size, out_dim, k, remainder, signs)

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Per-block contribution (n_blocks, tokens, out_dim). NO scale applied."""
        return self.block_pass(hidden_states, expert_idx)


class CodebookModule(nn.Module):
    """One SphericalCodebook + nn.ModuleList of AdditiveCodebooks.

    Each codebook carries its OWN block partition (n_blocks_c, block_size_c) and
    a raw remainder. Forward sums:
      - primary per-block pass scaled by the primary per-block scale, plus the
        primary raw remainder;
      - each residual per-block pass (unscaled, magnitude carried by centroids),
        plus its raw remainder.

    SSVQ sign bits are a PER-CODEBOOK concern: any codebook (primary or
    residual) may carry a ``signs`` buffer, which its per-block pass folds into
    the gathered centroids so the GPU forward matches ``reconstruct_codebooks``.
    """

    primary: SphericalCodebook
    additives: nn.ModuleList

    def __init__(
        self,
        primary: SphericalCodebook,
        additives: nn.ModuleList,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        n_codebooks: int,
    ):
        super().__init__()
        self.primary = primary
        self.additives = additives
        # Kept for back-compat / introspection (primary partition).
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.out_dim = out_dim
        self.n_codebooks = n_codebooks

    @classmethod
    def from_result(
        cls,
        result: CodebookResult,
        out_dim: int,
        n_blocks: int | None = None,
        block_size: int | None = None,
    ) -> CodebookModule:
        """Build a CodebookModule from a CodebookResult.

        Per-codebook block sizes, n_blocks, and remainders are derived from the
        result tensors. ``n_blocks``/``block_size`` args are accepted for
        back-compat but ignored (everything is derived).
        """
        n_rows = result.scales.shape[0]
        num_experts = n_rows // out_dim
        bs_list = result.bs_per_codebook()

        def _remainder(c: int) -> torch.Tensor | None:
            if result.remainders is None:
                return None
            rem = result.remainders[c]  # (n_rows, rem_c)
            if rem is None:
                return None
            return rem.reshape(num_experts, out_dim, rem.shape[-1]).contiguous()

        def _signs(c: int) -> torch.Tensor | None:
            if result.sign_bits is None:
                return None
            sb = result.sign_bits[c]  # (n_rows, cov_c) or None
            if sb is None:
                return None
            return sb.reshape(num_experts, out_dim, sb.shape[-1]).to(torch.int8).contiguous()

        def _assign_3d(c: int) -> torch.Tensor:
            asgn = result.assignments[c]  # (n_blocks_c, n_rows)
            n_blocks_c = asgn.shape[0]
            return asgn.reshape(n_blocks_c, num_experts, out_dim).permute(1, 0, 2).contiguous()

        n_blocks_0 = result.assignments[0].shape[0]
        # scales: (n_rows, n_blocks_0) -> (num_experts, n_blocks_0, out_dim)
        scales_3d = (
            result.scales.reshape(num_experts, out_dim, n_blocks_0).permute(0, 2, 1).contiguous()
        )

        primary = SphericalCodebook(
            codebook=result.codebooks[0],
            assignments=_assign_3d(0),
            scales=scales_3d,
            n_blocks=n_blocks_0,
            block_size=bs_list[0],
            out_dim=out_dim,
            k=result.codebooks[0].shape[-1],
            remainder=_remainder(0),
            signs=_signs(0),
        )
        additives = nn.ModuleList(
            [
                AdditiveCodebook(
                    codebook=result.codebooks[c],
                    assignments=_assign_3d(c),
                    n_blocks=result.assignments[c].shape[0],
                    block_size=bs_list[c],
                    out_dim=out_dim,
                    k=result.codebooks[c].shape[-1],
                    remainder=_remainder(c),
                    signs=_signs(c),
                )
                for c in range(1, result.n_codebooks)
            ]
        )
        return cls(primary, additives, n_blocks_0, bs_list[0], out_dim, result.n_codebooks)

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
    ) -> CodebookModule:
        """Create an empty (zero-filled) module sized for load_state_dict.

        ``n_blocks``/``block_size`` are the primary partition; per-codebook
        overrides are given via the *_list args (default: uniform).
        """
        n_cb = len(k_list)
        nbl = n_blocks_list or [n_blocks] * n_cb
        bsl = block_size_list or [block_size] * n_cb
        reml = rem_list or [0] * n_cb
        scl = sign_cov_list or [0] * n_cb
        primary = SphericalCodebook.empty(
            nbl[0], bsl[0], out_dim, k_list[0], num_experts, rem=reml[0], sign_cov=scl[0]
        )
        additives = nn.ModuleList(
            [
                AdditiveCodebook.empty(
                    nbl[c], bsl[c], out_dim, k_list[c], num_experts, rem=reml[c], sign_cov=scl[c]
                )
                for c in range(1, n_cb)
            ]
        )
        return cls(primary, additives, nbl[0], bsl[0], out_dim, n_cb)

    @classmethod
    def load(cls, path: str | Any) -> CodebookModule:
        """Load a CodebookModule saved via state_dict + meta."""
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
        )
        module.load_state_dict(data["state_dict"])
        return module

    def _codebooks(self) -> list[_CodebookBase]:
        return [self.primary, *list(self.additives)]

    def state_dict_with_meta(self) -> dict[str, Any]:
        """Return a serializable dict with state_dict + meta (for torch.save)."""
        cbs = self._codebooks()
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
                "num_experts": self.primary.assignments.shape[0],
            },
        }

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """y = scale_0 * primary_pass + sum_c residual_pass_c + remainders."""
        p = self.primary
        prim = p(hidden_states, expert_idx)  # (n_blocks_0, tokens, out_dim)
        scale = p.scales[expert_idx]  # (n_blocks_0, out_dim)
        out = (prim * scale.unsqueeze(1)).sum(dim=0)  # (tokens, out_dim)
        out = out + p.remainder_pass(hidden_states, expert_idx)
        for cb in self.additives:
            out = out + cb(hidden_states, expert_idx).sum(dim=0)
            out = out + cb.remainder_pass(hidden_states, expert_idx)
        return out


class _SwiGLU(nn.Module):
    """SwiGLU intermediate: act_fn(gate) * up. Exposed as a sub-module so it can be hooked."""

    def __init__(self, act_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self.act_fn(gate) * up


class CodebookExperts(nn.Module):
    """Drop-in replacement for Qwen3_5MoeExperts using codebook quantization.

    Each projection (gate, up, down) uses n_codebooks codebooks (residual) along
    the input axis, each split into n_blocks sub-codebooks (PQ):
      gate/up: codebook over hidden_size, output intermediate_size
      down:    codebook over intermediate_size, output hidden_size
    """

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

    @classmethod
    def from_codebook_results(
        cls,
        layer_results: dict[str, Any],
        num_experts: int,
        rows_per_expert_gate_up: int,
        rows_per_expert_down: int,
        hidden_size: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> CodebookExperts:
        """Build from CodebookResult objects.

        Per-codebook block sizes, n_blocks, K, and remainders are all derived
        from the CodebookResult tensors.

        Args:
            rows_per_expert_gate_up: out dim per expert for gate/up (= moe_intermediate_size)
            rows_per_expert_down: out dim per expert for down (= hidden_size)
        """
        intermediate_size = rows_per_expert_gate_up

        def _build(name: str, out_dim: int) -> CodebookModule:
            return CodebookModule.from_result(layer_results[name], out_dim=out_dim)

        gate = _build("gate", out_dim=intermediate_size)
        up = _build("up", out_dim=intermediate_size)
        down = _build("down", out_dim=hidden_size)

        return cls(
            gate=gate,
            up=up,
            down=down,
            num_experts=num_experts,
            act_fn=act_fn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass replacing the original Qwen3_5MoeExperts.

        Args:
            hidden_states: (total_tokens, hidden_size)
            top_k_index: (total_tokens, top_k) — expert indices per token
            top_k_weights: (total_tokens, top_k) — routing weights per token

        Returns:
            (total_tokens, hidden_size)
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, tokens)
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
