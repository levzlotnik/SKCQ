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
) -> torch.Tensor:
    """Compute Q @ gathered_centroids for each block.

    Returns (n_blocks, tokens, out_dim) — NO scale applied.
    """
    expert_assign = assignments[expert_idx]  # (n_blocks, out_dim)
    hidden_blocked = hidden_states.reshape(-1, n_blocks, block_size).permute(1, 0, 2)

    if forward_mode == "gather_matmul":
        gathered_cb = codebook.gather(
            dim=2,
            index=expert_assign.unsqueeze(1).expand(-1, block_size, -1),
        )
        logits = torch.bmm(hidden_blocked, gathered_cb)
    else:
        n_tokens = hidden_states.shape[0]
        logits = torch.bmm(hidden_blocked, codebook)
        logits = logits.gather(
            dim=2,
            index=expert_assign.unsqueeze(1).expand(-1, n_tokens, -1),
        )
    return logits  # (n_blocks, tokens, out_dim)


class SphericalCodebook(nn.Module):
    """Primary codebook (cb0). Spherical k-means. Stores direction codebook + per-row scale.

    The scale is re-fit to the final direction (sum of all codebooks' contributions).
    At inference, the scale multiplies the SUM of all codebook contributions.
    """

    codebook: nn.Parameter
    assignments: torch.Tensor
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
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.out_dim = out_dim
        self.k = k
        self.codebook = nn.Parameter(codebook)
        self.register_buffer("assignments", assignments)
        self.scales = nn.Parameter(scales)
        self.forward_mode: ForwardMode = _choose_forward_mode(k, out_dim)

    @classmethod
    def empty(
        cls, n_blocks: int, block_size: int, out_dim: int, k: int, num_experts: int
    ) -> SphericalCodebook:
        codebook = torch.zeros(n_blocks, block_size, k, dtype=torch.bfloat16)
        assignments = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.long)
        scales = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.bfloat16)
        return cls(codebook, assignments, scales, n_blocks, block_size, out_dim, k)

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Compute Q @ gathered_centroids for each block. Returns (n_blocks, tokens, out_dim).

        NO scale applied — the scale is applied by the owning CodebookModule.
        """
        return _codebook_pass(
            self.codebook,
            self.assignments,
            hidden_states,
            self.n_blocks,
            self.block_size,
            expert_idx,
            self.forward_mode,
        )


class AdditiveCodebook(nn.Module):
    """Residual codebook (cb1+). Euclidean k-means on unit-sphere residuals. No scale."""

    codebook: nn.Parameter
    assignments: torch.Tensor

    def __init__(
        self,
        codebook: torch.Tensor,
        assignments: torch.Tensor,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.out_dim = out_dim
        self.k = k
        self.codebook = nn.Parameter(codebook)
        self.register_buffer("assignments", assignments)
        self.forward_mode: ForwardMode = _choose_forward_mode(k, out_dim)

    @classmethod
    def empty(
        cls, n_blocks: int, block_size: int, out_dim: int, k: int, num_experts: int
    ) -> AdditiveCodebook:
        codebook = torch.zeros(n_blocks, block_size, k, dtype=torch.bfloat16)
        assignments = torch.zeros(num_experts, n_blocks, out_dim, dtype=torch.long)
        return cls(codebook, assignments, n_blocks, block_size, out_dim, k)

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Same as SphericalCodebook.forward — Q @ gathered_centroids per block.

        Returns (n_blocks, tokens, out_dim). NO scale applied.
        """
        return _codebook_pass(
            self.codebook,
            self.assignments,
            hidden_states,
            self.n_blocks,
            self.block_size,
            expert_idx,
            self.forward_mode,
        )


class CodebookModule(nn.Module):
    """One SphericalCodebook + nn.ModuleList of AdditiveCodebooks.

    Forward sums all codebook contributions per block, then applies the single
    scale (from the primary codebook) and sums over blocks (PQ).
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
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.out_dim = out_dim
        self.n_codebooks = n_codebooks

    @classmethod
    def from_result(
        cls, result: CodebookResult, n_blocks: int, block_size: int, out_dim: int
    ) -> CodebookModule:
        """Build a CodebookModule from a CodebookResult.

        CodebookResult stores assignments as (n_blocks, n_rows) and scales as
        (n_rows, n_blocks), but CodebookModule expects (num_experts, n_blocks, out_dim).
        """
        n_rows = result.scales.shape[0]
        num_experts = n_rows // out_dim

        # Reshape assignments: (n_blocks, n_rows) -> (n_blocks, num_experts, out_dim) -> (num_experts, n_blocks, out_dim)
        assigns_3d = []
        for asgn in result.assignments:
            a = asgn.reshape(n_blocks, num_experts, out_dim).permute(1, 0, 2).contiguous()
            assigns_3d.append(a)

        # Reshape scales: (n_rows, n_blocks) -> (num_experts, out_dim, n_blocks) -> (num_experts, n_blocks, out_dim)
        scales_3d = result.scales.reshape(num_experts, out_dim, n_blocks).permute(0, 2, 1).contiguous()

        primary = SphericalCodebook(
            codebook=result.codebooks[0],
            assignments=assigns_3d[0],
            scales=scales_3d,
            n_blocks=n_blocks,
            block_size=block_size,
            out_dim=out_dim,
            k=result.codebooks[0].shape[-1],
        )
        additives = nn.ModuleList(
            [
                AdditiveCodebook(
                    codebook=result.codebooks[c],
                    assignments=assigns_3d[c],
                    n_blocks=n_blocks,
                    block_size=block_size,
                    out_dim=out_dim,
                    k=result.codebooks[c].shape[-1],
                )
                for c in range(1, result.n_codebooks)
            ]
        )
        return cls(primary, additives, n_blocks, block_size, out_dim, result.n_codebooks)

    @classmethod
    def empty(
        cls,
        n_blocks: int,
        block_size: int,
        out_dim: int,
        k_list: list[int],
        num_experts: int,
    ) -> CodebookModule:
        """Create an empty (zero-filled) module sized for load_state_dict."""
        primary = SphericalCodebook.empty(n_blocks, block_size, out_dim, k_list[0], num_experts)
        additives = nn.ModuleList(
            [
                AdditiveCodebook.empty(n_blocks, block_size, out_dim, k_list[c], num_experts)
                for c in range(1, len(k_list))
            ]
        )
        return cls(primary, additives, n_blocks, block_size, out_dim, len(k_list))

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
        )
        module.load_state_dict(data["state_dict"])
        return module

    def state_dict_with_meta(self) -> dict[str, Any]:
        """Return a serializable dict with state_dict + meta (for torch.save)."""
        return {
            "state_dict": self.state_dict(),
            "meta": {
                "n_blocks": self.n_blocks,
                "block_size": self.block_size,
                "out_dim": self.out_dim,
                "n_codebooks": self.n_codebooks,
                "k_list": [self.primary.k]
                + [cb.k for cb in self.additives],
                "num_experts": self.primary.assignments.shape[0],
            },
        }

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """y = scale_0 * sum_c (Q @ centroid_c[assign_c]) summed over blocks (PQ)."""
        logits = self.primary(hidden_states, expert_idx)  # (n_blocks, tokens, out_dim)
        for cb in self.additives:
            logits = logits + cb(hidden_states, expert_idx)
        scale = self.primary.scales[expert_idx]  # (n_blocks, out_dim)
        return (logits * scale.unsqueeze(1)).sum(dim=0)  # (tokens, out_dim)


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

        n_blocks, n_codebooks, and per-codebook K are all derived from the
        CodebookResult tensors. Forward mode is auto-selected per codebook based
        on K vs out_dim.

        Args:
            rows_per_expert_gate_up: out dim per expert for gate/up (= moe_intermediate_size)
            rows_per_expert_down: out dim per expert for down (= hidden_size)
        """
        intermediate_size = rows_per_expert_gate_up

        def _build(name: str, out_dim: int, in_dim: int) -> CodebookModule:
            result = layer_results[name]
            block_size = in_dim // result.n_blocks
            return CodebookModule.from_result(
                result, n_blocks=result.n_blocks, block_size=block_size, out_dim=out_dim
            )

        gate = _build("gate", out_dim=intermediate_size, in_dim=hidden_size)
        up = _build("up", out_dim=intermediate_size, in_dim=hidden_size)
        down = _build("down", out_dim=hidden_size, in_dim=intermediate_size)

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
