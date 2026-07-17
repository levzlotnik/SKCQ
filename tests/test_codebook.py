from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from skcq.clustering import CodebookResult, build_codebook
from skcq.codebook_experts import (
    AdditiveCodebook,
    CodebookModule,
    SphericalCodebook,
)
from skcq.config import CodebookParams

# Tolerated dimensions for the smoke fixtures (small + fast).
NUM_EXPERTS = 4
OUT_DIM = 8
IN_DIM = 12
N_BLOCKS = 3
BLOCK_SIZE = IN_DIM // N_BLOCKS
K = 16
N_CODEBOOKS = 2


def _make_fake_module(
    n_codebooks: int = N_CODEBOOKS,
    n_blocks: int = N_BLOCKS,
    block_size: int = BLOCK_SIZE,
    k_list: list[int] | None = None,
    num_experts: int = NUM_EXPERTS,
    out_dim: int = OUT_DIM,
) -> CodebookModule:
    if k_list is None:
        k_list = [K] * n_codebooks
    if len(k_list) != n_codebooks:
        raise ValueError("k_list length must equal n_codebooks")

    primary_codebook = torch.randn(n_blocks, block_size, k_list[0])
    primary_assign = torch.randint(0, k_list[0], (num_experts, n_blocks, out_dim))
    scales = torch.randn(num_experts, n_blocks, out_dim)
    primary = SphericalCodebook(
        primary_codebook, primary_assign, scales, n_blocks, block_size, out_dim, k_list[0]
    )

    additives = nn.ModuleList()
    for c in range(1, n_codebooks):
        cb = torch.randn(n_blocks, block_size, k_list[c])
        asgn = torch.randint(0, k_list[c], (num_experts, n_blocks, out_dim))
        additives.append(AdditiveCodebook(cb, asgn, n_blocks, block_size, out_dim, k_list[c]))

    return CodebookModule(primary, additives, n_blocks, block_size, out_dim, n_codebooks)


def _set_forward_mode(module: CodebookModule, mode: str) -> None:
    module.primary.forward_mode = mode  # type: ignore[assignment]
    for cb in module.additives:
        cb.forward_mode = mode  # type: ignore[assignment]


def _codebook_contrib(
    cb: SphericalCodebook | AdditiveCodebook,
    hidden_blocked: torch.Tensor,
    expert_idx: int,
) -> torch.Tensor:
    """Per-block contribution (n_blocks, tokens, out_dim) — NO scale."""
    n_tokens = hidden_blocked.shape[0]
    out = torch.zeros(cb.n_blocks, n_tokens, cb.out_dim)
    for b in range(cb.n_blocks):
        hb = hidden_blocked[:, b, :]  # (tokens, block_size)
        cb_b = cb.codebook[b]  # (block_size, K)
        logits = hb @ cb_b  # (tokens, K)
        assign = cb.assignments[expert_idx, b]  # (out_dim,)
        out[b] = logits[:, assign]
    return out


def _manual_forward(module: CodebookModule, hidden: torch.Tensor, expert_idx: int) -> torch.Tensor:
    """Reference: sum codebook contributions per block, apply scale, sum over blocks (PQ)."""
    n_tokens = hidden.shape[0]
    hidden_blocked = hidden.reshape(n_tokens, module.n_blocks, module.block_size)
    logits = torch.zeros(module.n_blocks, n_tokens, module.out_dim)
    logits = logits + _codebook_contrib(module.primary, hidden_blocked, expert_idx)
    for cb in module.additives:
        logits = logits + _codebook_contrib(cb, hidden_blocked, expert_idx)
    scale = module.primary.scales[expert_idx]  # (n_blocks, out_dim)
    return (logits * scale.unsqueeze(1)).sum(dim=0)


class TestCodebookModuleForward:
    def test_output_shape(self) -> None:
        mod = _make_fake_module()
        hidden = torch.randn(5, IN_DIM)
        out = mod(hidden, expert_idx=2)
        assert out.shape == (5, OUT_DIM)

    def test_matches_manual_reconstruction(self) -> None:
        torch.manual_seed(0)
        mod = _make_fake_module()
        hidden = torch.randn(7, IN_DIM)
        actual = mod(hidden, expert_idx=1)
        expected = _manual_forward(mod, hidden, expert_idx=1)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_single_codebook_matches_manual(self) -> None:
        torch.manual_seed(1)
        mod = _make_fake_module(n_codebooks=1)
        hidden = torch.randn(4, IN_DIM)
        actual = mod(hidden, expert_idx=0)
        expected = _manual_forward(mod, hidden, expert_idx=0)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_single_block_matches_plain_matmul(self) -> None:
        """n_blocks=1 collapses to plain (tokens, K) gather + scale — no bmm needed."""
        torch.manual_seed(2)
        n_blocks = 1
        block_size = IN_DIM
        mod = _make_fake_module(n_blocks=n_blocks, block_size=block_size, n_codebooks=1)
        hidden = torch.randn(3, IN_DIM)
        expert_idx = 2
        actual = mod(hidden, expert_idx=expert_idx)

        # Reference: out = (hidden @ codebook[0,0])[:, assign] * scale
        logits = hidden @ mod.primary.codebook[0]  # (tokens, K)
        expected = (
            logits[:, mod.primary.assignments[expert_idx, 0]] * mod.primary.scales[expert_idx, 0]
        )
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_expert_isolation(self) -> None:
        """Different expert indices use different assignments/scales."""
        torch.manual_seed(3)
        mod = _make_fake_module()
        hidden = torch.randn(6, IN_DIM)
        out_a = mod(hidden, expert_idx=0)
        out_b = mod(hidden, expert_idx=1)
        assert not torch.allclose(out_a, out_b)


class TestBuildCodebook:
    def _params(self, **overrides: object) -> CodebookParams:
        defaults: dict[str, object] = dict(
            k_gate=K,
            k_up=K,
            k_down=K,
            n_blocks_gate_up=N_BLOCKS,
            n_blocks_down=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            max_iters=10,
            norm_threshold=1e-9,
            skip_zeros=False,
        )
        defaults.update(overrides)
        return CodebookParams(**defaults)  # type: ignore[arg-type]

    def test_output_shapes(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        result = build_codebook(
            rows,
            params=self._params(),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="test",
        )
        assert len(result.codebooks) == N_CODEBOOKS
        assert len(result.assignments) == N_CODEBOOKS
        for c in range(N_CODEBOOKS):
            assert result.codebooks[c].shape == (N_BLOCKS, BLOCK_SIZE, K)
            assert result.assignments[c].shape == (NUM_EXPERTS, N_BLOCKS, OUT_DIM)
            assert result.assignments[c].dtype == torch.long
        assert result.scales.shape == (NUM_EXPERTS, N_BLOCKS, OUT_DIM)
        assert result.zero_mask.shape == (NUM_EXPERTS * OUT_DIM,)
        assert result.n_blocks == N_BLOCKS
        assert result.n_codebooks == N_CODEBOOKS

    def test_in_dim_not_divisible_raises(self) -> None:
        rows = torch.randn(8, 13)  # 13 not divisible by 3
        try:
            build_codebook(
                rows,
                params=self._params(),
                k=K,
                n_blocks=3,
                n_codebooks=1,
                num_experts=1,
                out_dim=8,
                device=torch.device("cpu"),
                name="bad",
            )
        except ValueError as e:
            assert "not divisible" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def _reconstruct_error(
        self, result: CodebookResult, rows: torch.Tensor, expert_idx: int
    ) -> float:
        n_rows = NUM_EXPERTS * OUT_DIM
        w = rows.reshape(NUM_EXPERTS, OUT_DIM, IN_DIM)
        recon = torch.zeros(OUT_DIM, IN_DIM)
        for b in range(result.n_blocks):
            final_dir = torch.zeros(n_rows, BLOCK_SIZE)
            for c in range(result.n_codebooks):
                cb = result.codebooks[c][b]
                asgn = result.assignments[c][:, b, :].reshape(-1)
                final_dir = final_dir + cb.t()[asgn]
            scale = result.scales[:, b, :].reshape(-1)
            recon_block = scale.unsqueeze(-1) * final_dir
            recon[:, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE] = (
                recon[:, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
                + recon_block.reshape(NUM_EXPERTS, OUT_DIM, BLOCK_SIZE)[expert_idx]
            )
        return (w[expert_idx] - recon).norm().item()

    def test_residual_reduces_error(self) -> None:
        torch.manual_seed(42)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params1 = self._params(n_codebooks=1)
        params2 = self._params(n_codebooks=2)

        r1 = build_codebook(
            rows,
            params1,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=1,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="1cb",
        )
        r2 = build_codebook(
            rows,
            params2,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="2cb",
        )
        err1 = self._reconstruct_error(r1, rows, 1)
        err2 = self._reconstruct_error(r2, rows, 1)
        assert err2 < err1, f"residual should reduce error: {err2} >= {err1}"

    def test_zero_rows_excluded(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        rows[0] = 0  # zero out expert 0's first output row
        result = build_codebook(
            rows,
            params=self._params(norm_threshold=1e-5, skip_zeros=True),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=1,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="zeros",
        )
        # zero_mask flags the first row (expert 0, out_idx 0)
        assert result.zero_mask[0]
        # Scale for that row across all blocks should be zero
        assert result.scales[0, :, 0].sum() == 0
        # Other rows should generally have non-zero scales
        assert result.scales[0, :, 1].abs().sum() > 0

    def test_unit_sphere_residual(self) -> None:
        """cb1 operates on the unit-sphere residual (unit - centroid_0), not raw residual."""
        torch.manual_seed(0)
        # Large magnitude so raw-residual norms clearly exceed unit-sphere residual norms.
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 10.0
        params = self._params(n_codebooks=2, max_iters=50)
        result = build_codebook(
            rows,
            params,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="usr",
        )
        n_rows = NUM_EXPERTS * OUT_DIM
        unit = F.normalize(rows.float(), dim=-1)
        unit_blocks = unit.reshape(n_rows, N_BLOCKS, BLOCK_SIZE)
        raw_blocks = rows.float().reshape(n_rows, N_BLOCKS, BLOCK_SIZE)

        new_res = unit_blocks.clone()
        raw_res = raw_blocks.clone()
        for b in range(N_BLOCKS):
            cb0 = result.codebooks[0][b]  # (BLOCK_SIZE, K)
            asgn0 = result.assignments[0][:, b, :].reshape(-1)  # (n_rows,)
            centroid0 = cb0.t()[asgn0]  # (n_rows, BLOCK_SIZE) — unit-norm (spherical)
            # New scheme: residual = unit - centroid_0
            new_res[:, b, :] = new_res[:, b, :] - centroid0
            # Old scheme: residual = raw - scale_0 * centroid_0
            scale0 = (raw_blocks[:, b, :] * centroid0).sum(dim=-1)
            raw_res[:, b, :] = raw_res[:, b, :] - scale0.unsqueeze(-1) * centroid0

        # cb0 centroids must be unit-norm (spherical k-means)
        for b in range(N_BLOCKS):
            norms = result.codebooks[0][b].t().norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

        # cb1 should reconstruct the unit-sphere residual, NOT the raw residual
        new_err = 0.0
        raw_err = 0.0
        for b in range(N_BLOCKS):
            cb1 = result.codebooks[1][b]  # (BLOCK_SIZE, K1)
            asgn1 = result.assignments[1][:, b, :].reshape(-1)
            centroid1 = cb1.t()[asgn1]
            new_err += (new_res[:, b, :] - centroid1).norm(dim=-1).mean().item()
            raw_err += (raw_res[:, b, :] - centroid1).norm(dim=-1).mean().item()
        assert new_err < raw_err / 5, (
            f"cb1 should fit unit-sphere residual: new_err={new_err} raw_err={raw_err}"
        )

    def test_asymmetric_k(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params = self._params(n_codebooks=2, k_residual_mult=2.0)
        result = build_codebook(
            rows,
            params,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="asym",
        )
        assert result.codebooks[0].shape[-1] == K
        assert result.codebooks[1].shape[-1] == int(K / 2)
        # assignments per codebook sized to (n_blocks, n_rows)
        assert result.assignments[0].shape == (N_BLOCKS, NUM_EXPERTS * OUT_DIM)
        assert result.assignments[1].shape == (N_BLOCKS, NUM_EXPERTS * OUT_DIM)

    def test_single_scale(self) -> None:
        """Only cb0 has scales — CodebookResult.scales is a single tensor, not per-codebook."""
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        result = build_codebook(
            rows,
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="sscale",
        )
        assert isinstance(result.scales, torch.Tensor)
        assert result.scales.ndim == 3
        assert result.scales.shape == (NUM_EXPERTS, N_BLOCKS, OUT_DIM)

    def test_scale_refit(self) -> None:
        """scale = dot(raw, final_direction)/||final_direction||^2 (not just cb0 projection)."""
        torch.manual_seed(7)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        result = build_codebook(
            rows,
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="refit",
        )
        n_rows = NUM_EXPERTS * OUT_DIM
        raw_blocks = rows.float().reshape(n_rows, N_BLOCKS, BLOCK_SIZE)
        for b in range(N_BLOCKS):
            final_dir = torch.zeros(n_rows, BLOCK_SIZE)
            for c in range(result.n_codebooks):
                cb = result.codebooks[c][b]
                asgn = result.assignments[c][:, b, :].reshape(-1)
                final_dir = final_dir + cb.t()[asgn]
            expected = (raw_blocks[:, b, :] * final_dir).sum(dim=-1) / (
                final_dir.norm(dim=-1) ** 2 + 1e-10
            )
            actual = result.scales[:, b, :].reshape(-1)
            assert torch.allclose(actual.float(), expected, atol=1e-4)

        # Also confirm the refit scale differs from the naive cb0-only projection scale,
        # i.e. the scale actually accounts for the final (multi-codebook) direction.
        naive_scale = torch.zeros(n_rows)
        for b in range(N_BLOCKS):
            cb0 = result.codebooks[0][b]
            asgn0 = result.assignments[0][:, b, :].reshape(-1)
            c0 = cb0.t()[asgn0]
            naive_scale = naive_scale + (raw_blocks[:, b, :] * c0).sum(dim=-1)
        actual_flat = result.scales.permute(1, 0, 2).reshape(N_BLOCKS, n_rows).sum(dim=0)
        assert not torch.allclose(actual_flat.float(), naive_scale, atol=1e-4)


class TestCodebookParameters:
    def test_codebooks_are_parameters(self) -> None:
        mod = _make_fake_module()
        assert isinstance(mod.primary.codebook, nn.Parameter)
        assert isinstance(mod.primary.scales, nn.Parameter)
        assert isinstance(mod.additives[0].codebook, nn.Parameter)
        # assignments are buffers (not Parameters), int64
        assert not isinstance(mod.primary.assignments, nn.Parameter)
        assert mod.primary.assignments.dtype == torch.long
        assert not isinstance(mod.additives[0].assignments, nn.Parameter)


class TestForwardMatchesFormula:
    def test_forward_matches_formula(self) -> None:
        """Forward output matches scale_0 * sum_c(Q @ centroid_c[assign_c]) computed manually."""
        torch.manual_seed(11)
        mod = _make_fake_module(k_list=[K, K // 2])
        hidden = torch.randn(5, IN_DIM)
        expert_idx = 1
        actual = mod(hidden, expert_idx)

        n_tokens = hidden.shape[0]
        hidden_blocked = hidden.reshape(n_tokens, mod.n_blocks, mod.block_size)
        total = torch.zeros(n_tokens, mod.out_dim)
        for b in range(mod.n_blocks):
            hb = hidden_blocked[:, b, :]
            block_contrib = torch.zeros(n_tokens, mod.out_dim)
            l0 = hb @ mod.primary.codebook[b]
            block_contrib = block_contrib + l0[:, mod.primary.assignments[expert_idx, b]]
            for cb_mod in mod.additives:
                lc = hb @ cb_mod.codebook[b]
                block_contrib = block_contrib + lc[:, cb_mod.assignments[expert_idx, b]]
            total = total + block_contrib * mod.primary.scales[expert_idx, b]
        assert torch.allclose(actual, total, atol=1e-5)


class TestStateDictRoundTrip:
    def test_state_dict_roundtrip(self) -> None:
        """Create CodebookModule, save state_dict, load, verify forward gives same output.

        Empty modules are bf16, so the module + hidden are cast to bf16 for an exact
        (bit-for-bit) roundtrip — matching the production save/load path.
        """
        torch.manual_seed(13)
        mod = _make_fake_module(k_list=[K, K // 2]).to(torch.bfloat16)
        hidden = torch.randn(6, IN_DIM).to(torch.bfloat16)
        expert_idx = 2
        out_before = mod(hidden, expert_idx)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "module.pt"
            torch.save(mod.state_dict_with_meta(), p)
            loaded = CodebookModule.load(p)
        out_after = loaded(hidden, expert_idx)
        assert out_before.shape == out_after.shape
        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestForwardModesEquivalent:
    def _make(self, k_list: list[int], n_blocks: int, block_size: int) -> CodebookModule:
        num_experts = 4
        out_dim = 8
        return _make_fake_module(
            n_codebooks=len(k_list),
            n_blocks=n_blocks,
            block_size=block_size,
            k_list=k_list,
            num_experts=num_experts,
            out_dim=out_dim,
        )

    def test_modes_produce_identical_output(self) -> None:
        """matmul_gather and gather_matmul must produce identical results."""
        torch.manual_seed(42)
        in_dim = 3 * 4
        mod_mg = self._make(k_list=[16, 8], n_blocks=3, block_size=4)
        mod_gm = self._make(k_list=[16, 8], n_blocks=3, block_size=4)
        # Same tensors for both
        mod_gm.load_state_dict(mod_mg.state_dict())
        _set_forward_mode(mod_mg, "matmul_gather")
        _set_forward_mode(mod_gm, "gather_matmul")

        hidden = torch.randn(5, in_dim)
        out_mg = mod_mg(hidden, expert_idx=1)
        out_gm = mod_gm(hidden, expert_idx=1)
        assert torch.allclose(out_mg, out_gm, atol=1e-6)

    def test_modes_match_manual(self) -> None:
        """Both modes match the manual reference implementation."""
        torch.manual_seed(99)
        in_dim = 2 * 4
        mod = self._make(k_list=[8, 4], n_blocks=2, block_size=4)
        hidden = torch.randn(4, in_dim)

        for mode in ["matmul_gather", "gather_matmul"]:
            m = self._make(k_list=[8, 4], n_blocks=2, block_size=4)
            m.load_state_dict(mod.state_dict())
            _set_forward_mode(m, mode)
            actual = m(hidden, expert_idx=0)
            expected = _manual_forward(m, hidden, expert_idx=0)
            assert torch.allclose(actual, expected, atol=1e-5), f"mismatch in {mode}"

    def test_modes_equivalent_large_k(self) -> None:
        """Modes equivalent when K >> out_dim (the case gather_matmul is designed for)."""
        torch.manual_seed(7)
        in_dim = 1 * 8
        mod_mg = self._make(k_list=[256], n_blocks=1, block_size=8)
        mod_gm = self._make(k_list=[256], n_blocks=1, block_size=8)
        mod_gm.load_state_dict(mod_mg.state_dict())
        _set_forward_mode(mod_mg, "matmul_gather")
        _set_forward_mode(mod_gm, "gather_matmul")

        hidden = torch.randn(10, in_dim)
        out_mg = mod_mg(hidden, expert_idx=0)
        out_gm = mod_gm(hidden, expert_idx=0)
        assert torch.allclose(out_mg, out_gm, atol=1e-6)

    def test_build_then_forward(self) -> None:
        """End-to-end: build codebook from weights, construct module, forward matches."""
        torch.manual_seed(7)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params = CodebookParams(
            k_gate=K,
            n_blocks_gate_up=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            max_iters=10,
            norm_threshold=1e-9,
            skip_zeros=False,
        )
        result = build_codebook(
            rows,
            params,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="int",
        )
        mod = CodebookModule.from_result(
            result, n_blocks=result.n_blocks, block_size=BLOCK_SIZE, out_dim=OUT_DIM
        )
        hidden = torch.randn(5, IN_DIM)
        out = mod(hidden, expert_idx=1)
        assert out.shape == (5, OUT_DIM)

        # Compare against manual forward using the *built* codebook
        expected = _manual_forward(mod, hidden, expert_idx=1)
        assert torch.allclose(out, expected, atol=1e-4)
