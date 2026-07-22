from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from skcq.clustering import CodebookResult
from skcq.experiment import CodebookConfig, CodebookExperiment
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
    """Reference (new scheme): primary scaled per block, residuals unscaled, + remainders."""
    n_tokens = hidden.shape[0]
    p = module.primary
    cov_p = p.n_blocks * p.block_size
    hp = hidden[:, :cov_p].reshape(n_tokens, p.n_blocks, p.block_size)
    plogits = _codebook_contrib(p, hp, expert_idx)  # (n_blocks, tokens, out_dim)
    scale = p.scales[expert_idx]  # (n_blocks, out_dim)
    out = (plogits * scale.unsqueeze(1)).sum(dim=0)
    out = out + hidden[:, cov_p:] @ p.remainder[expert_idx].mT
    for cb in module.additives:
        cov_c = cb.n_blocks * cb.block_size
        hc = hidden[:, :cov_c].reshape(n_tokens, cb.n_blocks, cb.block_size)
        out = out + _codebook_contrib(cb, hc, expert_idx).sum(dim=0)
        out = out + hidden[:, cov_c:] @ cb.remainder[expert_idx].mT
    return out


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
        result = CodebookExperiment(CodebookConfig(
            params=self._params(),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="test",
        )).fit(rows)
        assert len(result.codebooks) == N_CODEBOOKS
        assert len(result.assignments) == N_CODEBOOKS
        n_rows = NUM_EXPERTS * OUT_DIM
        for c in range(N_CODEBOOKS):
            assert result.codebooks[c].shape == (N_BLOCKS, BLOCK_SIZE, K)
            assert result.assignments[c].shape == (N_BLOCKS, n_rows)
            assert result.assignments[c].dtype == torch.long
        assert result.scales.shape == (n_rows, N_BLOCKS)
        assert result.zero_mask.shape == (n_rows,)
        assert result.n_blocks == N_BLOCKS
        assert result.n_codebooks == N_CODEBOOKS

    def test_in_dim_not_divisible_builds_remainder(self) -> None:
        """Non-dividing primary block size now builds a remainder (no raise)."""
        rows = torch.randn(8, 13)  # 13 not divisible by 3 -> bs_0=4, cov=12, rem=1
        result = CodebookExperiment(CodebookConfig(
            params=self._params(),
            k=K,
            n_blocks=3,
            n_codebooks=1,
            num_experts=1,
            out_dim=8,
            device=torch.device("cpu"),
            name="rem",
        )).fit(rows)
        assert result.block_sizes == [4]
        assert result.remainders is not None
        assert result.remainders[0] is not None
        assert result.remainders[0].shape == (8, 1)
        recon = result.reconstruct()
        assert recon.shape == (8, 13)
        # Remainder column reconstructed exactly.
        assert torch.allclose(recon[:, 12:].float(), rows[:, 12:].float(), atol=1e-2)

    def _reconstruct_error(
        self, result: CodebookResult, rows: torch.Tensor, expert_idx: int
    ) -> float:
        recon = result.reconstruct()  # (n_rows, in_dim)
        in_dim = rows.shape[1]
        w = rows.reshape(NUM_EXPERTS, OUT_DIM, in_dim)
        r = recon.reshape(NUM_EXPERTS, OUT_DIM, in_dim)
        return (w[expert_idx].float() - r[expert_idx]).norm().item()

    def test_residual_reduces_error(self) -> None:
        torch.manual_seed(42)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params1 = self._params(n_codebooks=1)
        params2 = self._params(n_codebooks=2)

        r1 = CodebookExperiment(CodebookConfig(
            params=params1,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=1,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="1cb",
        )).fit(rows)
        r2 = CodebookExperiment(CodebookConfig(
            params=params2,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="2cb",
        )).fit(rows)
        err1 = self._reconstruct_error(r1, rows, 1)
        err2 = self._reconstruct_error(r2, rows, 1)
        assert err2 < err1, f"residual should reduce error: {err2} >= {err1}"

    def test_zero_rows_excluded(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        rows[0] = 0  # zero out expert 0's first output row
        result = CodebookExperiment(CodebookConfig(
            params=self._params(norm_threshold=1e-5, skip_zeros=True),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=1,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="zeros",
        )).fit(rows)
        # zero_mask flags the first row (expert 0, out_idx 0)
        assert result.zero_mask[0]
        # Scale for that row across all blocks should be zero
        assert result.scales[0, :].sum() == 0
        # Other rows should generally have non-zero scales
        assert result.scales[1, :].abs().sum() > 0

    def test_real_error_residual(self) -> None:
        """cb1 operates on the REAL error (W - recon_0), euclidean, magnitude included."""
        torch.manual_seed(0)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 10.0
        params = self._params(n_codebooks=2, max_iters=50)
        result = CodebookExperiment(CodebookConfig(
            params=params,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="real",
        )).fit(rows)
        n_rows = NUM_EXPERTS * OUT_DIM
        raw_blocks = rows.float().reshape(n_rows, N_BLOCKS, BLOCK_SIZE)
        unit = F.normalize(rows.float(), dim=-1)
        unit_blocks = unit.reshape(n_rows, N_BLOCKS, BLOCK_SIZE)

        # Primary reconstruction (from result's primary centroids + scales).
        primary_recon = torch.zeros(n_rows, N_BLOCKS, BLOCK_SIZE)
        for b in range(N_BLOCKS):
            cb0 = result.codebooks[0][b]  # (BLOCK_SIZE, K)
            dir0 = cb0.t()[result.assignments[0][b]]  # (n_rows, BLOCK_SIZE)
            primary_recon[:, b, :] = result.scales[:, b].unsqueeze(-1) * dir0

        real_error = raw_blocks - primary_recon  # E_1 = W - recon_0
        unit_residual = unit_blocks - primary_recon / (primary_recon.norm() + 1e-9)

        # cb1 centroids should fit the REAL error, not a unit-sphere residual.
        new_err = 0.0
        unit_err = 0.0
        for b in range(N_BLOCKS):
            cb1 = result.codebooks[1][b]
            centroid1 = cb1.t()[result.assignments[1][b]]
            new_err += (real_error[:, b, :] - centroid1).norm(dim=-1).mean().item()
            unit_err += (unit_residual[:, b, :] - centroid1).norm(dim=-1).mean().item()
        assert new_err < unit_err, (
            f"cb1 should fit real error: new_err={new_err} unit_err={unit_err}"
        )
        # Euclidean centroids carry magnitude (not unit-sphere ~O(1)).
        assert result.codebooks[1].norm(dim=1).mean().item() > 1.0

    def test_asymmetric_k(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params = self._params(n_codebooks=2, residual_k=8)
        result = CodebookExperiment(CodebookConfig(
            params=params,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="asym",
        )).fit(rows)
        assert result.codebooks[0].shape[-1] == K
        assert result.codebooks[1].shape[-1] == 8  # residual_k=8
        # assignments per codebook sized to (n_blocks, n_rows)
        assert result.assignments[0].shape == (N_BLOCKS, NUM_EXPERTS * OUT_DIM)
        assert result.assignments[1].shape == (N_BLOCKS, NUM_EXPERTS * OUT_DIM)

    def test_single_scale(self) -> None:
        """Only cb0 has scales — CodebookResult.scales is a single tensor, not per-codebook."""
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        result = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="sscale",
        )).fit(rows)
        assert isinstance(result.scales, torch.Tensor)
        assert result.scales.ndim == 2
        assert result.scales.shape == (NUM_EXPERTS * OUT_DIM, N_BLOCKS)

    def test_scale_refit(self) -> None:
        """Re-fit scale is the LS optimum: residual is orthogonal to the primary direction."""
        torch.manual_seed(7)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        result = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="refit",
        )).fit(rows)
        n_rows = NUM_EXPERTS * OUT_DIM
        raw = rows.float()
        recon = result.reconstruct()
        for b in range(N_BLOCKS):
            cols = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
            dir_b = result.codebooks[0][b].t()[result.assignments[0][b]]  # (n_rows, bs)
            resid = raw[:, cols] - recon[:, cols]
            # LS optimum => residual orthogonal to primary direction.
            dot = torch.einsum("nd,nd->n", resid, dir_b)
            assert dot.abs().max().item() < 1e-3, f"block {b} not orthogonal: {dot.abs().max()}"
        assert result.scales.shape == (n_rows, N_BLOCKS)

    def test_residual_bs_not_dividing_reconstructs_remainder(self) -> None:
        """A residual block size that does not divide in_dim reconstructs its remainder."""
        torch.manual_seed(3)
        in_dim = 12  # primary bs=4 divides; residual bs=5 -> cov=10, rem=2
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, in_dim) * 0.1
        params = self._params(n_codebooks=2, residual_block_sizes=5)
        result = CodebookExperiment(CodebookConfig(
            params=params,
            k=K,
            n_blocks=3,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="resrem",
        )).fit(rows)
        assert result.block_sizes == [4, 5]
        assert result.remainders is not None
        assert result.remainders[1] is not None
        assert result.remainders[1].shape == (NUM_EXPERTS * OUT_DIM, 2)
        recon = result.reconstruct()  # must not raise
        assert recon.shape == (NUM_EXPERTS * OUT_DIM, in_dim)

    def test_noncommensurate_block_sizes(self) -> None:
        """Non-commensurate primary/residual block sizes build + reconstruct (no IndexError)."""
        torch.manual_seed(5)
        in_dim = 34  # bs_0=10 -> cov0=30 rem0=4 ; bs_1=12 -> cov1=24 rem1=10
        n_rows = NUM_EXPERTS * OUT_DIM
        rows = torch.randn(n_rows, in_dim) * 0.1
        params = self._params(n_codebooks=2, residual_block_sizes=12)
        result = CodebookExperiment(CodebookConfig(
            params=params,
            k=K,
            n_blocks=0,  # ignored (primary_block_size given)
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="noncomm",
            primary_block_size=10,
        )).fit(rows)
        assert result.block_sizes == [10, 12]
        recon = result.reconstruct()
        assert recon.shape == (n_rows, in_dim)
        assert torch.isfinite(recon).all()

        # Reference reconstruction implementing the documented formula.
        ref = torch.zeros(n_rows, in_dim)
        # primary (bs=10, cov=30)
        for b in range(3):
            d = result.codebooks[0][b].t()[result.assignments[0][b]]
            ref[:, b * 10 : (b + 1) * 10] += result.scales[:, b].unsqueeze(-1) * d
        assert result.remainders is not None and result.remainders[0] is not None
        ref[:, 30:] += result.remainders[0].float()
        # residual (bs=12, cov=24)
        for b in range(2):
            ref[:, b * 12 : (b + 1) * 12] += result.codebooks[1][b].t()[result.assignments[1][b]]
        assert result.remainders[1] is not None
        ref[:, 24:] += result.remainders[1].float()
        assert torch.allclose(ref, recon, atol=1e-4)

        # Error is finite and better than the zero reconstruction.
        err = (rows.float() - recon).norm().item()
        assert err < rows.float().norm().item()

    def test_reproduction_failing_config(self) -> None:
        """Regression: shared + SSVQ + cb8, non-commensurate cb0 bs=10 / cb1 bs=12.

        This is the config that previously crashed with IndexError in reconstruct.
        """
        torch.manual_seed(0)
        num_experts = 2
        out_dim = 150
        in_dim = 2048  # divisible by neither 10 nor 12
        rows = torch.randn(num_experts * out_dim, in_dim) * 0.1
        params = CodebookParams(
            k_gate=256,
            n_blocks_gate_up=1,
            n_codebooks=2,
            max_iters=3,
            norm_threshold=1e-9,
            skip_zeros=False,
            residual_k=16,
            residual_block_sizes=12,
        )
        result = CodebookExperiment(CodebookConfig(
            params=params,
            k=256,
            n_blocks=0,
            n_codebooks=2,
            num_experts=num_experts,
            out_dim=out_dim,
            device=torch.device("cpu"),
            name="repro",
            shared_codebook=True,
            sign_split=True,
            codebook_bits=8,
            residual_block_sizes=12,
            primary_block_size=10,
        )).fit(rows)
        recon = result.reconstruct()  # must NOT raise IndexError
        assert recon.shape == (num_experts * out_dim, in_dim)
        err = (rows.float() - recon).norm().item()
        assert err == err and err != float("inf")  # finite
        assert err < rows.float().norm().item()


def _ref_reconstruct(result: CodebookResult) -> torch.Tensor:
    """Independent (non-shared) reference reconstruction applying per-codebook signs.

    Mirrors the documented scheme: primary = signs_0 ⊙ (scale_0 ⊙ dir_0); each
    residual = signs_c ⊙ centroids_c[assign_c]; per-codebook raw remainder added.
    """
    n_rows = result.scales.shape[0]
    bs_list = result.bs_per_codebook()
    in_dim = result.in_dim()
    recon = torch.zeros(n_rows, in_dim)
    for c in range(result.n_codebooks):
        bs_c = bs_list[c]
        n_blocks_c = in_dim // bs_c
        cov_c = n_blocks_c * bs_c
        buf = torch.zeros(n_rows, cov_c)
        for b in range(n_blocks_c):
            cb = result.codebooks[c][b]  # (bs_c, K)
            centroid = cb.t()[result.assignments[c][b]]  # (n_rows, bs_c)
            if c == 0:
                centroid = result.scales[:, b].unsqueeze(-1) * centroid
            buf[:, b * bs_c : (b + 1) * bs_c] = centroid
        if result.sign_bits is not None and result.sign_bits[c] is not None:
            sc = result.sign_bits[c]
            assert sc is not None
            buf = buf * sc.reshape(n_rows, cov_c).float()
        recon[:, :cov_c] += buf
        if result.remainders is not None and result.remainders[c] is not None:
            rem = result.remainders[c]
            assert rem is not None
            recon[:, cov_c:] += rem.float()
    return recon


class TestPerCodebookSignSplit:
    def _params(self, **overrides: object) -> CodebookParams:
        defaults: dict[str, object] = dict(
            k_gate=K,
            k_up=K,
            k_down=K,
            n_blocks_gate_up=N_BLOCKS,
            n_blocks_down=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            max_iters=20,
            norm_threshold=1e-9,
            skip_zeros=False,
        )
        defaults.update(overrides)
        return CodebookParams(**defaults)  # type: ignore[arg-type]

    def test_primary_only_signsplit_is_list_and_matches_reference(self) -> None:
        """(a) Bare-bool sign_split = primary-only; signs stored as a list at [0]."""
        torch.manual_seed(4)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        result = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="pss",
            sign_split=True,
        )).fit(rows)
        # sign_bits is now a per-codebook list; primary present, residual absent.
        assert isinstance(result.sign_bits, list)
        assert result.sign_bits[0] is not None
        assert result.sign_bits[0].shape == (NUM_EXPERTS * OUT_DIM, IN_DIM)
        assert result.sign_bits[1] is None
        # reconstruct() matches an independent reference.
        recon = result.reconstruct()
        assert torch.allclose(recon, _ref_reconstruct(result), atol=1e-4)

    def test_bool_true_equals_explicit_primary_only_list(self) -> None:
        """Regression: sign_split=True is IDENTICAL to sign_split=[True, False]."""
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        kwargs = dict(
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
        )
        # Reset RNG before each build: euclidean residual uses global-RNG Forgy init.
        torch.manual_seed(4)
        r_bool = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2), name="b", sign_split=True, **kwargs,
        )).fit(rows)
        torch.manual_seed(4)
        r_list = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2), name="l", sign_split=[True, False], **kwargs,
        )).fit(rows)
        assert torch.allclose(r_bool.reconstruct(), r_list.reconstruct(), atol=1e-6)

    def test_residual_signsplit_matches_reference(self) -> None:
        """(b) A residual with sign_split builds; reconstruct matches per-codebook ref."""
        torch.manual_seed(8)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.5
        result = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="rss",
            sign_split=[False, True],
        )).fit(rows)
        assert isinstance(result.sign_bits, list)
        assert result.sign_bits[0] is None
        assert result.sign_bits[1] is not None
        assert result.sign_bits[1].shape == (NUM_EXPERTS * OUT_DIM, IN_DIM)
        # Residual centroids clustered on folded (abs) error => all non-negative.
        assert result.codebooks[1].min().item() >= -1e-4
        recon = result.reconstruct()
        assert torch.allclose(recon, _ref_reconstruct(result), atol=1e-4)

    def test_both_signsplit_no_shape_error_and_reduces_error(self) -> None:
        """(c) Primary AND residual sign_split: valid shapes + residual reduces error."""
        torch.manual_seed(9)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.3
        r1 = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=1),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=1,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="p_only",
            sign_split=[True],
        )).fit(rows)
        r2 = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="both",
            sign_split=[True, True],
        )).fit(rows)
        recon2 = r2.reconstruct()
        assert recon2.shape == (NUM_EXPERTS * OUT_DIM, IN_DIM)
        assert torch.isfinite(recon2).all()
        assert torch.allclose(recon2, _ref_reconstruct(r2), atol=1e-4)
        err1 = (rows.float() - r1.reconstruct()).norm().item()
        err2 = (rows.float() - recon2).norm().item()
        assert err2 < err1, f"residual should reduce error: {err2} >= {err1}"

    def test_signsplit_length_mismatch_raises(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        try:
            CodebookExperiment(CodebookConfig(
                params=self._params(n_codebooks=2),
                k=K,
                n_blocks=N_BLOCKS,
                n_codebooks=2,
                num_experts=NUM_EXPERTS,
                out_dim=OUT_DIM,
                device=torch.device("cpu"),
                name="bad",
                sign_split=[True],  # too short for n_codebooks=2
            )).fit(rows)
        except ValueError:
            return
        raise AssertionError("expected ValueError for sign_split length mismatch")

    def test_gpu_forward_matches_reconstruct_with_signs(self) -> None:
        """CodebookModule forward (with per-codebook signs) matches result.reconstruct()."""
        torch.manual_seed(10)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.3
        result = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="gpuforward",
            sign_split=[True, True],
        )).fit(rows)
        mod = CodebookModule.from_result(result, out_dim=OUT_DIM)
        # signs carried through to both codebooks.
        assert mod.primary.signs is not None
        assert mod.additives[0].signs is not None

        # y = W_recon @ x  (row-major weights, per expert).
        recon = result.reconstruct().reshape(NUM_EXPERTS, OUT_DIM, IN_DIM)
        hidden = torch.randn(5, IN_DIM)
        for e in range(NUM_EXPERTS):
            actual = mod(hidden, expert_idx=e)
            expected = hidden @ recon[e].t()
            assert torch.allclose(actual, expected, atol=1e-3), f"expert {e} mismatch"

    def test_signsplit_state_dict_roundtrip(self) -> None:
        """Per-codebook signs survive save/load (meta carries sign_cov_list)."""
        torch.manual_seed(12)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.3
        result = CodebookExperiment(CodebookConfig(
            params=self._params(n_codebooks=2),
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=2,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="rt",
            sign_split=[True, True],
        )).fit(rows)
        mod = CodebookModule.from_result(result, out_dim=OUT_DIM).to(torch.bfloat16)
        hidden = torch.randn(6, IN_DIM).to(torch.bfloat16)
        out_before = mod(hidden, expert_idx=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "module.pt"
            torch.save(mod.state_dict_with_meta(), p)
            loaded = CodebookModule.load(p)
        assert loaded.primary.signs is not None
        assert loaded.additives[0].signs is not None
        out_after = loaded(hidden, expert_idx=2)
        assert torch.allclose(out_before, out_after, atol=1e-6)


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
        """Forward = scale_0 * primary_pass + sum_c residual_pass_c (residuals unscaled)."""
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
            l0 = hb @ mod.primary.codebook[b]
            prim_b = l0[:, mod.primary.assignments[expert_idx, b]]
            total = total + prim_b * mod.primary.scales[expert_idx, b]
            for cb_mod in mod.additives:
                lc = hb @ cb_mod.codebook[b]
                total = total + lc[:, cb_mod.assignments[expert_idx, b]]  # unscaled
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
        result = CodebookExperiment(CodebookConfig(
            params=params,
            k=K,
            n_blocks=N_BLOCKS,
            n_codebooks=N_CODEBOOKS,
            num_experts=NUM_EXPERTS,
            out_dim=OUT_DIM,
            device=torch.device("cpu"),
            name="int",
        )).fit(rows)
        mod = CodebookModule.from_result(result, out_dim=OUT_DIM)
        hidden = torch.randn(5, IN_DIM)
        out = mod(hidden, expert_idx=1)
        assert out.shape == (5, OUT_DIM)

        # Compare against manual forward using the *built* codebook
        expected = _manual_forward(mod, hidden, expert_idx=1)
        assert torch.allclose(out, expected, atol=1e-4)
