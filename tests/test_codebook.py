from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from skcq.codebook import (
    AQLMCodebook,
    AQLMCodebookConfig,
    EuclideanCodebook,
    EuclideanCodebookConfig,
    SphericalCodebook,
    SphericalCodebookConfig,
    SSVQCodebook,
    quantize_spherical_scales,
)
from skcq.codebook.cwr import AQLMCWR, EuclideanCWR, SphericalCWR, SSVQCWR
from skcq.codebook_experts import CodebookModule
from skcq.config import CodebookParams, build_aqlm_codebook

NUM_EXPERTS = 4
OUT_DIM = 8
IN_DIM = 12
BLOCK_SIZE = 4
N_BLOCKS = IN_DIM // BLOCK_SIZE
K = 16
N_CODEBOOKS = 2


def _make_aqlm(
    n_codebooks: int = N_CODEBOOKS,
    block_size: int = BLOCK_SIZE,
    k_list: list[int] | None = None,
    shared: bool = False,
    sign_split: bool = False,
    residual_block_sizes: int | None = None,
) -> AQLMCodebook:
    if k_list is None:
        k_list = [K] * n_codebooks
    primary = SphericalCodebook(SphericalCodebookConfig(
        block_size=block_size, K=k_list[0], shared=shared,
        max_iters=10, norm_threshold=1e-9, skip_zeros=False,
    ))
    if sign_split:
        primary = SSVQCodebook(primary)
    residuals = []
    for c in range(1, n_codebooks):
        rbs = residual_block_sizes or block_size
        residuals.append(EuclideanCodebook(EuclideanCodebookConfig(
            block_size=rbs, K=k_list[c], shared=shared,
            max_iters=10, norm_threshold=1e-9, skip_zeros=False,
        )))
    return AQLMCodebook(primary, residuals, AQLMCodebookConfig(norm_threshold=1e-9))


def _manual_forward(aqlm: AQLMCodebook, cwr: AQLMCWR, hidden: torch.Tensor, expert_idx: int) -> torch.Tensor:
    recon = aqlm.reconstruct(cwr).reshape(NUM_EXPERTS, OUT_DIM, -1)
    return hidden @ recon[expert_idx].t()


class TestCodebookModuleForward:
    def test_output_shape(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm()
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        hidden = torch.randn(5, IN_DIM)
        out = mod(hidden, expert_idx=2)
        assert out.shape == (5, OUT_DIM)

    def test_matches_reconstruct(self) -> None:
        torch.manual_seed(1)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm()
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        hidden = torch.randn(7, IN_DIM)
        for e in range(NUM_EXPERTS):
            actual = mod(hidden, expert_idx=e)
            expected = _manual_forward(aqlm, cwr, hidden, e)
            assert torch.allclose(actual, expected, atol=1e-3), f"expert {e} mismatch"

    def test_single_codebook(self) -> None:
        torch.manual_seed(2)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm(n_codebooks=1)
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        hidden = torch.randn(4, IN_DIM)
        actual = mod(hidden, expert_idx=0)
        expected = _manual_forward(aqlm, cwr, hidden, 0)
        assert torch.allclose(actual, expected, atol=1e-3)

    def test_ssvq_forward(self) -> None:
        torch.manual_seed(3)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.3
        aqlm = _make_aqlm(sign_split=True)
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        assert mod.primary.signs is not None
        hidden = torch.randn(5, IN_DIM)
        for e in range(NUM_EXPERTS):
            actual = mod(hidden, expert_idx=e)
            expected = _manual_forward(aqlm, cwr, hidden, e)
            assert torch.allclose(actual, expected, atol=1e-3), f"ssvq expert {e}"

    def test_noncommensurate_forward(self) -> None:
        torch.manual_seed(5)
        in_dim = 34
        w = torch.randn(NUM_EXPERTS * OUT_DIM, in_dim) * 0.1
        aqlm = _make_aqlm(block_size=10, residual_block_sizes=12)
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        recon = aqlm.reconstruct(cwr).reshape(NUM_EXPERTS, OUT_DIM, in_dim)
        hidden = torch.randn(5, in_dim)
        for e in range(NUM_EXPERTS):
            actual = mod(hidden, expert_idx=e)
            expected = hidden @ recon[e].t()
            assert torch.allclose(actual, expected, atol=1e-3), f"noncomm expert {e}"

    def test_expert_isolation(self) -> None:
        torch.manual_seed(4)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm()
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        hidden = torch.randn(6, IN_DIM)
        out_a = mod(hidden, expert_idx=0)
        out_b = mod(hidden, expert_idx=1)
        assert not torch.allclose(out_a, out_b)


class TestBuildCodebook:
    def _params(self, **kw) -> CodebookParams:
        defaults = dict(
            k_gate=K, k_up=K, k_down=K, n_blocks_gate_up=N_BLOCKS, n_blocks_down=N_BLOCKS,
            n_codebooks=N_CODEBOOKS, max_iters=10, norm_threshold=1e-9, skip_zeros=False,
        )
        defaults.update(kw)
        return CodebookParams(**defaults)

    def test_output_shapes(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params = self._params()
        aqlm = build_aqlm_codebook(params, k=K, n_blocks=N_BLOCKS, n_codebooks=N_CODEBOOKS,
                                  in_dim=IN_DIM, out_dim=OUT_DIM, num_experts=NUM_EXPERTS)
        cwr = aqlm.fit(rows)
        assert isinstance(cwr, AQLMCWR)
        assert isinstance(cwr.primary, SphericalCWR)
        assert len(cwr.residuals) == N_CODEBOOKS - 1
        assert cwr.primary.idxs.shape == (N_BLOCKS, NUM_EXPERTS * OUT_DIM)
        assert cwr.primary.idxs.dtype == torch.long
        assert cwr.primary.scales.shape == (NUM_EXPERTS * OUT_DIM, N_BLOCKS)

    def test_in_dim_not_divisible(self) -> None:
        rows = torch.randn(8, 13) * 0.1
        aqlm = _make_aqlm(block_size=4, n_codebooks=1)
        cwr = aqlm.fit(rows)
        assert cwr.primary.remainder is not None
        assert cwr.primary.remainder.shape == (8, 1)
        recon = aqlm.reconstruct(cwr)
        assert recon.shape == (8, 13)
        assert torch.allclose(recon[:, 12:].float(), rows[:, 12:].float(), atol=1e-2)

    def test_residual_reduces_error(self) -> None:
        torch.manual_seed(42)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        a1 = _make_aqlm(n_codebooks=1)
        c1 = a1.fit(rows)
        a2 = _make_aqlm(n_codebooks=2)
        c2 = a2.fit(rows)
        err1 = (rows.float() - a1.reconstruct(c1)).norm().item() / rows.float().norm().item()
        err2 = (rows.float() - a2.reconstruct(c2)).norm().item() / rows.float().norm().item()
        assert err2 < err1, f"residual should reduce: {err2} >= {err1}"

    def test_real_error_residual(self) -> None:
        torch.manual_seed(0)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 10.0
        aqlm = _make_aqlm(n_codebooks=2, k_list=[K, K // 2])
        cwr = aqlm.fit(rows)
        res_cwr = cwr.residuals[0]
        assert isinstance(res_cwr, EuclideanCWR)
        assert res_cwr.idxs.shape[0] == N_BLOCKS

    def test_asymmetric_k(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        params = self._params(n_codebooks=2, residual_k=8)
        aqlm = build_aqlm_codebook(params, k=K, n_blocks=N_BLOCKS, n_codebooks=2,
                                  in_dim=IN_DIM, out_dim=OUT_DIM, num_experts=NUM_EXPERTS)
        cwr = aqlm.fit(rows)
        assert isinstance(cwr.residuals[0], EuclideanCWR)
        assert aqlm.residuals[0].K == 8

    def test_noncommensurate_block_sizes(self) -> None:
        torch.manual_seed(5)
        in_dim = 34
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, in_dim) * 0.1
        aqlm = _make_aqlm(block_size=10, residual_block_sizes=12)
        cwr = aqlm.fit(rows)
        recon = aqlm.reconstruct(cwr)
        assert recon.shape == (NUM_EXPERTS * OUT_DIM, in_dim)
        assert torch.isfinite(recon).all()
        err = (rows.float() - recon).norm().item()
        assert err < rows.float().norm().item()

    def test_scale_refit_orthogonal(self) -> None:
        torch.manual_seed(7)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm(n_codebooks=2)
        cwr = aqlm.fit(rows)
        recon = aqlm.reconstruct(cwr)
        sph_cwr = cwr.primary
        for b in range(N_BLOCKS):
            cols = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
            cb = aqlm.primary.directions[b].float()
            d = cb.t()[sph_cwr.idxs[b]]
            resid = rows.float()[:, cols] - recon[:, cols]
            dot = torch.einsum("nd,nd->n", resid, d)
            assert dot.abs().max().item() < 1e-2, f"block {b} not orthogonal: {dot.abs().max()}"

    def test_zero_rows_excluded(self) -> None:
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        rows[0] = 0
        aqlm = AQLMCodebook(
            primary=SphericalCodebook(SphericalCodebookConfig(
                block_size=BLOCK_SIZE, K=K, max_iters=10, norm_threshold=1e-5, skip_zeros=True)),
            residuals=[EuclideanCodebook(EuclideanCodebookConfig(
                block_size=BLOCK_SIZE, K=K, max_iters=10, norm_threshold=1e-5, skip_zeros=False))],
            config=AQLMCodebookConfig(norm_threshold=1e-5),
        )
        cwr = aqlm.fit(rows)
        assert cwr.primary.scales[0, :].sum() == 0
        assert cwr.primary.scales[1, :].abs().sum() > 0

    def test_subset_reconstruction(self) -> None:
        torch.manual_seed(11)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm()
        cwr = aqlm.fit(rows)
        full_recon = aqlm.reconstruct(cwr)
        subset_cwr = AQLMCWR(
            primary=SphericalCWR(
                idxs=cwr.primary.idxs[:, :8], scales=cwr.primary.scales[:8],
                remainder=cwr.primary.remainder[:8] if cwr.primary.remainder is not None else None,
            ),
            residuals=[EuclideanCWR(
                idxs=r.idxs[:, :8],
                remainder=r.remainder[:8] if r.remainder is not None else None,
            ) for r in cwr.residuals],
        )
        subset_recon = aqlm.reconstruct(subset_cwr)
        assert subset_recon.shape == (8, IN_DIM)
        assert torch.allclose(subset_recon, full_recon[:8], atol=1e-5)


class TestSSVQ:
    def test_primary_signsplit(self) -> None:
        torch.manual_seed(4)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm(sign_split=True)
        cwr = aqlm.fit(rows)
        assert isinstance(cwr.primary, SSVQCWR)
        assert cwr.primary.signs is not None
        assert cwr.primary.signs.shape == (NUM_EXPERTS * OUT_DIM, IN_DIM)

    def test_ssvq_reduces_error(self) -> None:
        torch.manual_seed(8)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.5
        a1 = _make_aqlm(sign_split=False)
        c1 = a1.fit(rows)
        a2 = _make_aqlm(sign_split=True)
        c2 = a2.fit(rows)
        err1 = (rows.float() - a1.reconstruct(c1)).norm().item() / rows.float().norm().item()
        err2 = (rows.float() - a2.reconstruct(c2)).norm().item() / rows.float().norm().item()
        assert err2 < err1, f"ssvq should reduce: {err2} >= {err1}"

    def test_ssvq_forward_matches_reconstruct(self) -> None:
        torch.manual_seed(10)
        rows = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.3
        aqlm = _make_aqlm(sign_split=True)
        cwr = aqlm.fit(rows)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        recon = aqlm.reconstruct(cwr).reshape(NUM_EXPERTS, OUT_DIM, IN_DIM)
        hidden = torch.randn(5, IN_DIM)
        for e in range(NUM_EXPERTS):
            actual = mod(hidden, expert_idx=e)
            expected = hidden @ recon[e].t()
            assert torch.allclose(actual, expected, atol=1e-3), f"ssvq expert {e}"


class TestStateDictRoundTrip:
    def test_roundtrip(self) -> None:
        torch.manual_seed(13)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm(k_list=[K, K // 2])
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.bfloat16)
        hidden = torch.randn(6, IN_DIM).to(torch.bfloat16)
        out_before = mod(hidden, expert_idx=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "module.pt"
            torch.save(mod.state_dict_with_meta(), p)
            loaded = CodebookModule.load(p)
        out_after = loaded(hidden, expert_idx=2)
        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_ssvq_roundtrip(self) -> None:
        torch.manual_seed(12)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.3
        aqlm = _make_aqlm(sign_split=True)
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.bfloat16)
        hidden = torch.randn(6, IN_DIM).to(torch.bfloat16)
        out_before = mod(hidden, expert_idx=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "module.pt"
            torch.save(mod.state_dict_with_meta(), p)
            loaded = CodebookModule.load(p)
        assert loaded.primary.signs is not None
        out_after = loaded(hidden, expert_idx=2)
        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestForwardModes:
    def _make_mod(self, k_list: list[int]) -> tuple[AQLMCodebook, AQLMCWR, CodebookModule]:
        torch.manual_seed(42)
        w = torch.randn(NUM_EXPERTS * OUT_DIM, IN_DIM) * 0.1
        aqlm = _make_aqlm(k_list=k_list)
        cwr = aqlm.fit(w)
        mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32)
        return aqlm, cwr, mod

    def _set_mode(self, mod: CodebookModule, mode: str) -> CodebookModule:
        for layer in mod.layers:
            layer.forward_mode = mode  # type: ignore[assignment]
        return mod

    def test_modes_equivalent(self) -> None:
        aqlm, cwr, mod = self._make_mod([K, K // 2])
        mod_mg = self._set_mode(mod, "matmul_gather")
        mod_gm = self._set_mode(CodebookModule.from_cwr(aqlm, cwr, out_dim=OUT_DIM).to(torch.float32), "gather_matmul")
        hidden = torch.randn(5, IN_DIM)
        out_mg = mod_mg(hidden, expert_idx=1)
        out_gm = mod_gm(hidden, expert_idx=1)
        assert torch.allclose(out_mg, out_gm, atol=1e-6)


class TestScaleQuant:
    def test_int8_scale_quant(self) -> None:
        torch.manual_seed(15)
        w = torch.randn(32, 12) * 0.3
        aqlm = _make_aqlm()
        cwr = aqlm.fit(w)
        recon_before = aqlm.reconstruct(cwr)
        err_before = (w.float() - recon_before).norm().item() / w.float().norm().item()
        quantize_spherical_scales(cwr.primary, aqlm.primary, "int8")
        assert cwr.primary.scales.dtype == torch.int8
        assert aqlm.primary.scale_quant.q_scale is not None
        recon_after = aqlm.reconstruct(cwr)
        err_after = (w.float() - recon_after).norm().item() / w.float().norm().item()
        assert err_after >= err_before
        assert err_after < err_before * 1.5


class TestSharedCodebook:
    def test_shared_builds_and_reconstructs(self) -> None:
        torch.manual_seed(6)
        w = torch.randn(64, 12) * 0.3
        aqlm = _make_aqlm(shared=True)
        cwr = aqlm.fit(w)
        recon = aqlm.reconstruct(cwr)
        assert recon.shape == (64, 12)
        err = (w.float() - recon).norm().item() / w.float().norm().item()
        assert err < 1.0
