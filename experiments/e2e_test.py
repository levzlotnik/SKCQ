#!/usr/bin/env python3
"""End-to-end test on real GPU hardware.

Generates synthetic weights at Qwen3.6-35B-A3B gate dimensions, builds AQLM
codebooks (spherical primary + euclidean residual), quantizes scales, runs
CodebookModule forward, and verifies the full pipeline.

Usage:
    cuda/.venv/bin/python experiments/e2e_test.py
    cuda/.venv/bin/python experiments/e2e_test.py --experts 64 --K 4096
    cuda/.venv/bin/python experiments/e2e_test.py --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from skcq.codebook import (
    AQLMCodebook,
    AQLMCodebookConfig,
    EuclideanCodebook,
    EuclideanCodebookConfig,
    SphericalCodebook,
    SphericalCodebookConfig,
    TqdmListener,
    quantize_spherical_scales,
    scale_bits_per_elem,
)
from skcq.codebook.cwr import SphericalCWR
from skcq.codebook.events import KmeansDoneEvent, KmeansIterEvent, KmeansStartEvent
from skcq.codebook_experts import CodebookModule
from skcq.vq.bpw import bits_per_weight_kmeans

# Qwen3.6-35B-A3B dimensions
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 1024
NUM_EXPERTS = 256


def generate_weights(num_experts: int, out_dim: int, in_dim: int, device: torch.device) -> torch.Tensor:
    """Generate synthetic MoE gate weights.

    Real Qwen weights have a specific distribution — roughly normal with
    std ~0.02-0.05, with some rows near-zero (dead experts). We approximate
    this with a mixture: 90% normal(0, 0.03), 10% normal(0, 0.005) (near-dead).
    """
    torch.manual_seed(42)
    n_rows = num_experts * out_dim
    w = torch.randn(n_rows, in_dim, device=device, dtype=torch.bfloat16) * 0.03
    # Sprinkle near-dead rows
    dead_mask = torch.rand(n_rows, device=device) < 0.1
    w[dead_mask] = torch.randn(dead_mask.sum().item(), in_dim, device=device, dtype=torch.bfloat16) * 0.005
    return w


def run_e2e(
    device: torch.device,
    num_experts: int,
    out_dim: int,
    in_dim: int,
    block_size: int,
    k: int,
    n_codebooks: int,
    max_iters: int,
    scale_dtype: str,
    chunk_budget_mb: int,
) -> int:
    print(f"=== E2E Test ===")
    print(f"  Device: {device}")
    print(f"  Dimensions: {num_experts} experts × {out_dim} out_dim × {in_dim} in_dim")
    print(f"  Codebook: block_size={block_size}, K={k}, n_codebooks={n_codebooks}")
    print(f"  Scale quant: {scale_dtype}")
    print(f"  K-means iters: {max_iters}")
    print()

    # --- 1. Generate weights ---
    t0 = time.perf_counter()
    w = generate_weights(num_experts, out_dim, in_dim, device)
    t_gen = time.perf_counter() - t0
    n_rows = w.shape[0]
    print(f"[1/6] Generated weights: {tuple(w.shape)} ({w.element_size()}B, {t_gen:.2f}s)")

    # --- 2. Build AQLM codebook ---
    t0 = time.perf_counter()
    primary = SphericalCodebook(
        SphericalCodebookConfig(
            block_size=block_size,
            K=k,
            shared=False,
            max_iters=max_iters,
            norm_threshold=0.001,
            skip_zeros=True,
            chunk_budget_mb=chunk_budget_mb,
            device=device,
        )
    )
    residuals = [
        EuclideanCodebook(
            EuclideanCodebookConfig(
                block_size=block_size,
                K=k,
                shared=False,
                max_iters=max_iters,
                norm_threshold=0.001,
                skip_zeros=True,
                chunk_budget_mb=chunk_budget_mb,
                device=device,
            )
        )
        for _ in range(n_codebooks - 1)
    ]
    aqlm = AQLMCodebook(primary, residuals, AQLMCodebookConfig(norm_threshold=0.001))  # type: ignore[arg-type]

    # Wire tqdm progress to stderr
    tqdm_listener = TqdmListener()
    aqlm.on(KmeansStartEvent, tqdm_listener.on_start)
    aqlm.on(KmeansIterEvent, tqdm_listener.on_iter)
    aqlm.on(KmeansDoneEvent, tqdm_listener.on_done)

    cwr = aqlm.fit(w)
    t_build = time.perf_counter() - t0
    print(f"[2/6] Built AQLM codebook: {t_build:.2f}s")

    # --- 3. Quantize scales ---
    t0 = time.perf_counter()
    scale_bits = scale_bits_per_elem(scale_dtype)
    primary_cwr: SphericalCWR = cwr.primary  # type: ignore[assignment]  # runtime: SphericalCWR
    quantize_spherical_scales(primary_cwr, primary, scale_dtype=scale_dtype)
    t_quant = time.perf_counter() - t0
    print(f"[3/6] Quantized scales to {scale_dtype} ({scale_bits}b/elem): {t_quant:.2f}s")

    # --- 4. Reconstruct and measure error ---
    t0 = time.perf_counter()
    recon = aqlm.reconstruct(cwr)
    w_f32 = w.float()
    rel_err = (w_f32 - recon).norm().item() / w_f32.norm().item()
    t_recon = time.perf_counter() - t0
    print(f"[4/6] Reconstruction: rel_fro_err={rel_err:.4f} ({t_recon:.2f}s)")

    # --- 5. Build CodebookModule and run forward ---
    t0 = time.perf_counter()
    mod = CodebookModule.from_cwr(aqlm, cwr, out_dim=out_dim).to(torch.float32)
    hidden = torch.randn(8, in_dim, device=device)
    expert_idx = 3
    out_quantized = mod(hidden, expert_idx=expert_idx)
    # Direct matmul with original weights for comparison
    w_real = w.float().reshape(num_experts, out_dim, in_dim)
    out_reference = hidden @ w_real[expert_idx].t()
    forward_err = (out_quantized - out_reference).norm().item() / out_reference.norm().item()
    t_fwd = time.perf_counter() - t0
    print(f"[5/6] Forward pass: shape={tuple(out_quantized.shape)}, "
          f"forward_err={forward_err:.4f} ({t_fwd:.2f}s)")

    # --- 6. Bits per weight ---
    n_blocks = in_dim // block_size
    bpw = bits_per_weight_kmeans(
        n_rows=n_rows,
        in_dim=in_dim,
        n_blocks=n_blocks,
        block_size=block_size,
        n_codebooks=n_codebooks,
        k_per_codebook=[k] * n_codebooks,
        shared_codebook=False,
        sign_split=False,
        scale_bits_per_elem=scale_bits,
        codebook_bits=16,
        primary_metric="cosine",
    )
    compression = 16.0 / bpw
    print(f"[6/6] Bits per weight: {bpw:.3f} (compression {compression:.1f}× vs bf16)")

    # --- Summary ---
    print()
    print("=== Summary ===")
    print(f"  rel_fro_err:   {rel_err:.4f}")
    print(f"  forward_err:   {forward_err:.4f}")
    print(f"  bits/weight:   {bpw:.3f}")
    print(f"  compression:   {compression:.1f}×")
    print(f"  build_time:    {t_build:.2f}s")
    print(f"  recon_time:    {t_recon:.2f}s")
    print(f"  forward_time:  {t_fwd:.4f}s")

    # Assertions — these are the e2e correctness checks
    assert rel_err < 0.20, f"Reconstruction error too high: {rel_err:.4f}"
    assert forward_err < 0.25, f"Forward error too high: {forward_err:.4f}"
    assert bpw < 16.0, f"bpw higher than bf16 baseline: {bpw:.3f}"
    print()
    print("ALL CHECKS PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E codebook test on real GPU")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--experts", type=int, default=32, help="Number of experts (subset of 256)")
    parser.add_argument("--out-dim", type=int, default=INTERMEDIATE_SIZE, help="Output dimension")
    parser.add_argument("--in-dim", type=int, default=HIDDEN_SIZE, help="Input dimension")
    parser.add_argument("--block-size", type=int, default=8, help="Codebook block size")
    parser.add_argument("--K", type=int, default=8192, help="Codebook size (centroids per block)")
    parser.add_argument("--n-codebooks", type=int, default=2, help="Primary + residuals")
    parser.add_argument("--max-iters", type=int, default=25, help="K-means iterations")
    parser.add_argument("--scale-dtype", default="int8", help="Scale quantization dtype")
    parser.add_argument("--chunk-budget-mb", type=int, default=2048, help="K-means memory budget")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)

    sys.exit(
        run_e2e(
            device=device,
            num_experts=args.experts,
            out_dim=args.out_dim,
            in_dim=args.in_dim,
            block_size=args.block_size,
            k=args.K,
            n_codebooks=args.n_codebooks,
            max_iters=args.max_iters,
            scale_dtype=args.scale_dtype,
            chunk_budget_mb=args.chunk_budget_mb,
        )
    )


if __name__ == "__main__":
    main()
