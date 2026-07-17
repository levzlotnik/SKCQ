#!/usr/bin/env python3
"""Zador's theorem distortion estimate — spherical k-means on S^(d-1).

On the unit sphere, effective dimension is d-1 (norm is captured by scale).
  D ~ K^(-2/(d-1))   (angular MSE as fraction of signal variance)
  R_assign = log2(K) / d   (bits per weight for assignments)
  R_scale  = 16 / d         (bits per weight for bf16 scales)

Compare with Euclidean k-means (no scale, full d-dim):
  D ~ K^(-2/d)
  R = log2(K) / d
"""
import math

DATASETS = {
    "gate/up": (131072, 2048),
    "down":    (524288, 512),
}

block_sizes = range(5, 15)        # 5..14
k_mults     = [16, 32, 64, 128, 256]  # K = 2^(bs-1) * mult

print(f"{'bs':>3} {'d_eff':>5} {'mult':>5} {'K':>10} {'R_assign':>8} {'R_scale':>7} {'D_sph':>8} {'D_euc':>8}  | {'proj':>8} {'cb_bpw':>8} {'total_bpw':>9} {'cr':>5}")
print("-" * 105)

for bs in block_sizes:
    d_eff = bs - 1
    for mult in k_mults:
        K = (2 ** d_eff) * mult
        R_assign = math.log2(K) / bs
        R_scale  = 16.0 / bs
        D_sph    = K ** (-2.0 / d_eff)   # spherical: d_eff = bs-1
        D_euc    = K ** (-2.0 / bs)      # euclidean: d = bs (no scale needed)

        for proj, (n_rows, in_dim) in DATASETS.items():
            n_blocks = in_dim // bs
            cb_bpw_shared = (K * 16) / (n_rows * n_blocks)
            total_bpw = R_assign + R_scale + cb_bpw_shared
            cr = 16.0 / total_bpw

            # Also show euclidean total (no scale)
            euc_bpw = math.log2(K) / bs + cb_bpw_shared
            euc_cr = 16.0 / euc_bpw

            print(f"{bs:>3} {d_eff:>5} {mult:>5} {K:>10} {R_assign:>8.3f} {R_scale:>7.3f} {D_sph:>8.5f} {D_euc:>8.5f}  | {proj:>8} {cb_bpw_shared:>8.5f} {total_bpw:>9.3f} {cr:>5.1f}")
        print()
