"""Generate sweep configs for weight_quant_error.py experiments.

Produces a JSON list of configs (CLI arg strings + metadata) for the distributed
runner. Filters by divisibility, bpw range, and K <= n_rows.

Usage:
    uv run python scripts/gen_vq_hyperparams.py --out vq_results/configs.json
    uv run python scripts/gen_vq_hyperparams.py --out vq_results/configs.json --projection down
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Model: Qwen/Qwen3.6-35B-A3B
NUM_EXPERTS = 256
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
LAYER = 24

# Per-projection dimensions: (n_rows, in_dim)
PROJECTIONS = {
    "gate": (NUM_EXPERTS * INTERMEDIATE_SIZE, HIDDEN_SIZE),  # (131072, 2048)
    "up": (NUM_EXPERTS * INTERMEDIATE_SIZE, HIDDEN_SIZE),  # (131072, 2048)
    "down": (NUM_EXPERTS * HIDDEN_SIZE, INTERMEDIATE_SIZE),  # (524288, 512)
}

# Fixed sweep hyperparameters
SCALE_DTYPE = "int8"
METRIC = "cosine"
SHARED = True
SIGN_SPLIT = True
KMEANS_ITERS = 50

# Search space
BLOCK_SIZES = [8, 10, 12, 16, 24, 32, 64, 128]
K_VALUES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
N_CODEBOOKS = [1, 2, 3]
CODEBOOK_BITS = [8, 16]

# BPW bounds
BPW_MIN = 1.0
BPW_MAX = 6.0


# Curated residual (rbs, rk) pairs for n_cb >= 2.
# Not a full cross product — only combinations likely to be near Pareto.
# Key invariant: log2(rk)/rbs roughly constant at fixed assignment bpw.
# We pick (rbs, rk) so that rbs divides or is a multiple of primary bs,
# and rk spans a useful range.
def residual_pairs(primary_bs: int, in_dim: int) -> list[tuple[int, int]]:
    """Return curated (rbs, rk) pairs for residual codebooks (c >= 1).

    Each pair has rbs dividing in_dim and (rbs divides primary_bs OR primary_bs divides rbs).
    rk is chosen so log2(rk)/rbs covers a useful range (~0.5 to ~1.5 bits/elem).
    """
    candidates = []
    # All valid rbs values
    all_rbs = [4, 8, 10, 12, 16, 24, 32, 48, 64, 128]
    valid_rbs = []
    for rbs in all_rbs:
        if in_dim % rbs != 0:
            continue
        if rbs == primary_bs:
            continue
        if primary_bs % rbs == 0 or rbs % primary_bs == 0:
            valid_rbs.append(rbs)
    valid_rbs.append(primary_bs)  # same-bs residual

    # For each valid rbs, pick a few rk values targeting ~0.5-1.5 bits/elem of assignment
    # assignment_bpw_per_codebook = log2(rk) / rbs
    # so rk ≈ 2^(target_bits_per_elem * rbs)
    target_bits_per_elem = [0.5, 0.75, 1.0, 1.25, 1.5]
    for rbs in valid_rbs:
        for tbpe in target_bits_per_elem:
            rk = 1 << int(round(tbpe * rbs))
            if rk < 16:
                rk = 16
            if rk > 8192:
                rk = 8192
            candidates.append((rbs, rk))

    # Dedupe
    seen = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def bits_per_weight(
    n_rows: int,
    in_dim: int,
    block_size: int,
    k_per_codebook: list[int],
    bs_per_codebook: list[int],
    shared: bool,
    sign_split: bool,
    scale_bits_per_elem: int = 8,  # int8
    codebook_bits: int = 16,
) -> float:
    """Estimate BPW for a config (mirrors weight_quant_error.bits_per_weight_kmeans)."""
    n_blocks = in_dim // block_size
    quant_dim = n_blocks * block_size

    codebook_bits_total = 0
    assign_bits = 0
    for _c, (bs_c, k_c) in enumerate(zip(bs_per_codebook, k_per_codebook, strict=True)):
        n_blocks_c = in_dim // bs_c
        n_cb_c = 1 if shared else n_blocks_c
        codebook_bits_total += n_cb_c * k_c * bs_c * codebook_bits
        if k_c <= 1:
            assign_bits += n_rows * n_blocks_c * 1
        else:
            assign_bits += n_rows * n_blocks_c * math.ceil(math.log2(k_c))

    scale_bits = n_rows * n_blocks * scale_bits_per_elem
    sign_bits = n_rows * n_blocks * block_size if sign_split else 0

    total_bits = codebook_bits_total + assign_bits + scale_bits + sign_bits
    remainder_dim = in_dim - quant_dim
    if remainder_dim > 0:
        total_bits += 16 * n_rows * remainder_dim  # bf16 remainder

    return total_bits / (n_rows * in_dim)


def estimate_runtime_seconds(
    n_rows: int,
    in_dim: int,
    block_size: int,
    k_per_codebook: list[int],
    n_codebooks: int,
) -> float:
    """Rough runtime estimate: O(iters * n_rows * n_blocks * K * bs).

    Empirically: bs=8, K=8192, n_rows=131072 → ~30s on 3090.
    Calibrate from that.
    """
    total_work = 0
    for c in range(n_codebooks):
        n_blocks_c = in_dim // block_size  # assume primary bs for simplicity
        total_work += n_rows * n_blocks_c * k_per_codebook[c] * block_size
    # Calibrate: bs=8 K=8192 n_rows=131072 → work = 131072 * 256 * 8192 * 8 ≈ 2.2e12 → 30s
    # So factor ≈ 30 / 2.2e12 ≈ 1.36e-11
    return max(5.0, total_work * 1.36e-11 * KMEANS_ITERS / 100)


def gen_configs(projection: str) -> list[dict]:
    """Generate all valid configs for one projection."""
    n_rows, in_dim = PROJECTIONS[projection]
    configs = []
    seen_keys = set()

    for bs in BLOCK_SIZES:
        if in_dim % bs != 0:
            continue

        for k_val in K_VALUES:
            if n_rows < k_val:
                continue
            if n_rows // 10 < k_val and bs > 32:
                # too few points per centroid
                continue

            for n_cb in N_CODEBOOKS:
                # Build the list of (residual_block_sizes_list, residual_k_list) variants
                residual_variants: list[tuple[list[int] | None, list[int] | None]]
                if n_cb == 1:
                    residual_variants = [(None, None)]
                else:
                    residual_variants = [(None, None)]  # baseline: same as primary
                    for rbs, rk in residual_pairs(bs, in_dim):
                        rbs_v = [rbs] * (n_cb - 1)
                        rk_v = [rk] * (n_cb - 1)
                        residual_variants.append((rbs_v, rk_v))

                for rbs_v, rk_v in residual_variants:
                    # Build k_per_codebook and bs_per_codebook
                    k_pc = [k_val] * n_cb if rk_v is None else [k_val] + rk_v
                    bs_pc = [bs] * n_cb if rbs_v is None else [bs] + rbs_v

                    for cb_bits in CODEBOOK_BITS:
                        bpw = bits_per_weight(
                            n_rows,
                            in_dim,
                            bs,
                            k_pc,
                            bs_pc,
                            shared=SHARED,
                            sign_split=SIGN_SPLIT,
                            scale_bits_per_elem=8,
                            codebook_bits=cb_bits,
                        )
                        if bpw < BPW_MIN or bpw > BPW_MAX:
                            continue

                        # Dedupe key
                        key = (
                            projection,
                            bs,
                            k_val,
                            n_cb,
                            tuple(rbs_v) if rbs_v else None,
                            tuple(rk_v) if rk_v else None,
                            cb_bits,
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        # Build CLI args
                        args = [
                            "--projection",
                            projection,
                            "--block-size",
                            str(bs),
                            "--K",
                            str(k_val),
                            "--n-codebooks",
                            str(n_cb),
                            "--metric",
                            METRIC,
                            "--scale-dtype",
                            SCALE_DTYPE,
                            "--kmeans-iters",
                            str(KMEANS_ITERS),
                            "--codebook-bits",
                            str(cb_bits),
                        ]
                        if SHARED:
                            args.append("--shared")
                        if SIGN_SPLIT:
                            args.append("--sign-split")
                        if rbs_v is not None:
                            args += ["--residual-block-sizes", ",".join(str(x) for x in rbs_v)]
                        if rk_v is not None:
                            args += ["--residual-k", ",".join(str(x) for x in rk_v)]

                        est_runtime = estimate_runtime_seconds(
                            n_rows,
                            in_dim,
                            bs,
                            k_pc,
                            n_cb,
                        )

                        # Build label
                        cb_parts = [f"cb{c}_b{bs_pc[c]}k{k_pc[c]}" for c in range(n_cb)]
                        label = (
                            f"kmeans_{'-'.join(cb_parts)}_{METRIC[:3]}"
                            f"_{'shared' if SHARED else 'perblock'}"
                            f"_{'ssvq' if SIGN_SPLIT else 'nosign'}"
                            f"_int8_cb{cb_bits}"
                        )

                        configs.append(
                            {
                                "id": (
                                    f"{projection}_{bs}_K{k_val}_cb{n_cb}"
                                    f"_rbs{rbs_v}_rk{rk_v}_cbits{cb_bits}"
                                ),
                                "projection": projection,
                                "block_size": bs,
                                "K": k_val,
                                "n_codebooks": n_cb,
                                "residual_block_sizes": rbs_v,
                                "residual_k": rk_v,
                                "codebook_bits": cb_bits,
                                "est_bpw": round(bpw, 3),
                                "est_runtime_s": round(est_runtime, 1),
                                "label": label,
                                "args": args,
                            }
                        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep config JSON")
    parser.add_argument("--out", type=Path, default=Path("vq_results/configs.json"))
    parser.add_argument(
        "--projection",
        choices=["gate", "up", "down", "grid", "all"],
        default="grid",
        help="grid = gate+down (up is identical to gate, skip)",
    )
    args = parser.parse_args()

    all_configs: list[dict] = []
    if args.projection == "all":
        projections = ["gate", "up", "down"]
    elif args.projection == "grid":
        projections = ["gate", "down"]
    else:
        projections = [args.projection]
    for proj in projections:
        configs = gen_configs(proj)
        print(f"{proj}: {len(configs)} configs")
        all_configs.extend(configs)

    # Sort by estimated runtime descending (longest first for load balancing)
    all_configs.sort(key=lambda c: c["est_runtime_s"], reverse=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_configs, f, indent=2, separators=(",", ": "))

    # Summary
    print(f"\nTotal: {len(all_configs)} configs")
    print(f"BPW range: {BPW_MIN}-{BPW_MAX}")
    print(f"Est total runtime (serial): {sum(c['est_runtime_s'] for c in all_configs) / 3600:.1f}h")
    print(
        "Est total runtime (7 workers): "
        f"{sum(c['est_runtime_s'] for c in all_configs) / 7 / 3600:.1f}h"
    )

    # Distribution by projection
    by_proj = {}
    for c in all_configs:
        by_proj[c["projection"]] = by_proj.get(c["projection"], 0) + 1
    print(f"By projection: {by_proj}")

    # Distribution by bpw bucket
    buckets = {}
    for c in all_configs:
        b = int(c["est_bpw"])
        buckets[b] = buckets.get(b, 0) + 1
    print(f"By bpw bucket: {dict(sorted(buckets.items()))}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
