"""Generate sweep YAML configs and report compression rates.

Inspects the actual model config to derive dimensions, so no magic numbers.
For each config, prints a table row showing n_blocks, block_size, K, ratio,
bits-per-weight, and compression ratio.

Usage: uv run python scripts/gen_sweep_configs.py [--out configs/sweep]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from skcq.config import CodebookParams, ExperimentConfig, LayerOverride
from skcq.eval_model import load_model


def codebook_bits(k_list: list[int], block_size: int) -> int:
    """Bits to store all sub-codebooks: sum_c(K_c * block_size * 16) (bf16)."""
    return sum(k_c * block_size * 16 for k_c in k_list)


def index_bits(k_list: list[int], n_blocks: int, num_experts: int, out_dim: int) -> int:
    """Bits to store assignments: per expert, per out_dim, per block, per codebook.

    Asymmetric K means a different bits-per-index per codebook: ceil(log2(K_c)).
    """
    bits = 0
    for k_c in k_list:
        bits_per_index = max(1, math.ceil(math.log2(k_c))) if k_c > 1 else 1
        bits += num_experts * out_dim * n_blocks * bits_per_index
    return bits


def scale_bits(n_blocks: int, num_experts: int, out_dim: int) -> int:
    """Bits to store scales (bf16): only cb0 has scales — one per expert/out_dim/block."""
    return num_experts * out_dim * n_blocks * 16


def projection_bits(
    k_list: list[int], n_blocks: int, in_dim: int, out_dim: int, num_experts: int
) -> int:
    block_size = in_dim // n_blocks
    return (
        codebook_bits(k_list, block_size)
        + index_bits(k_list, n_blocks, num_experts, out_dim)
        + scale_bits(n_blocks, num_experts, out_dim)
    )


def original_bits(in_dim: int, out_dim: int, num_experts: int, bits: int = 16) -> int:
    return num_experts * out_dim * in_dim * bits


def compression_ratio(
    k_gu: int,
    k_dn: int,
    n_blocks: int,
    n_codebooks: int,
    k_residual_mult: float,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> float:
    """Total quantized bits / total original bits for one layer's MoE."""
    k_list_gu = [max(1, int(k_gu / k_residual_mult**c)) for c in range(n_codebooks)]
    k_list_dn = [max(1, int(k_dn / k_residual_mult**c)) for c in range(n_codebooks)]
    # gate + up (in=hidden, out=intermediate), down (in=intermediate, out=hidden)
    orig = 2 * original_bits(hidden_size, intermediate_size, num_experts) + original_bits(
        intermediate_size, hidden_size, num_experts
    )
    quant = 2 * projection_bits(
        k_list_gu, n_blocks, hidden_size, intermediate_size, num_experts
    ) + projection_bits(k_list_dn, n_blocks, intermediate_size, hidden_size, num_experts)
    return orig / quant


def gen_config(
    label: str,
    r: int,
    n_blocks: int,
    n_codebooks: int,
    k_residual_mult: float,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    out_dir: Path,
) -> dict[str, Any]:
    """r = K/block_size (centroids per block-dimension, uniform across projections)."""
    bs_gu = hidden_size // n_blocks
    bs_dn = intermediate_size // n_blocks
    k_gu = r * bs_gu
    k_dn = r * bs_dn

    config = ExperimentConfig(
        model_id="Qwen/Qwen3.6-35B-A3B",
        eval_samples=100,
        output_dir=Path(f"codebooks_{label}"),
        defaults=CodebookParams(
            k_gate=k_gu,
            k_up=k_gu,
            k_down=k_dn,
            n_blocks_gate_up=n_blocks,
            n_blocks_down=n_blocks,
            n_codebooks=n_codebooks,
            k_residual_mult=k_residual_mult,
            max_iters=100,
            norm_threshold=0.001,
            skip_zeros=True,
        ),
        layer_overrides={0: LayerOverride(skip_zeros=True, norm_threshold=0.001)},
    )
    yaml_path = out_dir / f"{label}.yaml"
    yaml_path.write_text(
        yaml.dump(config.model_dump(mode="json"), default_flow_style=False, sort_keys=False)
    )
    return {
        "label": label,
        "r": r,
        "n_blocks": n_blocks,
        "n_codebooks": n_codebooks,
        "k_residual_mult": k_residual_mult,
        "bs_gu": bs_gu,
        "bs_dn": bs_dn,
        "k_gu": k_gu,
        "k_dn": k_dn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep configs + compression table")
    parser.add_argument("--out", type=Path, default=Path("configs/sweep"), help="Output dir")
    parser.add_argument("--no-model", action="store_true", help="Skip model load, use known dims")
    args = parser.parse_args()

    if args.no_model:
        hidden_size = 2048
        intermediate_size = 512
        num_experts = 256
    else:
        print("Loading model to inspect dims...", flush=True)
        model, _ = load_model("Qwen/Qwen3.6-35B-A3B", device="cpu")
        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        del model

    print(
        f"Model dims: hidden={hidden_size}, intermediate={intermediate_size}, experts={num_experts}"
    )
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    # Clear old configs
    for f in args.out.glob("*.yaml"):
        f.unlink()

    # r = K/block_size (centroids per block-dimension, uniform across projections)
    r_list = [8, 16, 32, 64, 128, 256]
    n_blocks_list = [1, 2, 4, 8, 16]
    n_codebooks_list = [1, 2, 3]
    k_residual_mult = 2.0  # K_0/K_r ratio (was 0.5 = multiply, now 2.0 = divide)

    n_gu = num_experts * intermediate_size  # total rows clustered per gate/up codebook
    n_dn = num_experts * hidden_size  # total rows clustered per down codebook

    rows = []
    skipped = 0
    for r in r_list:
        for nb in n_blocks_list:
            for n_cb in n_codebooks_list:
                bs_gu = hidden_size // nb
                bs_dn = intermediate_size // nb
                k_gu = r * bs_gu
                k_dn = r * bs_dn
                # Skip if K exceeds training rows (can't cluster meaningfully)
                if k_gu > n_gu or k_dn > n_dn:
                    skipped += 1
                    continue
                label = f"kbs{r}_nb{nb}_cb{n_cb}"
                info = gen_config(
                    label,
                    r,
                    nb,
                    n_cb,
                    k_residual_mult,
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    args.out,
                )
                comp = compression_ratio(
                    info["k_gu"],
                    info["k_dn"],
                    nb,
                    n_cb,
                    k_residual_mult,
                    hidden_size,
                    intermediate_size,
                    num_experts,
                )
                info["compression"] = comp
                info["n_gu"] = n_gu
                info["n_dn"] = n_dn
                # The printed table focuses on the 4-16x compression target, but ALL
                # valid yamls (K <= N) are kept on disk so curated tiers (e.g.
                # sweep_tier1.sh) can reference high-compression configs too.
                if comp < 4 or comp > 16:
                    skipped += 1
                    continue
                rows.append(info)

    # Print table
    print(
        f"{'label':28} {'r':>4} {'N_gu/K_gu':>9} {'N_dn/K_dn':>9} {'K_gu':>6} {'bs_gu':>5} "
        f"{'K_dn':>6} {'bs_dn':>5} {'n_cb':>4} {'krm':>5} {'compress':>9}"
    )
    print("-" * 106)
    for r_info in rows:
        n_over_k_gu = r_info["n_gu"] / r_info["k_gu"]
        n_over_k_dn = r_info["n_dn"] / r_info["k_dn"]
        print(
            f"{r_info['label']:28} {r_info['r']:>4} {n_over_k_gu:>9.1f} {n_over_k_dn:>9.1f} "
            f"{r_info['k_gu']:>6} {r_info['bs_gu']:>5} "
            f"{r_info['k_dn']:>6} {r_info['bs_dn']:>5} "
            f"{r_info['n_codebooks']:>4} {r_info['k_residual_mult']:>5.2f} "
            f"{r_info['compression']:>8.1f}x"
        )

    if skipped:
        print(
            f"\n{skipped} configs kept on disk but omitted from table "
            "(K > N, or compression outside 4-16x)"
        )
    print(f"Generated configs in {args.out}/ ({len(rows)} shown in table)")


if __name__ == "__main__":
    main()
