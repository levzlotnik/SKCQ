#!/usr/bin/env python3
"""Weight-space quantization error experiment.

Compares integer baselines (int8/int4/FP8) against spherical k-means
codebook quantization. Measures relative Frobenius reconstruction error.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from skcq.clustering import CodebookParams, build_codebook

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("weight_quant_error")
logger.setLevel(logging.INFO)

for quiet in ("skcq.clustering", "skcq", "pt_kmeans", "pt_kmeans.pt_kmeans"):
    logging.getLogger(quiet).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Model config loading
# ---------------------------------------------------------------------------
def load_model_config(model_id: str) -> dict:
    """Read model dimensions from config.json."""
    model_dir = Path(snapshot_download(model_id))
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    if "text_config" in config:
        config = config["text_config"]
    return {
        "num_experts": config["num_experts"],
        "hidden_size": config["hidden_size"],
        "moe_intermediate_size": config["moe_intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
    }


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_layer_rows(model_id: str, layer_idx: int, hidden_size: int, intermediate_size: int) -> dict[str, torch.Tensor]:
    """Extract gate, up, down weight rows for one layer from safetensors."""
    model_dir = Path(snapshot_download(model_id))
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    shards: dict[str, str] = {}
    for key, shard in weight_map.items():
        if f"layers.{layer_idx}." not in key:
            continue
        if ".mlp.experts.gate_up_proj" in key:
            shards["gate_up"] = shard
        elif ".mlp.experts.down_proj" in key:
            shards["down"] = shard

    gu_path = model_dir / shards["gate_up"]
    with safe_open(gu_path, framework="pt", device="cpu") as f:
        gu_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        gate_up = f.get_tensor(gu_key)
    gate_rows = gate_up[:, :intermediate_size, :].reshape(-1, hidden_size)
    up_rows = gate_up[:, intermediate_size:, :].reshape(-1, hidden_size)

    dn_path = model_dir / shards["down"]
    with safe_open(dn_path, framework="pt", device="cpu") as f:
        dn_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        down = f.get_tensor(dn_key)
    down_rows = down.reshape(-1, intermediate_size)

    return {"gate": gate_rows, "up": up_rows, "down": down_rows}


# ---------------------------------------------------------------------------
# Integer baseline quantization
# ---------------------------------------------------------------------------
def quant_int8_per_tensor(W: torch.Tensor) -> torch.Tensor:
    scale = W.abs().max() / 127
    q = torch.round(W / scale).clamp(-127, 127)
    return q * scale


def quant_int8_per_channel(W: torch.Tensor) -> torch.Tensor:
    scale = W.abs().amax(dim=-1, keepdim=True) / 127
    q = torch.round(W / scale).clamp(-127, 127)
    return q * scale


def quant_int4_per_tensor(W: torch.Tensor) -> torch.Tensor:
    scale = W.abs().max() / 7
    q = torch.round(W / scale).clamp(-7, 7)
    return q * scale


def quant_int4_per_channel(W: torch.Tensor) -> torch.Tensor:
    scale = W.abs().amax(dim=-1, keepdim=True) / 7
    q = torch.round(W / scale).clamp(-7, 7)
    return q * scale


def quant_fp8_e4m3(W: torch.Tensor) -> torch.Tensor:
    return W.to(torch.float8_e4m3fn).to(W.dtype)


def quant_fp8_e5m2(W: torch.Tensor) -> torch.Tensor:
    return W.to(torch.float8_e5m2).to(W.dtype)


INTEGER_SCHEMES = [
    ("int8_per_tensor", quant_int8_per_tensor, 8),
    ("int8_per_channel", quant_int8_per_channel, 8),
    ("int4_per_tensor", quant_int4_per_tensor, 4),
    ("int4_per_channel", quant_int4_per_channel, 4),
    ("fp8_e4m3", quant_fp8_e4m3, 8),
    ("fp8_e5m2", quant_fp8_e5m2, 8),
]


# ---------------------------------------------------------------------------
# Scale dtype parsing and quantization
# ---------------------------------------------------------------------------
# Supported formats:
#   int<Nbits>  — symmetric per-tensor integer quantization (e.g. int4, int8)
#   fp16        — torch.float16
#   bf16        — torch.bfloat16  (default)
#   fp8_e4m3    — torch.float8_e4m3fn
#   fp8_e5m2    — torch.float8_e5m2

_FP8_MAP = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def parse_scale_dtype(s: str) -> str:
    """Validate and normalise a scale-dtype string."""
    s = s.strip().lower()
    if s.startswith("int"):
        bits = int(s[3:])
        if bits < 2 or bits > 16:
            raise ValueError(f"int bits must be 2-16, got {bits}")
        return s
    if s in ("fp16", "bf16"):
        return s
    if s in _FP8_MAP:
        return s
    raise ValueError(
        f"Unknown scale dtype '{s}'. Expected: int<Nbits>, fp16, bf16, fp8_e4m3, fp8_e5m2"
    )


def scale_bits_per_elem(dtype: str) -> int:
    """Bits per scale element for the given dtype."""
    if dtype.startswith("int"):
        return int(dtype[3:])
    if dtype in _FP8_MAP:
        return 8
    if dtype in ("fp16", "bf16"):
        return 16
    raise ValueError(f"Unknown scale dtype: {dtype}")


def quantize_scales(scales: torch.Tensor, dtype: str) -> torch.Tensor:
    """Quantize-and-dequantize scales to the target dtype, returning float32.

    For integer dtypes: per-tensor symmetric quantization. The single global
    fp32 scale factor is negligible (one float per (n_rows, n_blocks) tensor).
    """
    if dtype == "bf16":
        return scales.to(torch.bfloat16).to(torch.float32)
    if dtype == "fp16":
        return scales.to(torch.float16).to(torch.float32)
    if dtype in _FP8_MAP:
        return scales.to(_FP8_MAP[dtype]).to(torch.float32)
    if dtype.startswith("int"):
        bits = int(dtype[3:])
        levels = 2 ** (bits - 1) - 1  # symmetric: -levels..+levels
        abs_max = scales.abs().max()
        if abs_max == 0:
            return scales.clone()
        q_scale = abs_max / levels
        q = torch.round(scales / q_scale).clamp(-levels, levels)
        return (q * q_scale).to(torch.float32)
    raise ValueError(f"Unknown scale dtype: {dtype}")


# ---------------------------------------------------------------------------
# Spherical k-means reconstruction
# ---------------------------------------------------------------------------
def reconstruct_from_codebook(result, n_rows: int, n_blocks: int, block_size: int):
    """Reconstruct weight matrix from CodebookResult (all on CPU)."""
    recon_blocks = []
    for b in range(n_blocks):
        final_direction = torch.zeros(n_rows, block_size, dtype=torch.float32)
        for c in range(result.n_codebooks):
            if result.shared_codebook:
                cb_b = result.codebooks[c][0].float()
            else:
                cb_b = result.codebooks[c][b].float()
            asg_b = result.assignments[c][b]
            final_direction = final_direction + cb_b.t()[asg_b]
        scale_b = result.scales[:, b].float()
        recon_block = scale_b.unsqueeze(-1) * final_direction
        # Reapply signs if sign-split was used
        if result.sign_bits is not None:
            recon_block = result.sign_bits[:, b, :].float() * recon_block
        recon_blocks.append(recon_block)
    return torch.cat(recon_blocks, dim=1)


def bits_per_weight_kmeans(
    n_rows: int, in_dim: int, n_blocks: int, block_size: int,
    n_codebooks: int, k_per_codebook: list[int],
    shared_codebook: bool = False,
    sign_split: bool = False,
    scale_bits_per_elem: int = 16,
) -> float:
    """Compute effective bits per weight for k-means quantization."""
    n_cb = 1 if shared_codebook else n_blocks
    codebook_bits = n_cb * sum(K_c * block_size * 16 for K_c in k_per_codebook)
    assign_bits = 0
    for K_c in k_per_codebook:
        if K_c <= 1:
            assign_bits += n_rows * n_blocks * 1
        else:
            assign_bits += n_rows * n_blocks * math.ceil(math.log2(K_c))
    scale_bits = n_rows * n_blocks * scale_bits_per_elem
    sign_bits = n_rows * n_blocks * block_size if sign_split else 0  # 1 bit per element
    total_bits = codebook_bits + assign_bits + scale_bits + sign_bits
    total_weights = n_rows * in_dim
    return total_bits / total_weights


# ---------------------------------------------------------------------------
# Single-config k-means run
# ---------------------------------------------------------------------------
def run_one_kmeans(
    W_raw: torch.Tensor,
    projection: str,
    in_dim: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    K: int,
    n_codebooks: int,
    metric: str,
    k_residual_mult: float = 1.0,
    shared_codebook: bool = False,
    sign_split: bool = False,
    max_iters: int = 100,
    scale_dtype: str = "bf16",
    layer_idx: int = 24,
    device: torch.device | None = None,
    chunk_budget_mb: int = 256,
) -> dict:
    """Run a single k-means config on one projection."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_rows = W_raw.shape[0]
    W_norm = W_raw.float().norm().item()
    n_blocks = in_dim // block_size
    quant_dim = n_blocks * block_size
    remainder_dim = in_dim - quant_dim
    k_per_codebook = [max(1, int(K / k_residual_mult ** c)) for c in range(n_codebooks)]

    shared_tag = "shared" if shared_codebook else "perblock"
    ssvq_tag = "ssvq" if sign_split else "nosign"
    label = f"kmeans_bs{block_size}_K{K}_cb{n_codebooks}_{metric[:3]}_{shared_tag}_{ssvq_tag}_{scale_dtype}"
    logger.info(
        "[%s] %s (n_blocks=%d, remainder=%d, K=%d, cb=%d, metric=%s, krm=%.1f, shared=%s)",
        projection, label, n_blocks, remainder_dim, K, n_codebooks, metric, k_residual_mult, shared_codebook,
    )

    out_dim = intermediate_size if projection != "down" else hidden_size

    params = CodebookParams(
        k_gate=K, k_up=K, k_down=K,
        n_blocks_gate_up=n_blocks, n_blocks_down=n_blocks,
        n_codebooks=n_codebooks,
        k_residual_mult=k_residual_mult,
        max_iters=max_iters,
        norm_threshold=0.001,
        skip_zeros=True,
        chunk_budget_mb=chunk_budget_mb,
    )

    W_quant = W_raw[:, :quant_dim]
    W_remainder = W_raw[:, quant_dim:] if remainder_dim > 0 else None

    result = build_codebook(
        rows=W_quant.to(device),
        params=params,
        k=K,
        n_blocks=n_blocks,
        n_codebooks=n_codebooks,
        num_experts=num_experts,
        out_dim=out_dim,
        device=device,
        name=f"L{layer_idx}.{projection}.{label}",
        distance_metric=metric,
        shared_codebook=shared_codebook,
        sign_split=sign_split,
    )

    # Quantize scales (full-precision fp32 → target dtype → dequantized fp32)
    sc_bits = scale_bits_per_elem(scale_dtype)
    result.scales = quantize_scales(result.scales, scale_dtype)
    logger.info("  [%s] scale quantized to %s (%d bits/elem)", projection, scale_dtype, sc_bits)

    W_recon_quant = reconstruct_from_codebook(result, n_rows, n_blocks, block_size)
    if W_remainder is not None:
        W_recon = torch.cat([W_recon_quant, W_remainder.float()], dim=1)
    else:
        W_recon = W_recon_quant

    err = torch.norm(W_raw.float() - W_recon).item() / W_norm

    bpw_quant = bits_per_weight_kmeans(
        n_rows, quant_dim, n_blocks, block_size, n_codebooks, k_per_codebook,
        shared_codebook=shared_codebook, sign_split=sign_split,
        scale_bits_per_elem=sc_bits,
    )
    if remainder_dim > 0:
        total_bits = bpw_quant * n_rows * quant_dim + 16 * n_rows * remainder_dim
        bpw = total_bits / (n_rows * in_dim)
    else:
        bpw = bpw_quant
    comp_ratio = 16.0 / bpw

    logger.info("  [%s] err=%.6f bpw=%.3f cr=%.2f", projection, err, bpw, comp_ratio)

    del result, W_recon
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "projection": projection,
        "scheme": label,
        "block_size": block_size,
        "K": K,
        "n_codebooks": n_codebooks,
        "metric": metric,
        "shared": shared_codebook,
        "sign_split": sign_split,
        "scale_dtype": scale_dtype,
        "kmeans_iters": max_iters,
        "rel_fro_err": err,
        "bits_per_weight": bpw,
        "compression_ratio": comp_ratio,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Weight quantization error experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-35B-A3B", help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=24, help="Layer index")
    parser.add_argument("--projection", choices=["gate", "up", "down", "all"], default="all", help="Projection(s) to run")
    parser.add_argument("--block-size", type=int, required=True, help="Sub-vector block size (in_dim must be divisible)")
    parser.add_argument("--K", type=int, required=True, help="Codebook size (number of centroids)")
    parser.add_argument("--n-codebooks", type=int, default=1, help="Number of codebooks (1=no residual, 2+=residual)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="Distance metric (cosine=spherical, euclidean=l2)")
    parser.add_argument("--shared", action="store_true", help="Use a single shared codebook across all blocks")
    parser.add_argument("--sign-split", action="store_true", help="Extract signs, cluster on first orthant (SSVQ)")
    parser.add_argument("--krm", type=float, default=1.0, help="K_0/K_r ratio: primary codebook size divided by residual codebook size (e.g. 32 means K_r = K/32). Default 1.0 = equal sizes")
    parser.add_argument("--kmeans-iters", type=int, default=100, help="Max k-means iterations")
    parser.add_argument(
        "--scale-dtype", type=str, default="bf16",
        help="Scale quantization dtype: int<Nbits> (e.g. int8, int4), fp16, bf16, fp8_e4m3, fp8_e5m2",
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: experiments/weight_quant_error.csv)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite CSV instead of appending")
    parser.add_argument("--chunk-budget-mb", type=int, default=256, help="Memory budget for k-means chunking (MB)")
    args = parser.parse_args()

    # Validate scale dtype
    args.scale_dtype = parse_scale_dtype(args.scale_dtype)
    logger.info("Scale dtype: %s (%d bits/elem)", args.scale_dtype, scale_bits_per_elem(args.scale_dtype))

    # Resolve output path
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = Path(__file__).parent / "weight_quant_error.csv"

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning("CUDA unavailable, using CPU")

    # Load model config
    logger.info("Loading config for %s...", args.model)
    config = load_model_config(args.model)
    num_experts = config["num_experts"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["moe_intermediate_size"]

    # Extract layer weights
    logger.info("Extracting layer %d weights...", args.layer)
    rows_map = extract_layer_rows(args.model, args.layer, hidden_size, intermediate_size)
    logger.info(
        "gate: %s, up: %s, down: %s",
        tuple(rows_map["gate"].shape),
        tuple(rows_map["up"].shape),
        tuple(rows_map["down"].shape),
    )

    all_results: list[dict] = []

    all_projections = [
        ("gate", rows_map["gate"], hidden_size),
        ("up", rows_map["up"], hidden_size),
        ("down", rows_map["down"], intermediate_size),
    ]

    if args.projection == "all":
        projections = all_projections
    else:
        projections = [p for p in all_projections if p[0] == args.projection]

    for proj_name, W_raw, in_dim in projections:
        logger.info("Processing %s projection (%s)...", proj_name, tuple(W_raw.shape))

        # Integer baselines
        W_norm = W_raw.float().norm().item()
        for name, quant_fn, bits in INTEGER_SCHEMES:
            W_q = quant_fn(W_raw)
            err = torch.norm(W_raw.float() - W_q.float()).item() / W_norm
            all_results.append({
                "projection": proj_name,
                "scheme": name,
                "block_size": 0,
                "K": 0,
                "n_codebooks": 0,
                "metric": "",
                "shared": False,
                "sign_split": False,
                "scale_dtype": "",
                "kmeans_iters": 0,
                "rel_fro_err": err,
                "bits_per_weight": float(bits),
                "compression_ratio": 16.0 / bits,
            })
            logger.info("  [%s] %-20s err=%.6f bpw=%.1f", proj_name, name, err, bits)

        # K-means codebook
        result = run_one_kmeans(
            W_raw, proj_name, in_dim,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_size=args.block_size,
            K=args.K,
            n_codebooks=args.n_codebooks,
            metric=args.metric,
            k_residual_mult=args.krm,
            shared_codebook=args.shared,
            sign_split=args.sign_split,
            max_iters=args.kmeans_iters,
            scale_dtype=args.scale_dtype,
            layer_idx=args.layer,
            device=device,
            chunk_budget_mb=args.chunk_budget_mb,
        )
        all_results.append(result)
        del W_raw

    # Sort by bits_per_weight within each projection
    all_results.sort(key=lambda r: (r["projection"], r["bits_per_weight"]))

    # Write CSV (append by default, --overwrite to reset)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "projection", "scheme", "block_size", "K", "n_codebooks", "metric",
        "shared", "sign_split", "scale_dtype", "kmeans_iters",
        "rel_fro_err", "bits_per_weight", "compression_ratio",
    ]
    write_header = args.overwrite or not output_csv.exists()
    mode = "w" if args.overwrite else "a"
    with open(output_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(all_results)
    logger.info("Wrote %d rows to %s (mode=%s)", len(all_results), output_csv, mode)

    # Print table
    print()
    print(f"{'projection':<10} {'scheme':<50} {'rel_fro_err':>12} {'bpw':>8} {'compression':>12}")
    print("-" * 96)
    current_proj = None
    for r in all_results:
        if r["projection"] != current_proj:
            current_proj = r["projection"]
            print(f"\n--- {current_proj} ---")
        print(f"{r['projection']:<10} {r['scheme']:<50} {r['rel_fro_err']:>12.6f} {r['bits_per_weight']:>8.3f} {r['compression_ratio']:>12.2f}")
    print()


if __name__ == "__main__":
    main()
