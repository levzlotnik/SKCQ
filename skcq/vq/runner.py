"""VQ config runner: the core logic for evaluating one VQ hyperparameter config.

Moved out of experiments/weight_quant_error.py so both the worker and the
experiment script can import it as a first-class package member (no sys.path
hacks).

Contains:
  - load_model_config / extract_layer_rows: model weight loading
  - Integer baseline quantization (int2/3/4/8, fp8)
  - Scale dtype quantization (int<N>, fp16, bf16, fp8)
  - bits_per_weight_kmeans: BPW accounting
  - run_one_kmeans: the main VQ config runner
"""

# ruff: noqa: N806, N803, N802, E501  (W_raw / quant_intN match existing convention)
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from skcq.clustering import CodebookParams, build_codebook

logger = logging.getLogger("skcq.vq.runner")


# ---------------------------------------------------------------------------
# Model config + weight extraction
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


def extract_layer_rows(
    model_id: str, layer_idx: int, hidden_size: int, intermediate_size: int
) -> dict[str, torch.Tensor]:
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
# Integer baseline quantization (affine, all bit widths)
# ---------------------------------------------------------------------------
# Affine (asymmetric) quantization: maps [min, max] -> [0, 2^b - 1].
# Uses all 2^b codes. Stores min + scale per group (both fp16 = 32 bits overhead).
#
# Per-block variant uses `block_size` matching the VQ run for apples-to-apples.
#
# BPW accounting:
#   per-tensor:  bits + 32/(n_rows * in_dim)  (~bits, negligible overhead)
#   per-channel: bits + 32/in_dim             (one min+scale per row)
#   per-block:   bits + 32/block_size         (one min+scale per block, per row)
# Remainder columns (in_dim % block_size) kept as bf16 (16 bpw).


def _quant_affine(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Affine quantize W to `bits` bits. Returns reconstructed tensor."""
    n_levels = 2**bits - 1
    w_min = W.min()
    w_max = W.max()
    scale = (w_max - w_min) / n_levels
    if scale == 0:
        return W.clone()
    q = torch.round((W - w_min) / scale).clamp(0, n_levels)
    return q * scale + w_min


def quant_intN_per_tensor(W: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """Affine per-tensor: one min+scale for the whole tensor."""
    n_rows, in_dim = W.shape
    recon = _quant_affine(W, bits)
    data_bits = bits * n_rows * in_dim
    overhead_bits = 32  # one fp16 min + one fp16 scale
    bpw = (data_bits + overhead_bits) / (n_rows * in_dim)
    return recon, bpw


def quant_intN_per_channel(W: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """Affine per-channel: each row gets its own min+scale."""
    n_rows, in_dim = W.shape
    n_levels = 2**bits - 1
    w_min = W.amin(dim=-1, keepdim=True)
    w_max = W.amax(dim=-1, keepdim=True)
    scale = ((w_max - w_min) / n_levels).clamp(min=1e-10)
    q = torch.round((W - w_min) / scale).clamp(0, n_levels)
    recon = q * scale + w_min
    data_bits = bits * n_rows * in_dim
    overhead_bits = 32 * n_rows  # fp16 min + fp16 scale per row
    bpw = (data_bits + overhead_bits) / (n_rows * in_dim)
    return recon, bpw


def quant_intN_per_block(W: torch.Tensor, bits: int, block_size: int) -> tuple[torch.Tensor, float]:
    """Affine per-block: each group of `block_size` consecutive elements gets its own min+scale."""
    n_rows, in_dim = W.shape
    n_blocks = in_dim // block_size
    quant_dim = n_blocks * block_size
    remainder_dim = in_dim - quant_dim
    n_levels = 2**bits - 1

    recon = torch.empty_like(W)
    for b in range(n_blocks):
        block = W[:, b * block_size : (b + 1) * block_size]
        w_min = block.amin(dim=-1, keepdim=True)
        w_max = block.amax(dim=-1, keepdim=True)
        scale = ((w_max - w_min) / n_levels).clamp(min=1e-10)
        q = torch.round((block - w_min) / scale).clamp(0, n_levels)
        recon[:, b * block_size : (b + 1) * block_size] = q * scale + w_min

    # Remainder kept as bf16
    if remainder_dim > 0:
        recon[:, quant_dim:] = W[:, quant_dim:]

    data_bits = bits * n_rows * quant_dim
    overhead_bits = 32 * n_blocks * n_rows  # fp16 min + fp16 scale per block per row
    remainder_bits = 16 * n_rows * remainder_dim  # bf16 remainder
    total_bits = data_bits + overhead_bits + remainder_bits
    bpw = total_bits / (n_rows * in_dim)
    return recon, bpw


def quant_fp8_e4m3(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    return W.to(torch.float8_e4m3fn).to(W.dtype), 8.0


def quant_fp8_e5m2(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    return W.to(torch.float8_e5m2).to(W.dtype), 8.0


def integer_schemes(block_size: int) -> list[tuple[str, callable]]:
    """Build list of (name, quant_fn) pairs. quant_fn(W) -> (recon, bpw)."""
    schemes: list[tuple[str, callable]] = []
    for bits in [2, 3, 4, 8]:
        schemes.append((f"int{bits}_per_tensor", lambda W, b=bits: quant_intN_per_tensor(W, b)))
        schemes.append((f"int{bits}_per_channel", lambda W, b=bits: quant_intN_per_channel(W, b)))
        schemes.append(
            (
                f"int{bits}_per_block{block_size}",
                lambda W, b=bits, bs=block_size: quant_intN_per_block(W, b, bs),
            )
        )
    schemes.append(("fp8_e4m3", quant_fp8_e4m3))
    schemes.append(("fp8_e5m2", quant_fp8_e5m2))
    return schemes


# ---------------------------------------------------------------------------
# Scale dtype parsing and quantization
# ---------------------------------------------------------------------------

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
# BPW accounting
# ---------------------------------------------------------------------------


def bits_per_weight_kmeans(
    n_rows: int,
    in_dim: int,
    n_blocks: int,
    block_size: int,
    n_codebooks: int,
    k_per_codebook: list[int],
    shared_codebook: bool = False,
    sign_split: bool = False,
    scale_bits_per_elem: int = 16,
    bs_per_codebook: list[int] | None = None,
    codebook_bits: int = 16,
) -> float:
    """Compute effective bits per weight for k-means quantization."""
    if bs_per_codebook is None:
        bs_per_codebook = [block_size] * n_codebooks

    codebook_bits_total = 0
    assign_bits = 0
    for _c, (bs_c, k_c) in enumerate(zip(bs_per_codebook, k_per_codebook, strict=True)):
        n_blocks_c = in_dim // bs_c
        n_cb_c = 1 if shared_codebook else n_blocks_c
        codebook_bits_total += n_cb_c * k_c * bs_c * codebook_bits
        if k_c <= 1:
            assign_bits += n_rows * n_blocks_c * 1
        else:
            assign_bits += n_rows * n_blocks_c * math.ceil(math.log2(k_c))

    scale_bits = n_rows * n_blocks * scale_bits_per_elem
    sign_bits = n_rows * n_blocks * block_size if sign_split else 0

    total_bits = codebook_bits_total + assign_bits + scale_bits + sign_bits
    total_weights = n_rows * in_dim
    return total_bits / total_weights


# ---------------------------------------------------------------------------
# Single-config VQ run
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
    shared_codebook: bool = False,
    sign_split: bool = False,
    max_iters: int = 100,
    scale_dtype: str = "bf16",
    residual_block_sizes: int | list[int] | None = None,
    codebook_bits: int = 16,
    residual_k: int | list[int] | None = None,
    layer_idx: int = 24,
    device: torch.device | None = None,
    chunk_budget_mb: int = 256,
    primary_codebook_cache: object | None = None,
    cache_key_str: str | None = None,
) -> dict:
    """Run a single k-means config on one projection."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_rows = W_raw.shape[0]
    W_norm = W_raw.float().norm().item()
    n_blocks = in_dim // block_size
    quant_dim = n_blocks * block_size
    remainder_dim = in_dim - quant_dim
    # K per codebook: c=0 uses K, c>=1 uses residual_k
    if residual_k is None:
        k_per_codebook = [K] * n_codebooks
    elif isinstance(residual_k, int):
        k_per_codebook = [K if c == 0 else residual_k for c in range(n_codebooks)]
    else:  # list[int]
        if len(residual_k) < n_codebooks - 1:
            raise ValueError(
                f"residual_k list has {len(residual_k)} values, need {n_codebooks - 1}"
            )
        k_per_codebook = [K if c == 0 else residual_k[c - 1] for c in range(n_codebooks)]

    # Block size per codebook
    if residual_block_sizes is None:
        bs_per_codebook = [block_size] * n_codebooks
    elif isinstance(residual_block_sizes, int):
        bs_per_codebook = [
            block_size if c == 0 else residual_block_sizes for c in range(n_codebooks)
        ]
    else:  # list[int]
        if len(residual_block_sizes) < n_codebooks - 1:
            raise ValueError(
                f"residual_block_sizes has {len(residual_block_sizes)} values, need {n_codebooks - 1}"
            )
        bs_per_codebook = [
            block_size if c == 0 else residual_block_sizes[c - 1] for c in range(n_codebooks)
        ]

    shared_tag = "shared" if shared_codebook else "perblock"
    ssvq_tag = "ssvq" if sign_split else "nosign"
    cb_parts = [f"cb{c}_b{bs_per_codebook[c]}k{k_per_codebook[c]}" for c in range(n_codebooks)]
    cb_id = "-".join(cb_parts)
    scale_tag = f"_{scale_dtype}" if scale_dtype != "bf16" else ""
    cb_qtag = f"_cb{codebook_bits}" if codebook_bits < 16 else ""
    label = f"kmeans_{cb_id}_{metric[:3]}_{shared_tag}_{ssvq_tag}{scale_tag}{cb_qtag}"
    logger.info(
        "[%s] %s (n_blocks=%d, remainder=%d, K=%d, K_r=%s, bs_r=%s, cb=%d, metric=%s, shared=%s)",
        projection,
        label,
        n_blocks,
        remainder_dim,
        K,
        residual_k,
        residual_block_sizes,
        n_codebooks,
        metric,
        shared_codebook,
    )

    out_dim = intermediate_size if projection != "down" else hidden_size

    params = CodebookParams(
        k_gate=K,
        k_up=K,
        k_down=K,
        n_blocks_gate_up=n_blocks,
        n_blocks_down=n_blocks,
        n_codebooks=n_codebooks,
        residual_k=residual_k,
        residual_block_sizes=residual_block_sizes,
        max_iters=max_iters,
        norm_threshold=0.001,
        skip_zeros=True,
        chunk_budget_mb=chunk_budget_mb,
    )

    W_quant = W_raw[:, :quant_dim]
    W_remainder = W_raw[:, quant_dim:] if remainder_dim > 0 else None

    result = build_codebook(
        rows=W_quant,
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
        residual_block_sizes=residual_block_sizes,
        codebook_bits=codebook_bits,
        primary_codebook_cache=primary_codebook_cache,
        cache_key_str=cache_key_str,
    )

    # Quantize scales (full-precision fp32 → target dtype → dequantized fp32)
    sc_bits = scale_bits_per_elem(scale_dtype)
    result.scales = quantize_scales(result.scales, scale_dtype)
    logger.info("  [%s] scale quantized to %s (%d bits/elem)", projection, scale_dtype, sc_bits)

    W_recon_quant = result.reconstruct()
    if W_remainder is not None:
        W_recon = torch.cat([W_recon_quant, W_remainder.float()], dim=1)
    else:
        W_recon = W_recon_quant

    err = torch.norm(W_raw.float() - W_recon).item() / W_norm

    bpw_quant = bits_per_weight_kmeans(
        n_rows,
        quant_dim,
        n_blocks,
        block_size,
        n_codebooks,
        k_per_codebook,
        shared_codebook=shared_codebook,
        sign_split=sign_split,
        scale_bits_per_elem=sc_bits,
        bs_per_codebook=bs_per_codebook,
        codebook_bits=codebook_bits,
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
        "residual_block_sizes": (
            [block_size] + [bs for bs in bs_per_codebook[1:] if bs != block_size]
            if any(bs != block_size for bs in bs_per_codebook[1:])
            else []
        ),
        "rel_fro_err": err,
        "bits_per_weight": bpw,
        "compression_ratio": comp_ratio,
    }
