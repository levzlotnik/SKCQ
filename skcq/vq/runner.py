"""VQ config runner: the core logic for evaluating one VQ hyperparameter config.

Moved out of experiments/weight_quant_error.py so both the worker and the
experiment script can import it as a first-class package member (no sys.path
hacks).

Contains:
  - load_model_config / extract_layer_rows: model weight loading
  - Integer baseline quantization (int2/3/4/8, fp8)
  - Scale dtype quantization (int<N>, fp16, bf16, fp8)
  - quantize_scales, scale_bits_per_elem: scale quantization utilities
  - integer_schemes: int/fp8 baseline quantization functions
  - run_one_kmeans: the main VQ config runner
"""

# ruff: noqa: N806, N803, N802, E501  (W_raw / quant_intN match existing convention)
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

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


def integer_schemes(
    block_size: int,
) -> list[tuple[str, Callable[[torch.Tensor], tuple[torch.Tensor, float]]]]:
    """Build list of (name, quant_fn) pairs. quant_fn(W) -> (recon, bpw)."""
    schemes: list[tuple[str, Callable[[torch.Tensor], tuple[torch.Tensor, float]]]] = []
    for bits in [2, 3, 4, 8]:
        schemes.append((f"int{bits}_per_tensor", partial(quant_intN_per_tensor, bits=bits)))
        schemes.append((f"int{bits}_per_channel", partial(quant_intN_per_channel, bits=bits)))
        schemes.append(
            (
                f"int{bits}_per_block{block_size}",
                partial(quant_intN_per_block, bits=bits, block_size=block_size),
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
