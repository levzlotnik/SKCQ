#!/usr/bin/env python3
"""Weight-space quantization error experiment — thin CLI wrapper.

All reusable logic (model loading, integer baselines, scale quantization,
BPW accounting, run_one_kmeans) lives in skcq.vq.runner. This script is just
the argparse CLI for ad-hoc standalone runs.

Usage:
    rocm/.venv/bin/python experiments/weight_quant_error.py --block-size 8 --K 8192 --shared --sign-split --scale-dtype int8
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch

# skcq is importable when running from repo root (pythonpath=["."] in pyproject.toml)
from skcq.vq.runner import (
    integer_schemes,
    load_model_config,
    parse_scale_dtype,
    run_one_kmeans,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("weight_quant_error")
logger.setLevel(logging.INFO)
for quiet in ("skcq.clustering", "skcq", "pt_kmeans", "pt_kmeans.pt_kmeans"):
    logging.getLogger(quiet).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight quantization error experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--projection", choices=["gate", "up", "down", "all"], default="all")
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--n-codebooks", type=int, default=1)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--sign-split", action="store_true")
    parser.add_argument("--residual-k", type=str, default=None)
    parser.add_argument("--residual-block-sizes", type=str, default=None)
    parser.add_argument("--codebook-bits", type=int, default=16)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--scale-dtype", type=str, default="bf16")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-budget-mb", type=int, default=256)
    args = parser.parse_args()

    args.scale_dtype = parse_scale_dtype(args.scale_dtype)

    if args.residual_k is not None:
        args.residual_k = (
            [int(x) for x in args.residual_k.split(",")]
            if "," in args.residual_k
            else int(args.residual_k)
        )
    if args.residual_block_sizes is not None:
        args.residual_block_sizes = (
            [int(x) for x in args.residual_block_sizes.split(",")]
            if "," in args.residual_block_sizes
            else int(args.residual_block_sizes)
        )

    output_csv = (
        Path(args.output) if args.output else Path(__file__).parent / "weight_quant_error.csv"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Using GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    config = load_model_config(args.model)
    num_experts = config["num_experts"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["moe_intermediate_size"]

    from skcq.vq.runner import extract_layer_rows

    rows_map = extract_layer_rows(args.model, args.layer, hidden_size, intermediate_size)

    all_results: list[dict] = []
    all_projections = [
        ("gate", rows_map["gate"], hidden_size),
        ("up", rows_map["up"], hidden_size),
        ("down", rows_map["down"], intermediate_size),
    ]
    projections = (
        all_projections
        if args.projection == "all"
        else [p for p in all_projections if p[0] == args.projection]
    )

    for proj_name, W_raw, in_dim in projections:
        W_norm = W_raw.float().norm().item()
        for name, quant_fn in integer_schemes(args.block_size):
            W_q, bpw = quant_fn(W_raw)
            err = torch.norm(W_raw.float() - W_q.float()).item() / W_norm
            all_results.append(
                {
                    "projection": proj_name,
                    "scheme": name,
                    "block_size": args.block_size,
                    "K": 0,
                    "n_codebooks": 0,
                    "metric": "",
                    "shared": False,
                    "sign_split": False,
                    "scale_dtype": "fp16",
                    "kmeans_iters": 0,
                    "residual_block_sizes": [],
                    "rel_fro_err": err,
                    "bits_per_weight": bpw,
                    "compression_ratio": 16.0 / bpw,
                }
            )

        result = run_one_kmeans(
            W_raw=W_raw,
            projection=proj_name,
            in_dim=in_dim,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_size=args.block_size,
            K=args.K,
            n_codebooks=args.n_codebooks,
            metric=args.metric,
            residual_k=args.residual_k,
            shared_codebook=args.shared,
            sign_split=args.sign_split,
            max_iters=args.kmeans_iters,
            scale_dtype=args.scale_dtype,
            residual_block_sizes=args.residual_block_sizes,
            codebook_bits=args.codebook_bits,
            layer_idx=args.layer,
            device=device,
            chunk_budget_mb=args.chunk_budget_mb,
        )
        all_results.append(result)
        del W_raw

    all_results.sort(key=lambda r: (r["projection"], r["bits_per_weight"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "projection",
        "scheme",
        "block_size",
        "K",
        "n_codebooks",
        "metric",
        "shared",
        "sign_split",
        "scale_dtype",
        "kmeans_iters",
        "residual_block_sizes",
        "rel_fro_err",
        "bits_per_weight",
        "compression_ratio",
    ]
    write_header = args.overwrite or not output_csv.exists()
    with open(output_csv, "w" if args.overwrite else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(all_results)

    print()
    for r in all_results:
        print(
            f"{r['projection']:<10} {r['scheme']:<50} {r['rel_fro_err']:>12.6f} {r['bits_per_weight']:>8.3f} {r['compression_ratio']:>12.2f}"
        )


if __name__ == "__main__":
    main()
