#!/usr/bin/env python3
"""Distributed codebook build: orchestrates workers across multiple machines.

Computes the baseline on the local GPU (Strix Halo iGPU), then distributes
codebook building across all configured workers (local 3090 + remote machines
via SSH/tailscale). Workers read safetensors directly — no model loading on
workers, no shared memory, no RPC beyond simple TCP.

Usage:
    rocm/.venv/bin/python distributed_run.py --config configs/sweep/kbs16_nb4_cb2.yaml \\
        --workers workers.yaml --baseline-cache baseline.pt

After the build completes, run quantized eval separately:
    cuda/.venv/bin/python eval_quantized.py --config ... --codebook-dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from skcq.config import ExperimentConfig
from skcq.eval_model import compute_perplexity, get_calibration_text, load_model
from skcq.logging_setup import setup_logging
from skcq.metrics import capture_reference_logits, install_routing_capture, remove_hooks
from skcq.protocol import ModelConfig

logger = logging.getLogger("skcq")


def run_distributed(
    exp_config: ExperimentConfig,
    device: str,
    workers_yaml: Path,
    codebook_dir: Path,
    kld_tokens: int,
    baseline_cache: Path | None = None,
) -> None:
    """Compute baseline locally, then distribute codebook build across workers."""
    from transformers import AutoConfig

    from skcq.orchestrator import Orchestrator

    hf_config = AutoConfig.from_pretrained(exp_config.model_id, trust_remote_code=True)
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    model_config = ModelConfig(
        num_experts=hf_config.num_experts,
        hidden_size=hf_config.hidden_size,
        moe_intermediate_size=hf_config.moe_intermediate_size,
    )
    num_layers = hf_config.num_hidden_layers

    baseline_valid = False
    if baseline_cache is not None and baseline_cache.exists():
        logger.info("Loading baseline from %s...", baseline_cache)
        baseline = torch.load(baseline_cache, weights_only=False)
        cached_kld = baseline.get("kld_tokens", 0)
        if cached_kld != kld_tokens:
            logger.warning(
                "Baseline cache has kld_tokens=%d but requested %d — re-capturing baseline",
                cached_kld,
                kld_tokens,
            )
            baseline_cache.unlink()
        else:
            baseline_valid = True
            logger.info("Baseline ppl: %.4f (cached)", baseline["base_ppl"])

    if not baseline_valid:
        logger.info("Loading model for baseline computation...")
        model, tokenizer = load_model(exp_config.model_id, device)

        text = get_calibration_text(exp_config.eval_samples)
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)

        logger.info("Computing BASELINE perplexity...")
        base_ppl = compute_perplexity(model, tokenizer, text)
        logger.info("Baseline perplexity: %.4f", base_ppl)

        logger.info("Capturing baseline routing + KLD reference (%d tokens)...", kld_tokens)
        base_routing, base_handles = install_routing_capture(model)
        ref_logits = capture_reference_logits(model, input_ids, max_length=kld_tokens)
        remove_hooks(base_handles)

        if baseline_cache is not None:
            baseline_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "base_ppl": base_ppl,
                    "ref_logits": ref_logits,
                    "base_routing": base_routing,
                    "model_id": exp_config.model_id,
                    "eval_samples": exp_config.eval_samples,
                    "kld_tokens": kld_tokens,
                },
                baseline_cache,
            )
            logger.info("Saved baseline to %s", baseline_cache)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Starting distributed build with %d layers...", num_layers)
    orch = Orchestrator(
        exp_config=exp_config,
        workers_yaml=workers_yaml,
        model_config=model_config,
        output_dir=codebook_dir,
        num_layers=num_layers,
    )
    orch.run()

    meta = exp_config.model_dump()
    meta["model_config"] = {
        "num_experts": model_config.num_experts,
        "moe_intermediate_size": model_config.moe_intermediate_size,
        "hidden_size": model_config.hidden_size,
        "num_layers": num_layers,
    }
    (codebook_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Distributed build complete. Results in %s", codebook_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed codebook build across multiple machines"
    )
    parser.add_argument("--config", required=True, help="YAML experiment config file")
    parser.add_argument(
        "--workers", required=True, help="YAML worker config file (workers.yaml)"
    )
    parser.add_argument("--device", default="auto", help="Device for baseline computation")
    parser.add_argument(
        "--eval-samples", type=int, help="Number of calibration samples (overrides config)"
    )
    parser.add_argument(
        "--kld-tokens",
        type=int,
        default=2048,
        help="Number of tokens used for the KLD reference.",
    )
    parser.add_argument(
        "--baseline-cache",
        default="baseline.pt",
        help="Path to cache baseline. Set to empty string to disable.",
    )
    parser.add_argument("--output-dir", help="Directory to save codebooks (overrides config)")
    parser.add_argument(
        "--log-file",
        default="distributed_run.log",
        help="Path to log file. Set to empty string to disable.",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file) if args.log_file else None
    if log_file and not log_file.is_absolute():
        log_file = Path.cwd() / log_file
    setup_logging(log_file)

    exp_config = ExperimentConfig.from_yaml(args.config)
    if args.eval_samples is not None:
        exp_config.eval_samples = args.eval_samples
    if args.output_dir:
        exp_config.output_dir = Path(args.output_dir)

    output_dir = exp_config.output_dir
    baseline_cache = Path(args.baseline_cache) if args.baseline_cache else None

    run_distributed(
        exp_config=exp_config,
        device=args.device,
        workers_yaml=Path(args.workers),
        codebook_dir=output_dir,
        kld_tokens=args.kld_tokens,
        baseline_cache=baseline_cache,
    )


if __name__ == "__main__":
    main()
