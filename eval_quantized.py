#!/usr/bin/env python3
"""Run quantized eval on the 3090.

Usage:
    cuda/.venv/bin/python eval_quantized.py --config configs/sweep/kr0.0312_nb1_cb2.yaml \\
        --eval-samples 100 --kld-tokens 256

Requires PYTHONPATH to include the project root.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from skcq.codebook_experts import CodebookExperts, CodebookModule
from skcq.eval_model import (
    compute_perplexity,
    get_calibration_text,
    get_text_model,
    load_model,
)
from skcq.logging_setup import setup_logging
from skcq.metrics import (
    install_routing_capture,
    kld_against_reference,
    remove_hooks,
    routing_agreement,
)

logger = logging.getLogger("skcq")


def text_model_config(model: torch.nn.Module) -> Any:
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    return config


def load_codebooks(codebook_dir: Path, num_layers: int) -> list[dict[str, CodebookModule]]:
    """Load codebook CodebookModules from disk via state_dict."""
    codebook_modules: list[dict[str, CodebookModule]] = []
    logger.info("Loading %d codebook layers from %s...", num_layers, codebook_dir)
    for layer_idx in range(num_layers):
        layer_dir = codebook_dir / f"layer_{layer_idx}"
        layer_modules: dict[str, CodebookModule] = {}
        for name in ["gate", "up", "down"]:
            layer_modules[name] = CodebookModule.load(layer_dir / f"{name}.pt")
        codebook_modules.append(layer_modules)
        if (layer_idx + 1) % 10 == 0 or layer_idx + 1 == num_layers:
            logger.info("Loading codebooks: %d/%d layers", layer_idx + 1, num_layers)
    return codebook_modules


def replace_experts(
    model: torch.nn.Module,
    codebook_modules: list[dict[str, CodebookModule]],
    num_experts: int,
) -> None:
    text_model = get_text_model(model)
    num_layers = len(text_model.layers)
    logger.info("Replacing experts across %d layers...", num_layers)
    for layer_idx, layer in enumerate(text_model.layers):
        mlp = layer.mlp
        act_fn = mlp.experts.act_fn

        codebook_experts = CodebookExperts(
            gate=codebook_modules[layer_idx]["gate"],
            up=codebook_modules[layer_idx]["up"],
            down=codebook_modules[layer_idx]["down"],
            num_experts=num_experts,
            act_fn=act_fn,
        )

        mlp.experts = codebook_experts.to(next(mlp.parameters()).device)
        if (layer_idx + 1) % 5 == 0 or layer_idx + 1 == num_layers:
            logger.info("Replacing experts: %d/%d layers", layer_idx + 1, num_layers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantized eval on CUDA GPU")
    parser.add_argument("--config", required=True, help="YAML experiment config file")
    parser.add_argument("--codebook-dir", required=True, help="Directory with codebooks")
    parser.add_argument("--baseline-cache", default="baseline.pt", help="Baseline cache file")
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--kld-tokens", type=int, default=2048)
    parser.add_argument("--output", default="compare_cuda.json", help="Output JSON path")
    parser.add_argument("--log-file", default="eval_cuda.log", help="Log file path")
    args = parser.parse_args()

    log_file = Path(args.log_file) if args.log_file else None
    if log_file and not log_file.is_absolute():
        log_file = Path.cwd() / log_file
    setup_logging(log_file)

    from skcq.config import ExperimentConfig

    exp_config = ExperimentConfig.from_yaml(args.config)

    logger.info("Loading model (CPU first)...")
    model, tokenizer = load_model(exp_config.model_id, "cpu")
    config = text_model_config(model)
    num_experts: int = config.num_experts
    num_layers: int = config.num_hidden_layers

    codebook_modules = load_codebooks(Path(args.codebook_dir), num_layers)
    logger.info("Replacing experts...")
    replace_experts(model, codebook_modules, num_experts=num_experts)

    logger.info("Transferring model to cuda...")
    model = model.to("cuda")
    model.eval()
    logger.info("Model on cuda.")

    text = get_calibration_text(args.eval_samples)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    # Load baseline
    baseline_path = Path(args.baseline_cache)
    if not baseline_path.exists():
        logger.error("Baseline cache not found: %s", baseline_path)
        return
    logger.info("Loading baseline from %s...", baseline_path)
    baseline = torch.load(baseline_path, weights_only=False)
    cached_kld = baseline.get("kld_tokens", 0)
    if cached_kld != args.kld_tokens:
        logger.error(
            "Baseline cache has kld_tokens=%d but requested %d. "
            "Re-run baseline with --kld-tokens %d on the ROCm side first.",
            cached_kld,
            args.kld_tokens,
            args.kld_tokens,
        )
        return
    base_ppl = baseline["base_ppl"]
    ref_logits = baseline["ref_logits"]
    base_routing = baseline["base_routing"]
    logger.info("Baseline ppl: %.4f (cached)", base_ppl)

    logger.info("Computing QUANTIZED perplexity...")
    quant_ppl = compute_perplexity(model, tokenizer, text)
    logger.info("Quantized perplexity: %.4f", quant_ppl)

    logger.info("Computing token KLD vs baseline reference...")
    kld = kld_against_reference(model, input_ids, ref_logits)
    logger.info("KL(base || quant) = %.6f", kld)

    logger.info("Capturing quantized routing...")
    quant_routing, quant_handles = install_routing_capture(model)
    with torch.no_grad():
        model(input_ids[:, : args.kld_tokens])
    remove_hooks(quant_handles)

    logger.info("Computing routing divergence...")
    routing_summary = routing_agreement(base_routing, quant_routing)

    summary = {
        "model_id": exp_config.model_id,
        "baseline_ppl": base_ppl,
        "quantized_ppl": quant_ppl,
        "ppl_ratio": quant_ppl / base_ppl if base_ppl else None,
        "ppl_delta": quant_ppl - base_ppl,
        "kld_base_to_quant": kld,
        "kld_tokens": args.kld_tokens,
        "routing": routing_summary,
        "codebook_params": exp_config.defaults.model_dump(),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote comparison summary to %s", out_path)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
