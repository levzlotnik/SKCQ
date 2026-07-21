from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from skcq.codebook_experts import CodebookExperts, CodebookModule
from skcq.config import ExperimentConfig
from skcq.eval_model import compute_perplexity, get_calibration_text, get_text_model, load_model
from skcq.logging_setup import setup_logging
from skcq.measurements import Measurements, install_measurement_hooks
from skcq.metrics import (
    capture_reference_logits,
    install_routing_capture,
    remove_hooks,
)
from skcq.quantize import CodebookResult, extract_and_build_codebooks
from skcq.rocm_client import RocmClient
from skcq.tasks import run_tasks

logger = logging.getLogger("skcq")


def modules_from_results(
    codebook_results: list[dict[str, CodebookResult]],
) -> list[dict[str, CodebookModule]]:
    """Convert CodebookResults into CodebookModules (one per projection per layer)."""
    modules: list[dict[str, CodebookModule]] = []
    for layer_result in codebook_results:
        layer_modules: dict[str, CodebookModule] = {}
        for name in ["gate", "up", "down"]:
            result = layer_result[name]
            n_rows = result.scales.shape[0]
            assert result.num_experts is not None
            out_dim = n_rows // result.num_experts
            layer_modules[name] = CodebookModule.from_result(result, out_dim=out_dim)
        modules.append(layer_modules)
    return modules


def replace_experts(
    model: torch.nn.Module,
    codebook_modules: list[dict[str, CodebookModule]],
    num_experts: int,
) -> None:
    """Replace Qwen3_5MoeExperts modules with CodebookExperts in-place.

    ``codebook_modules`` is one CodebookModule dict (gate/up/down) per layer.
    """
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

        mlp.experts = codebook_experts.to(next(mlp.experts.parameters()).device)
        if (layer_idx + 1) % 5 == 0 or layer_idx + 1 == num_layers:
            logger.info("Replacing experts: %d/%d layers", layer_idx + 1, num_layers)


def save_codebooks(codebook_results: list[dict[str, CodebookResult]], output_dir: Path) -> None:
    """Save codebook results to disk as CodebookModule state_dicts + meta.

    Shapes (n_blocks, block_size, out_dim, num_experts, per-codebook K) are all
    inferred from the CodebookResult tensors, so no external dims are required.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_layers = len(codebook_results)
    for layer_idx, layer_result in enumerate(codebook_results):
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        for name in ["gate", "up", "down"]:
            result = layer_result[name]
            n_rows = result.scales.shape[0]
            assert result.num_experts is not None
            out_dim = n_rows // result.num_experts
            module = CodebookModule.from_result(result, out_dim=out_dim)
            payload = module.state_dict_with_meta()
            payload["zero_mask"] = result.zero_mask
            torch.save(payload, layer_dir / f"{name}.pt")
        if (layer_idx + 1) % 5 == 0 or layer_idx + 1 == num_layers:
            logger.info("Saving codebooks: %d/%d layers", layer_idx + 1, num_layers)


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


def text_model_config(model: torch.nn.Module) -> Any:
    """Return the inner text-config object, unwrapping multimodal wrappers."""
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    return config


def get_or_build_codebooks(
    model: torch.nn.Module,
    exp_config: ExperimentConfig,
    codebook_dir: Path | None,
    cuda_worker: RocmClient | None = None,
) -> list[dict[str, CodebookModule]]:
    """Load codebook modules from ``codebook_dir`` if present, else build and save them."""
    if codebook_dir is not None and (codebook_dir / "layer_0").exists():
        logger.info("Loading codebooks from %s...", codebook_dir)
        num_layers = text_model_config(model).num_hidden_layers
        return load_codebooks(codebook_dir, num_layers)

    logger.info("Building codebooks...")
    results = extract_and_build_codebooks(
        model,
        params=exp_config.defaults,
        layer_overrides=exp_config.layer_overrides,
        device=next(model.parameters()).device,
        cuda_worker=cuda_worker,
    )
    if codebook_dir is not None:
        logger.info("Saving codebooks to %s...", codebook_dir)
        save_codebooks(results, codebook_dir)
    return modules_from_results(results)


def run_baseline_and_build(
    exp_config: ExperimentConfig,
    device: str,
    codebook_dir: Path | None,
    kld_tokens: int,
    cuda_worker: RocmClient | None = None,
    baseline_cache: Path | None = None,
) -> None:
    """Compute baseline (ppl, routing, KLD ref) and build codebooks. Quantized eval
    is done separately on the 3090 via eval_quantized.py."""
    logger.info("Loading model for baseline + codebook build...")
    model, tokenizer = load_model(exp_config.model_id, device)

    text = get_calibration_text(exp_config.eval_samples)
    encodings = tokenizer(text, return_tensors="pt")  # type: ignore[operator]
    input_ids = encodings.input_ids.to(model.device)

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
            logger.info("Baseline ppl: %.4f (cached)", baseline["base_ppl"])

    if baseline_cache is not None and not baseline_cache.exists():
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

    get_or_build_codebooks(model, exp_config, codebook_dir, cuda_worker=cuda_worker)


def run_tasks_eval(
    exp_config: ExperimentConfig,
    device: str,
    codebook_dir: Path | None,
    tasks: list[str],
    num_examples: int,
    out_path: Path,
    cuda_worker: RocmClient | None = None,
) -> None:
    """Run downstream tasks on base (or quantized if codebook_dir given)."""
    logger.info("Loading model for task eval...")
    model, tokenizer = load_model(exp_config.model_id, device)
    config = text_model_config(model)

    quantized = codebook_dir is not None
    if quantized:
        codebook_modules = get_or_build_codebooks(
            model, exp_config, codebook_dir, cuda_worker=cuda_worker
        )
        logger.info("Replacing experts...")
        replace_experts(model, codebook_modules, num_experts=config.num_experts)

    results = run_tasks(model, tokenizer, tasks, num_examples=num_examples)
    summary = {
        "model_id": exp_config.model_id,
        "quantized": quantized,
        "codebook_dir": str(codebook_dir) if codebook_dir else None,
        "results": [
            {
                "task": r.task,
                "num_examples": r.num_examples,
                "accuracy": r.accuracy,
                "correct": r.correct,
            }
            for r in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote task results to %s", out_path)
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector Codebook Quantization for MoE models")
    parser.add_argument("--config", help="YAML experiment config file")
    parser.add_argument("--model-id", help="HuggingFace model ID (overrides config)")
    parser.add_argument(
        "--k-gate", type=int, help="Codebook size for gate projection (overrides config)"
    )
    parser.add_argument(
        "--k-up", type=int, help="Codebook size for up projection (overrides config)"
    )
    parser.add_argument(
        "--k-down", type=int, help="Codebook size for down projection (overrides config)"
    )
    parser.add_argument("--max-iters", type=int, help="Max k-means iterations (overrides config)")
    parser.add_argument(
        "--n-blocks-gate-up",
        type=int,
        help="Number of PQ sub-blocks for gate/up input dim (overrides config)",
    )
    parser.add_argument(
        "--n-blocks-down",
        type=int,
        help="Number of PQ sub-blocks for down input dim (overrides config)",
    )
    parser.add_argument(
        "--n-codebooks",
        type=int,
        help="Number of codebooks (1 = no residual, 2+ = primary + residuals, overrides config)",
    )
    parser.add_argument("--output-dir", help="Directory to save codebooks (overrides config)")
    parser.add_argument("--device", default="auto", help="Device for model loading")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't quantize")
    parser.add_argument("--codebook-dir", help="Directory to load codebooks from")
    parser.add_argument(
        "--eval-samples", type=int, help="Number of calibration samples (overrides config)"
    )
    parser.add_argument(
        "--log-file",
        default="run.log",
        help="Path to log file. Set to empty string to disable file logging.",
    )
    parser.add_argument(
        "--measurements",
        default="measurements.json",
        help="Path to dump measurements JSON. Set to empty string to disable.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base vs quantized model: ppl delta, token KLD, routing divergence.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="Run downstream multiple-choice tasks (e.g. mmlu hellaswag). "
        "Use with --eval-only/--codebook-dir for quantized eval, or alone for base eval.",
    )
    parser.add_argument(
        "--task-samples",
        type=int,
        default=200,
        help="Max examples per downstream task (applies per task).",
    )
    parser.add_argument(
        "--kld-tokens",
        type=int,
        default=2048,
        help="Number of tokens used for the base-vs-quant KLD reference.",
    )
    parser.add_argument(
        "--use-cuda-worker",
        action="store_true",
        help="Offload build_codebook to a CUDA subprocess (requires cuda/.venv).",
    )
    parser.add_argument(
        "--baseline-cache",
        default="baseline.pt",
        help="Path to cache baseline ppl/routing/KLD reference. Set to empty string to disable.",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file) if args.log_file else None
    if log_file and not log_file.is_absolute():
        log_file = Path.cwd() / log_file
    setup_logging(log_file)

    exp_config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()

    if args.model_id:
        exp_config.model_id = args.model_id
    if args.k_gate is not None:
        exp_config.defaults.k_gate = args.k_gate
    if args.k_up is not None:
        exp_config.defaults.k_up = args.k_up
    if args.k_down is not None:
        exp_config.defaults.k_down = args.k_down
    if args.max_iters is not None:
        exp_config.defaults.max_iters = args.max_iters
    if args.n_blocks_gate_up is not None:
        exp_config.defaults.n_blocks_gate_up = args.n_blocks_gate_up
    if args.n_blocks_down is not None:
        exp_config.defaults.n_blocks_down = args.n_blocks_down
    if args.n_codebooks is not None:
        exp_config.defaults.n_codebooks = args.n_codebooks
    if args.output_dir:
        exp_config.output_dir = Path(args.output_dir)
    if args.eval_samples is not None:
        exp_config.eval_samples = args.eval_samples

    output_dir = exp_config.output_dir

    measurements: Measurements | None = None
    hook_handles: list[torch.utils.hooks.RemovableHook] = []

    def _setup_measurements(model: torch.nn.Module, num_experts: int) -> None:
        nonlocal measurements, hook_handles
        if not args.measurements:
            return
        measurements = Measurements(num_experts=num_experts)
        hook_handles = install_measurement_hooks(model, measurements)
        logger.info("Measurement hooks installed (dumping to %s)", args.measurements)

    def _teardown_measurements() -> None:
        remove_hooks(hook_handles)
        if measurements is not None and args.measurements:
            out_path = Path(args.measurements)
            if not out_path.is_absolute():
                out_path = Path.cwd() / out_path
            measurements.dump_json(out_path)
            logger.info("Wrote measurements to %s", out_path)

    if args.compare:
        cb_dir: Path | None = (
            Path(args.codebook_dir) if args.codebook_dir else exp_config.output_dir
        )
        baseline_cache = Path(args.baseline_cache) if args.baseline_cache else None
        cuda_worker = RocmClient() if args.use_cuda_worker else None
        try:
            run_baseline_and_build(
                exp_config,
                args.device,
                cb_dir,
                args.kld_tokens,
                cuda_worker=cuda_worker,
                baseline_cache=baseline_cache,
            )
        finally:
            if cuda_worker is not None:
                cuda_worker.close()
        return

    if args.tasks:
        cb_dir = Path(args.codebook_dir) if args.codebook_dir else None
        out = Path("tasks.json")
        cuda_worker = RocmClient() if args.use_cuda_worker else None
        try:
            run_tasks_eval(
                exp_config,
                args.device,
                cb_dir,
                args.tasks,
                args.task_samples,
                out,
                cuda_worker=cuda_worker,
            )
        finally:
            if cuda_worker is not None:
                cuda_worker.close()
        return

    if args.eval_only and args.codebook_dir:
        logger.info("Loading model for evaluation...")
        model, tokenizer = load_model(exp_config.model_id, args.device)

        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config

        codebook_modules = load_codebooks(Path(args.codebook_dir), config.num_hidden_layers)

        replace_experts(model, codebook_modules, num_experts=config.num_experts)

        _setup_measurements(model, num_experts=config.num_experts)
        logger.info("Computing perplexity...")
        text = get_calibration_text(exp_config.eval_samples)
        ppl = compute_perplexity(model, tokenizer, text)
        logger.info("Perplexity: %.4f", ppl)
        _teardown_measurements()
        return

    logger.info("Loading model...")
    model, tokenizer = load_model(exp_config.model_id, args.device)

    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config

    cuda_worker = RocmClient() if args.use_cuda_worker else None

    logger.info(
        "Building codebooks (k_gate=%d, k_up=%d, k_down=%d, max_iters=%d)...",
        exp_config.defaults.k_gate,
        exp_config.defaults.k_up,
        exp_config.defaults.k_down,
        exp_config.defaults.max_iters,
    )
    try:
        codebook_results = extract_and_build_codebooks(
            model,
            params=exp_config.defaults,
            layer_overrides=exp_config.layer_overrides,
            device=next(model.parameters()).device,
            cuda_worker=cuda_worker,
        )
    finally:
        if cuda_worker is not None:
            cuda_worker.close()

    logger.info("Saving codebooks...")
    save_codebooks(codebook_results, output_dir)

    meta = exp_config.model_dump()
    meta["model_config"] = {
        "num_experts": config.num_experts,
        "moe_intermediate_size": config.moe_intermediate_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
    }
    (output_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))

    logger.info("Replacing experts...")
    codebook_modules = modules_from_results(codebook_results)
    replace_experts(model, codebook_modules, num_experts=config.num_experts)

    _setup_measurements(model, num_experts=config.num_experts)
    logger.info("Computing perplexity...")
    text = get_calibration_text(exp_config.eval_samples)
    ppl = compute_perplexity(model, tokenizer, text)
    logger.info("Perplexity: %.4f", ppl)
    _teardown_measurements()


if __name__ == "__main__":
    main()
