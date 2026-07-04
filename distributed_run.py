#!/usr/bin/env python3
"""Distributed codebook build using Ray actors.

Computes the baseline on the local GPU, then distributes codebook building
across Ray actors on local and remote machines.

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
import yaml

import ray

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
    """Compute baseline locally, then distribute codebook build across Ray actors."""
    from transformers import AutoConfig

    from skcq.orchestrator import _save_layer_results
    from skcq.clustering import CodebookParams
    from skcq.ray_worker import WorkerActor

    hf_config = AutoConfig.from_pretrained(exp_config.model_id, trust_remote_code=True)
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    model_config = ModelConfig(
        num_experts=hf_config.num_experts,
        hidden_size=hf_config.hidden_size,
        moe_intermediate_size=hf_config.moe_intermediate_size,
    )
    num_layers = hf_config.num_hidden_layers

    # ---- Baseline computation ----
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

    # ---- Ray actor setup ----
    logger.info("Initializing Ray...")
    ray.init(address="auto", ignore_reinit_error=True)
    logger.info("Ray initialized.")

    # Parse workers.yaml
    with open(workers_yaml) as f:
        workers_raw = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parent.parent

    workers = workers_raw["workers"]

    # Build list of actor references
    actor_refs: list[ray.ObjectRef] = []
    for w in workers:
        name = w["name"]
        host = w["host"]
        device = w.get("device", "auto")
        chunk_budget_mb = w.get("chunk_budget_mb", 2048)
        venv_path = repo_root / w["venv"]

        kwargs: dict = {
            "model_id": exp_config.model_id,
            "device": device,
            "chunk_budget_mb": chunk_budget_mb,
        }

        if host in ("localhost", "127.0.0.1"):
            kwargs["runtime_env"] = {"py_executable": str(venv_path)}

        actor_ref = WorkerActor.options(
            num_gpus=1, max_restarts=3, max_task_retries=2
        ).remote(**kwargs)
        actor_refs.append(actor_ref)

        logger.info(
            "Created WorkerActor '%s' (host=%s, device=%s, chunk_budget_mb=%d)",
            name,
            host,
            device,
            chunk_budget_mb,
        )

    # Build job queue — skip layers that already have saved codebooks
    skipped: list[int] = []
    job_queue: list[tuple[int, CodebookParams]] = []
    for i in range(num_layers):
        layer_dir = codebook_dir / f"layer_{i}"
        if all((layer_dir / f"{n}.pt").exists() for n in ("gate", "up", "down")):
            skipped.append(i)
        else:
            job_queue.append((i, exp_config.params_for_layer(i)))

    if skipped:
        logger.info("Resuming: %d layers already saved, %d to build", len(skipped), len(job_queue))

    # Submit jobs round-robin across actors
    pending_refs: list[ray.ObjectRef] = []
    ref_to_layer: dict[ray.ObjectRef, int] = {}
    actor_idx = 0
    for layer_idx, params in job_queue:
        ref = actor_refs[actor_idx % len(actor_refs)].process_layer.remote(
            layer=layer_idx,
            params=params.model_dump(),
            num_experts=model_config.num_experts,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.moe_intermediate_size,
        )
        pending_refs.append(ref)
        ref_to_layer[ref] = layer_idx
        actor_idx += 1

    logger.info(
        "Submitted %d layer jobs across %d actors (%d skipped)",
        len(pending_refs),
        len(actor_refs),
        len(skipped),
    )

    # Wait for results as they complete
    completed = len(skipped)
    while pending_refs:
        done_refs, _, still_pending = ray.wait(pending_refs, num_returns=1, timeout=60)

        if not done_refs:
            logger.warning("ray.wait timed out — waiting for remaining %d jobs", len(pending_refs))
            continue

        for ref in done_refs:
            result = ray.get(ref)
            pending_refs.remove(ref)
            layer_idx = ref_to_layer.pop(ref)

            _save_layer_results(
                result,
                codebook_dir,
                layer_idx,
                model_config.hidden_size,
                model_config.moe_intermediate_size,
            )

            logger.info("Layer %d complete (%d/%d)", layer_idx, completed + 1, num_layers)
            completed += 1

            if completed >= num_layers:
                break

        if completed >= num_layers:
            break

    if completed < num_layers:
        logger.error("Only %d/%d layers completed", completed, num_layers)
        raise RuntimeError(f"Build incomplete: {completed}/{num_layers} layers finished")

    # Save metadata
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
