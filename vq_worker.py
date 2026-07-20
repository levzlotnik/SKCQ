#!/usr/bin/env python3
"""VQ hyperparameter sweep worker.

Long-lived worker process: loads layer weights ONCE at startup, then pulls
VQ config jobs from the orchestrator and runs them in-process. Avoids the
~35s overhead of re-importing torch + re-extracting weights per config that
a subprocess-per-config design would incur.

Usage:
    python vq_worker.py --orchestrator jaguar:5555 \
        --model-id Qwen/Qwen3.6-35B-A3B --layer 24 --device cuda --name jaguar-3090

Protocol (over TCP, length-prefixed pickle, see skcq/protocol.py):
    Worker → Orch: ReadyMessage(device)
    Orch → Worker: VQJobMessage | DoneMessage
    Worker → Orch: VQResultsMessage(row) | VQErrorMessage
    Orch → Worker: AckMessage | DoneMessage
"""

# ruff: noqa: N806,N803  (W_raw / W_norm / W_q match experiments/weight_quant_error.py convention)
from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

# experiments/ is not a package — add to sys.path for run_one_kmeans import
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from experiments.weight_quant_error import (
    integer_schemes,
    load_model_config,
    run_one_kmeans,
)
from skcq.protocol import (
    DoneMessage,
    Message,
    ReadyMessage,
    VQErrorMessage,
    VQJobMessage,
    VQResultsMessage,
    recv_frame,
    send_frame,
)
from worker import build_layer_shard_map, extract_rows, resolve_device

logger = logging.getLogger("vq.worker")


def run_integer_baselines(W_raw: torch.Tensor, block_size: int, projection: str) -> list[dict]:
    """Compute all integer baselines for one projection. Returns list of CSV-row dicts."""
    rows = []
    W_norm = W_raw.float().norm().item()
    for name, quant_fn in integer_schemes(block_size):
        W_q, bpw = quant_fn(W_raw)
        err = torch.norm(W_raw.float() - W_q.float()).item() / W_norm
        rows.append(
            {
                "projection": projection,
                "scheme": name,
                "block_size": block_size,
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
        del W_q
    return rows


def process_vq_job(
    job: VQJobMessage,
    rows_map: dict[str, torch.Tensor],
    in_dims: dict[str, int],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    layer_idx: int,
    device: torch.device,
    chunk_budget_mb: int,
) -> dict:
    """Run one VQ config in-process. Returns the CSV row dict."""
    in_dim = in_dims[job.projection]
    W_raw = rows_map[job.projection]
    result = run_one_kmeans(
        W_raw=W_raw,
        projection=job.projection,
        in_dim=in_dim,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        block_size=job.block_size,
        K=job.K,
        n_codebooks=job.n_codebooks,
        metric=job.metric,
        residual_k=job.residual_k,
        shared_codebook=job.shared,
        sign_split=job.sign_split,
        max_iters=job.kmeans_iters,
        scale_dtype=job.scale_dtype,
        residual_block_sizes=job.residual_block_sizes,
        codebook_bits=job.codebook_bits,
        layer_idx=layer_idx,
        device=device,
        chunk_budget_mb=chunk_budget_mb,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VQ hyperparameter sweep worker")
    parser.add_argument("--orchestrator", required=True, help="host:port (e.g. jaguar:5555)")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=24, help="Layer index")
    parser.add_argument("--device", default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--name", default="vq-worker", help="Worker name for logging")
    parser.add_argument("--chunk-budget-mb", type=int, default=2048)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{args.name}] %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    host, port_str = args.orchestrator.rsplit(":", 1)
    port = int(port_str)
    device = resolve_device(args.device)
    logger.info("VQ worker starting, device=%s, orchestrator=%s:%d", device, host, port)

    # Load model config + extract layer weights ONCE at startup
    logger.info("Loading config for %s...", args.model_id)
    cfg = load_model_config(args.model_id)
    num_experts = cfg["num_experts"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["moe_intermediate_size"]

    logger.info("Extracting layer %d weights (one-time cost)...", args.layer)
    model_dir = Path(snapshot_download(args.model_id))
    layer_shards_map = build_layer_shard_map(model_dir)
    # Keep weights on CPU — run_one_kmeans moves chunks to device as needed
    # (avoids holding all 3 projections on GPU simultaneously)
    rows_map = extract_rows(
        model_dir,
        args.layer,
        layer_shards_map,
        num_experts,
        hidden_size,
        intermediate_size,
    )
    in_dims = {"gate": hidden_size, "up": hidden_size, "down": intermediate_size}
    logger.info(
        "Layer %d loaded: gate=%s, up=%s, down=%s",
        args.layer,
        tuple(rows_map["gate"].shape),
        tuple(rows_map["up"].shape),
        tuple(rows_map["down"].shape),
    )

    # Connect to orchestrator
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    logger.info("Connected to orchestrator at %s:%d", host, port)

    # Track which projections we've already computed integer baselines for
    int_baselines_done: set[str] = set()

    try:
        while True:
            send_frame(sock, ReadyMessage(device=str(device)))
            msg: Message | None = recv_frame(sock)
            if msg is None:
                logger.error("Orchestrator closed connection")
                break
            if isinstance(msg, DoneMessage):
                logger.info("Orchestrator says done — exiting")
                break
            if not isinstance(msg, VQJobMessage):
                logger.error("Unexpected message from orchestrator: %s", msg)
                continue

            job = msg
            logger.info(
                "Received job %s: %s bs=%d K=%d cb=%d",
                job.config_id,
                job.projection,
                job.block_size,
                job.K,
                job.n_codebooks,
            )

            try:
                # Compute integer baselines once per projection (cheap, avoids
                # redundant recompute across configs that share a projection).
                # We piggy-back on the first job for each projection: send the
                # baseline rows alongside the first VQ result.
                extra_rows: list[dict] = []
                if job.projection not in int_baselines_done:
                    W_raw = rows_map[job.projection]
                    extra_rows = run_integer_baselines(
                        W_raw,
                        job.block_size,
                        job.projection,
                    )
                    int_baselines_done.add(job.projection)
                    logger.info(
                        "Computed %d integer baselines for %s", len(extra_rows), job.projection
                    )

                # Run the VQ config
                row = process_vq_job(
                    job,
                    rows_map,
                    in_dims,
                    num_experts,
                    hidden_size,
                    intermediate_size,
                    args.layer,
                    device,
                    args.chunk_budget_mb,
                )
                # Send main result + any int baselines (as one message, so the
                # orchestrator can't shut down between them).
                send_frame(
                    sock,
                    VQResultsMessage(
                        config_id=job.config_id,
                        row=row,
                        extra_rows=extra_rows or None,
                    ),
                )
                # Single Ack for the whole message.
                ack = recv_frame(sock)
                if isinstance(ack, DoneMessage):
                    logger.info("Orchestrator says done after results — exiting")
                    return
                logger.info(
                    "Job %s complete (err=%.4f, bpw=%.3f)",
                    job.config_id,
                    row["rel_fro_err"],
                    row["bits_per_weight"],
                )
            except (RuntimeError, ValueError, KeyError, OSError) as e:
                logger.exception("Error processing job %s", job.config_id)
                send_frame(sock, VQErrorMessage(config_id=job.config_id, msg=str(e)))
    finally:
        sock.close()
        logger.info("Worker shut down")


if __name__ == "__main__":
    main()
