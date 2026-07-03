#!/usr/bin/env python3
"""Standalone worker: reads safetensors directly, builds codebooks, sends results.

No model loading, no transformers. Reads expert weight tensors from safetensors
shards via zero-copy mmap, extracts rows, runs build_codebook on the local GPU,
and sends CodebookResult objects back to the orchestrator over TCP.

Usage:
    python worker.py --orchestrator strix:5555 --model-id Qwen/Qwen3.6-35B-A3B --device auto

The worker pulls jobs from the orchestrator's queue (one job = one layer,
all 3 projections: gate, up, down). Results are sent as pickled CodebookResult
dicts with length-prefixed framing over TCP.

Requires PYTHONPATH to include the project root for skcq.clustering import.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from skcq.clustering import CodebookParams, CodebookResult, build_codebook

logger = logging.getLogger("skcq.worker")

# Reuse the same length-prefixed framing as the orchestrator.


def send_frame(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_frame(sock: socket.socket) -> Any:
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    data = _recv_exactly(sock, length)
    if data is None:
        return None
    return pickle.loads(data)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def build_layer_shard_map(model_dir: Path) -> dict[int, dict[str, str]]:
    """Parse model.safetensors.index.json → {layer_idx: {gate_up: shard, down: shard}}."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No safetensors index at {index_path}")

    with open(index_path) as f:
        idx = json.load(f)

    weight_map: dict[str, str] = idx["weight_map"]
    layer_shards: dict[int, dict[str, str]] = {}

    for key, shard in weight_map.items():
        if ".mlp.experts.gate_up_proj" in key:
            ln = _extract_layer_idx(key)
            if ln is not None:
                layer_shards.setdefault(ln, {})["gate_up"] = shard
        elif ".mlp.experts.down_proj" in key:
            ln = _extract_layer_idx(key)
            if ln is not None:
                layer_shards.setdefault(ln, {})["down"] = shard

    return layer_shards


def _extract_layer_idx(key: str) -> int | None:
    """Extract layer index from a safetensors key like 'model.language_model.layers.5.mlp...'."""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def extract_rows(
    model_dir: Path,
    layer_idx: int,
    layer_shards: dict[int, dict[str, str]],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    """Extract gate, up, down weight rows from safetensors for one layer.

    Returns:
        {"gate": (num_experts * intermediate_size, hidden_size),
         "up":   (num_experts * intermediate_size, hidden_size),
         "down": (num_experts * hidden_size, intermediate_size)}
    """
    shards = layer_shards[layer_idx]
    proj_rows: dict[str, torch.Tensor] = {}

    gate_up_path = model_dir / shards["gate_up"]
    with safe_open(gate_up_path, framework="pt", device="cpu") as f:
        gu_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        gate_up = f.get_tensor(gu_key)

    gate_rows = gate_up[:, :intermediate_size, :].reshape(-1, hidden_size)
    up_rows = gate_up[:, intermediate_size:, :].reshape(-1, hidden_size)
    proj_rows["gate"] = gate_rows
    proj_rows["up"] = up_rows

    down_path = model_dir / shards["down"]
    with safe_open(down_path, framework="pt", device="cpu") as f:
        dn_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        down = f.get_tensor(dn_key)

    down_rows = down.reshape(-1, intermediate_size)
    proj_rows["down"] = down_rows

    return proj_rows


def process_job(
    job: dict[str, Any],
    model_dir: Path,
    layer_shards_map: dict[int, dict[str, str]],
    device: torch.device,
) -> dict[str, CodebookResult]:
    """Process one layer job: extract rows, build codebooks for all 3 projections."""
    layer_idx = job["layer"]
    params = CodebookParams(**job["params"])
    num_experts = job["num_experts"]
    hidden_size = job["hidden_size"]
    intermediate_size = job["intermediate_size"]
    n_blocks_gu = params.n_blocks_gate_up
    n_blocks_dn = params.n_blocks_down
    n_codebooks = params.n_codebooks

    logger.info("Layer %d: extracting rows from safetensors...", layer_idx)
    rows_map = extract_rows(
        model_dir, layer_idx, layer_shards_map, num_experts, hidden_size, intermediate_size
    )

    results: dict[str, CodebookResult] = {}

    projections = [
        ("gate", rows_map["gate"], params.k_gate, n_blocks_gu, intermediate_size),
        ("up", rows_map["up"], params.k_up, n_blocks_gu, intermediate_size),
        ("down", rows_map["down"], params.k_down, n_blocks_dn, hidden_size),
    ]

    for name, rows, k, n_blocks, out_dim in projections:
        cb_name = f"L{layer_idx}.{name}"
        logger.info("Layer %d: building %s (k=%d, nb=%d, cb=%d)...",
                    layer_idx, name, k, n_blocks, n_codebooks)
        results[name] = build_codebook(
            rows=rows.to(device),
            params=params,
            k=k,
            n_blocks=n_blocks,
            n_codebooks=n_codebooks,
            num_experts=num_experts,
            out_dim=out_dim,
            device=device,
            name=cb_name,
        )

    logger.info("Layer %d: done.", layer_idx)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed codebook build worker")
    parser.add_argument(
        "--orchestrator", required=True, help="orchestrator host:port (e.g. strix:5555)"
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--device", default="auto", help="Device: cuda, mps, cpu, or auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    host, port_str = args.orchestrator.rsplit(":", 1)
    port = int(port_str)
    device = resolve_device(args.device)
    logger.info("Worker starting, device=%s, orchestrator=%s:%d", device, host, port)

    logger.info("Resolving model path for %s...", args.model_id)
    model_dir = Path(snapshot_download(args.model_id))
    logger.info("Model at %s", model_dir)

    logger.info("Building layer→shard map from safetensors index...")
    layer_shards_map = build_layer_shard_map(model_dir)
    logger.info("Found %d layers in safetensors index", len(layer_shards_map))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    logger.info("Connected to orchestrator at %s:%d", host, port)

    try:
        while True:
            send_frame(sock, {"type": "ready", "device": str(device)})
            job = recv_frame(sock)
            if job is None:
                logger.error("Orchestrator closed connection")
                break
            if job.get("type") == "done":
                logger.info("Orchestrator says done — exiting")
                break
            if job.get("type") != "job":
                logger.error("Unexpected message from orchestrator: %s", job)
                continue

            layer_idx = job["layer"]
            logger.info("Received job for layer %d", layer_idx)
            try:
                results = process_job(job, model_dir, layer_shards_map, device)
                send_frame(sock, {"type": "results", "layer": layer_idx, "data": results})
                ack = recv_frame(sock)
                if ack is not None and ack.get("type") == "done":
                    logger.info("Orchestrator says done after results — exiting")
                    break
            except (RuntimeError, ValueError, KeyError, OSError) as e:
                logger.exception("Error processing layer %d", layer_idx)
                send_frame(sock, {"type": "error", "layer": layer_idx, "msg": str(e)})
    finally:
        sock.close()
        logger.info("Worker shut down")


if __name__ == "__main__":
    main()
