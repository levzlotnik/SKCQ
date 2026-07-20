#!/usr/bin/env python3
"""VQ hyperparameter sweep worker.

Long-lived worker process: loads layer weights ONCE at startup, then pulls
VQ config jobs from the orchestrator and runs them in-process. Avoids the
~35s overhead of re-importing torch + re-extracting weights per config.

Key features:
  - Primary codebook caching: skips k-means for primaries that have already
    been trained (same layer/projection/spec). Cache is on-disk + in-memory.
  - Heartbeat thread: sends DeviceStats every 5s (VRAM + utilization from
    nvidia-smi/rocm-smi subprocess for GPU-wide view, not just PyTorch's).
  - WorkerInfoMessage on connect: device inventory (name, total VRAM).
  - DisableMessage handling: finish in-flight job, send result, exit cleanly.

Usage:
    python vq_worker.py --orchestrator jaguar:5555 \
        --model-id Qwen/Qwen3.6-35B-A3B --layer 24 --device cuda --name jaguar-3090

    # Multi-GPU: launch one process per device, pinned via env vars:
    HIP_VISIBLE_DEVICES=0 python vq_worker.py --name leopard-0 ...
    HIP_VISIBLE_DEVICES=1 python vq_worker.py --name leopard-1 ...
"""

# ruff: noqa: N806, N803  (W_raw / W_norm match experiments/weight_quant_error.py convention)
from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
import threading
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from skcq.protocol import (
    DeviceInfo,
    DeviceStats,
    DisableMessage,
    DoneMessage,
    HeartbeatMessage,
    Message,
    ReadyMessage,
    VQErrorMessage,
    VQJobMessage,
    VQResultsMessage,
    WorkerInfoMessage,
    recv_frame,
    send_frame,
)
from skcq.vq.cache import PrimaryCodebookCache
from skcq.vq.runner import integer_schemes, load_model_config, run_one_kmeans
from worker import build_layer_shard_map, extract_rows, resolve_device

logger = logging.getLogger("vq.worker")

# Heartbeat interval (seconds)
HEARTBEAT_INTERVAL = 5.0


# ---------------------------------------------------------------------------
# GPU monitoring (nvidia-smi / rocm-smi subprocess)
# ---------------------------------------------------------------------------


def _query_gpu_stats_nvidia() -> list[tuple[int, int, int, float]]:
    """Query nvidia-smi for per-GPU stats. Returns [(index, used_mb, total_mb, util_pct)]."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        results = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                used_mb = int(parts[1])
                total_mb = int(parts[2])
                util = float(parts[3])
                results.append((idx, used_mb, total_mb, util))
        return results
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


def _query_gpu_stats_rocm() -> list[tuple[int, int, int, float]]:
    """Query rocm-smi for per-GPU stats. Returns [(index, used_mb, total_mb, util_pct)]."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "gpu", "--json"],
            text=True,
            timeout=5,
        )

        data = json.loads(out)
        results = []
        for key, val in data.items():
            # rocm-smi keys: "card0", "card1", etc.
            if key.startswith("card"):
                idx = int(key[4:])
                vram_used = val.get("VRAM Total Used Memory (B)", 0)
                vram_total = val.get("VRAM Total Memory (B)", 0)
                util = val.get("GPU use (%)", 0)
                results.append(
                    (idx, vram_used // (1024 * 1024), vram_total // (1024 * 1024), float(util))
                )
        return results
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, ImportError):
        return []


def _query_gpu_stats() -> list[tuple[int, int, int, float]]:
    """Query GPU stats via subprocess. Tries nvidia-smi first, then rocm-smi."""
    stats = _query_gpu_stats_nvidia()
    if stats:
        return stats
    return _query_gpu_stats_rocm()


def get_device_inventory() -> list[DeviceInfo]:
    """Get static device info (called once on startup)."""
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append(
            DeviceInfo(
                index=i,
                name=props.name,
                total_vram_mb=int(props.total_memory / (1024 * 1024)),
            )
        )
    return devices


def get_device_stats() -> list[DeviceStats]:
    """Get live device stats (called every HEARTBEAT_INTERVAL)."""
    pytorch_count = torch.cuda.device_count()
    gpu_stats = _query_gpu_stats()  # GPU-wide (from smi subprocess)
    stats = []
    for i in range(pytorch_count):
        alloc_mb = int(torch.cuda.memory_allocated(i) / (1024 * 1024))
        reserved_mb = int(torch.cuda.memory_reserved(i) / (1024 * 1024))
        # Match GPU-wide stats by index (works when HIP/CUDA_VISIBLE_DEVICES
        # hasn't remapped indices — within this process, device 0 = the one
        # we were pinned to)
        used_mb = alloc_mb
        util = 0.0
        if i < len(gpu_stats):
            _, used_mb, _, util = gpu_stats[i]
        stats.append(
            DeviceStats(
                index=i,
                allocated_mb=alloc_mb,
                reserved_mb=reserved_mb,
                used_mb=used_mb,
                utilization_pct=util,
            )
        )
    return stats


# ---------------------------------------------------------------------------
# Integer baselines (computed once per projection)
# ---------------------------------------------------------------------------


def run_integer_baselines(W_raw: torch.Tensor, block_size: int, projection: str) -> list[dict]:
    """Compute all integer baselines for one projection."""
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


# ---------------------------------------------------------------------------
# VQ job processing (with primary codebook caching)
# ---------------------------------------------------------------------------


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
    cache: PrimaryCodebookCache | None,
) -> dict:
    """Run one VQ config in-process. Returns the CSV row dict."""
    cfg = job.config
    in_dim = in_dims[cfg.projection]
    W_raw = rows_map[cfg.projection]

    # Derive params for run_one_kmeans from VQConfig
    primary = cfg.primary
    residual_block_sizes = [r.block_size for r in cfg.residuals] if cfg.residuals else None
    residual_k = [r.K for r in cfg.residuals] if cfg.residuals else None
    metric = primary.metric or "cosine"
    scale_dtype = primary.scale_dtype or "bf16"
    sign_split = bool(primary.sign_split) if primary.sign_split is not None else False

    # Build cache key for primary (if caching is enabled)
    cache_key_str = None
    if cache is not None:
        cache_key_str = primary.cache_key(layer_idx, cfg.projection)

    result = run_one_kmeans(
        W_raw=W_raw,
        projection=cfg.projection,
        in_dim=in_dim,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        block_size=primary.block_size,
        K=primary.K,
        n_codebooks=cfg.n_codebooks,
        metric=metric,
        residual_k=residual_k,
        shared_codebook=job.shared,
        sign_split=sign_split,
        max_iters=job.kmeans_iters,
        scale_dtype=scale_dtype,
        residual_block_sizes=residual_block_sizes,
        codebook_bits=16,  # always original dtype
        layer_idx=layer_idx,
        device=device,
        chunk_budget_mb=chunk_budget_mb,
        primary_codebook_cache=cache,
        cache_key_str=cache_key_str,
    )
    return result


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------


class HeartbeatThread(threading.Thread):
    """Sends HeartbeatMessage every HEARTBEAT_INTERVAL seconds."""

    def __init__(self, sock: socket.socket, worker_name: str, shutdown_event: threading.Event):
        super().__init__(daemon=True, name="heartbeat")
        self.sock = sock
        self.worker_name = worker_name
        self.shutdown_event = shutdown_event

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                stats = get_device_stats()
                msg = HeartbeatMessage(worker_name=self.worker_name, devices=stats)
                send_frame(self.sock, msg)
            except (ConnectionError, OSError):
                return
            self.shutdown_event.wait(HEARTBEAT_INTERVAL)


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VQ hyperparameter sweep worker")
    parser.add_argument("--orchestrator", required=True, help="host:port (e.g. jaguar:5555)")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=24, help="Layer index")
    parser.add_argument("--device", default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--name", default="vq-worker", help="Worker name for logging")
    parser.add_argument("--chunk-budget-mb", type=int, default=2048)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("vq_cache"),
        help="Directory for primary codebook cache (set to empty string to disable)",
    )
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

    # Initialize primary codebook cache
    cache = PrimaryCodebookCache(args.cache_dir) if args.cache_dir else None

    # Load model config + extract layer weights ONCE at startup
    logger.info("Loading config for %s...", args.model_id)
    cfg = load_model_config(args.model_id)
    num_experts = cfg["num_experts"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["moe_intermediate_size"]

    logger.info("Extracting layer %d weights (one-time cost)...", args.layer)
    model_dir = Path(snapshot_download(args.model_id))
    layer_shards_map = build_layer_shard_map(model_dir)
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

    # Send WorkerInfoMessage (device inventory, once)
    devices = get_device_inventory()
    send_frame(
        sock, WorkerInfoMessage(worker_name=args.name, host=socket.gethostname(), devices=devices)
    )
    logger.info("Sent WorkerInfoMessage: %d devices", len(devices))

    # Start heartbeat thread
    shutdown_event = threading.Event()
    heartbeat = HeartbeatThread(sock, args.name, shutdown_event)
    heartbeat.start()

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
            if isinstance(msg, DisableMessage):
                logger.info("Orchestrator says disable — finishing in-flight and exiting")
                break
            if not isinstance(msg, VQJobMessage):
                logger.error("Unexpected message from orchestrator: %s", msg)
                continue

            job = msg
            logger.info(
                "Received job %s: %s bs=%d K=%d ncb=%d",
                job.config.id,
                job.config.projection,
                job.config.primary.block_size,
                job.config.primary.K,
                job.config.n_codebooks,
            )

            try:
                # Compute integer baselines once per projection
                extra_rows: list[dict] = []
                if job.config.projection not in int_baselines_done:
                    W_raw = rows_map[job.config.projection]
                    extra_rows = run_integer_baselines(
                        W_raw,
                        job.config.primary.block_size,
                        job.config.projection,
                    )
                    int_baselines_done.add(job.config.projection)
                    logger.info(
                        "Computed %d integer baselines for %s",
                        len(extra_rows),
                        job.config.projection,
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
                    cache,
                )
                send_frame(
                    sock,
                    VQResultsMessage(
                        config_id=job.config.id,
                        row=row,
                        extra_rows=extra_rows or None,
                    ),
                )
                ack = recv_frame(sock)
                if isinstance(ack, DoneMessage):
                    logger.info("Orchestrator says done after results — exiting")
                    break
                if isinstance(ack, DisableMessage):
                    logger.info("Orchestrator says disable after results — exiting")
                    break
                logger.info(
                    "Job %s complete (err=%.4f, bpw=%.3f)",
                    job.config.id,
                    row["rel_fro_err"],
                    row["bits_per_weight"],
                )
            except (RuntimeError, ValueError, KeyError, OSError) as e:
                logger.exception("Error processing job %s", job.config.id)
                send_frame(sock, VQErrorMessage(config_id=job.config.id, msg=str(e)))
    finally:
        shutdown_event.set()
        sock.close()
        logger.info("Worker shut down")


if __name__ == "__main__":
    main()
