"""Distributed orchestrator: TCP server + job queue + worker launcher.

Runs on the main node (alongside baseline computation). Launches workers on
local and remote machines via SSH, distributes layer jobs, collects results,
and saves codebooks to disk.

Usage (via distributed_run.py --workers workers.yaml):
    The orchestrator is invoked by distributed_run.py. It computes the baseline
    first (if not cached), then launches workers and distributes the 40-layer
    build across all available machines.
"""

from __future__ import annotations

import logging
import os
import pickle
import queue
import socket
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any

import torch
import yaml

from skcq.clustering import CodebookResult
from skcq.codebook_experts import CodebookModule
from skcq.config import CodebookParams, ExperimentConfig

logger = logging.getLogger("skcq.orchestrator")


def _send_frame(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_frame(sock: socket.socket) -> Any:
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


def _merge_params(
    exp_config: ExperimentConfig, layer_idx: int
) -> CodebookParams:
    """Merge defaults with layer_overrides for a specific layer."""
    params = exp_config.defaults
    if layer_idx in exp_config.layer_overrides:
        override = exp_config.layer_overrides[layer_idx]
        updates = {k: v for k, v in override.model_dump().items() if v is not None}
        if updates:
            params = params.model_copy(update=updates)
    return params


def _build_job(
    exp_config: ExperimentConfig,
    model_id: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, Any]:
    params = _merge_params(exp_config, layer_idx)
    return {
        "type": "job",
        "layer": layer_idx,
        "model_id": model_id,
        "params": params.model_dump(),
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }


def _save_layer_results(
    results: dict[str, CodebookResult],
    output_dir: Path,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> None:
    """Save CodebookResults as state_dict per projection."""
    layer_dir = output_dir / f"layer_{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        if name == "gate" or name == "up":
            out_dim = intermediate_size
            in_dim = hidden_size
        else:  # down
            out_dim = hidden_size
            in_dim = intermediate_size

        block_size = in_dim // result.n_blocks
        module = CodebookModule.from_result(
            result, n_blocks=result.n_blocks, block_size=block_size, out_dim=out_dim
        )
        torch.save(module.state_dict_with_meta(), layer_dir / f"{name}.pt")

    logger.info("Saved layer %d results to %s", layer_idx, layer_dir)


class Orchestrator:
    def __init__(
        self,
        exp_config: ExperimentConfig,
        workers_yaml: Path,
        model_config: dict[str, Any],
        output_dir: Path,
        num_layers: int,
    ) -> None:
        self.exp_config = exp_config
        self.output_dir = output_dir
        self.num_layers = num_layers

        self.model_id = exp_config.model_id
        self.num_experts = model_config["num_experts"]
        self.hidden_size = model_config["hidden_size"]
        self.intermediate_size = model_config["moe_intermediate_size"]

        with open(workers_yaml) as f:
            self.workers_config = yaml.safe_load(f)

        self.port = self.workers_config.get("port", 5555)
        self.orchestrator_host = self.workers_config.get("orchestrator_host", "localhost")

        self.job_queue: queue.Queue[int | None] = queue.Queue()
        for i in range(num_layers):
            self.job_queue.put(i)
        self.job_queue.put(None)  # sentinel

        self.results_lock = threading.Lock()
        self.completed = 0
        self.layer_results: dict[int, dict[str, CodebookResult]] = {}
        self.worker_procs: list[subprocess.Popen] = []
        self.shutdown = threading.Event()

    def _launch_workers(self) -> None:
        """Launch worker processes (local via subprocess, remote via SSH)."""
        for w in self.workers_config["workers"]:
            name = w["name"]
            host = w["host"]
            venv = w["venv"]
            workdir = w.get("workdir", ".")
            device = w.get("device", "auto")

            if host == "localhost" or host == "127.0.0.1":
                cmd = [
                    venv,
                    str(Path(__file__).parent.parent / "worker.py"),
                    "--orchestrator",
                    f"localhost:{self.port}",
                    "--model-id",
                    self.model_id,
                    "--device",
                    device,
                ]
                env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
                proc = subprocess.Popen(cmd, cwd=workdir, env=env)
            else:
                remote_cmd = (
                    f"cd {workdir} && git pull && "
                    f"{venv} worker.py --orchestrator "
                    f"{self.orchestrator_host}:{self.port} --model-id {self.model_id} "
                    f"--device {device}"
                )
                cmd = ["ssh", host, remote_cmd]
                proc = subprocess.Popen(cmd)

            self.worker_procs.append(proc)
            logger.info("Launched worker %s (pid=%d, host=%s, device=%s)",
                        name, proc.pid, host, device)

    def _handle_worker(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        """Handle one worker connection: dispatch jobs, collect results."""
        worker_name = f"{addr[0]}:{addr[1]}"
        current_job: int | None = None

        try:
            while not self.shutdown.is_set():
                msg = _recv_frame(conn)
                if msg is None:
                    logger.warning("Worker %s disconnected", worker_name)
                    if current_job is not None:
                        logger.info("Re-queueing layer %d (worker disconnect)", current_job)
                        self.job_queue.put(current_job)
                    return

                if msg.get("type") == "ready":
                    job_idx = self.job_queue.get()
                    if job_idx is None:
                        _send_frame(conn, {"type": "done"})
                        return
                    current_job = job_idx
                    job = _build_job(
                        self.exp_config,
                        self.model_id,
                        job_idx,
                        self.num_experts,
                        self.hidden_size,
                        self.intermediate_size,
                    )
                    logger.info("Dispatching layer %d to %s", job_idx, worker_name)
                    _send_frame(conn, job)

                elif msg.get("type") == "results":
                    layer_idx = msg["layer"]
                    data = msg["data"]
                    with self.results_lock:
                        self.layer_results[layer_idx] = data
                        self.completed += 1
                        _save_layer_results(
                            data,
                            self.output_dir,
                            layer_idx,
                            self.hidden_size,
                            self.intermediate_size,
                            self.num_experts,
                        )
                    current_job = None
                    logger.info("Layer %d complete (%d/%d)",
                                layer_idx, self.completed, self.num_layers)
                    _send_frame(conn, {"type": "ack"})

                    if self.completed >= self.num_layers:
                        logger.info("All %d layers complete — shutting down workers",
                                    self.num_layers)
                        self.shutdown.set()
                        while not self.job_queue.empty():
                            self.job_queue.get()
                        return

                elif msg.get("type") == "error":
                    layer_idx = msg.get("layer", -1)
                    logger.error("Worker %s error on layer %d: %s",
                                 worker_name, layer_idx, msg.get("msg"))
                    if layer_idx >= 0 and current_job == layer_idx:
                        logger.info("Re-queueing layer %d (worker error)", layer_idx)
                        self.job_queue.put(layer_idx)
                        current_job = None

        except (ConnectionError, pickle.UnpicklingError, struct.error) as e:
            logger.warning("Worker %s connection failed: %s", worker_name, e)
            if current_job is not None:
                self.job_queue.put(current_job)
        finally:
            conn.close()

    def run(self) -> None:
        """Launch workers, accept connections, distribute jobs, collect results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(len(self.workers_config["workers"]))
        logger.info("Orchestrator listening on 0.0.0.0:%d", self.port)

        self._launch_workers()

        threads: list[threading.Thread] = []
        timeout = 600  # 10 min per worker connection

        while self.completed < self.num_layers and not self.shutdown.is_set():
            try:
                conn, addr = server.accept()
                conn.settimeout(timeout)
                t = threading.Thread(target=self._handle_worker, args=(conn, addr), daemon=True)
                t.start()
                threads.append(t)
            except OSError as e:
                if self.shutdown.is_set():
                    break
                logger.error("Accept failed: %s", e)

        self.shutdown.set()

        for t in threads:
            t.join(timeout=5)

        for proc in self.worker_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        server.close()
        logger.info("Orchestrator done: %d/%d layers complete",
                    self.completed, self.num_layers)

        if self.completed < self.num_layers:
            missing = set(range(self.num_layers)) - set(self.layer_results.keys())
            logger.error("Missing layers: %s", sorted(missing))
            raise RuntimeError(f"Only {self.completed}/{self.num_layers} layers completed")
