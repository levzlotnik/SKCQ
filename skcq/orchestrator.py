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
import queue
import socket
import subprocess
import threading
from pathlib import Path

import torch
import yaml

from skcq.clustering import CodebookResult
from skcq.codebook_experts import CodebookModule
from skcq.config import CodebookParams, ExperimentConfig
from skcq.protocol import (
    AckMessage,
    DoneMessage,
    ErrorMessage,
    JobMessage,
    Message,
    ModelConfig,
    ReadyMessage,
    ResultsMessage,
    WorkerConfig,
    WorkersConfig,
    recv_frame,
    send_frame,
)

logger = logging.getLogger("skcq.orchestrator")

LayerResults = dict[str, CodebookResult]


def _load_workers_config(path: Path) -> WorkersConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    workers = [
        WorkerConfig(
            name=w["name"],
            host=w["host"],
            venv=w["venv"],
            workdir=w.get("workdir", "."),
            device=w.get("device", "auto"),
            chunk_budget_mb=w.get("chunk_budget_mb", 2048),
        )
        for w in raw["workers"]
    ]
    return WorkersConfig(
        orchestrator_host=raw.get("orchestrator_host", "localhost"),
        port=raw.get("port", 5555),
        workers=workers,
    )


def _merge_params(exp_config: ExperimentConfig, layer_idx: int) -> CodebookParams:
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
    model_config: ModelConfig,
) -> JobMessage:
    params = _merge_params(exp_config, layer_idx)
    return JobMessage(
        layer=layer_idx,
        model_id=model_id,
        params=params.model_dump(),
        num_experts=model_config.num_experts,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.moe_intermediate_size,
    )


def _save_layer_results(
    results: LayerResults,
    output_dir: Path,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    """Save CodebookResults as state_dict per projection."""
    layer_dir = output_dir / f"layer_{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        out_dim = intermediate_size if name in ("gate", "up") else hidden_size
        module = CodebookModule.from_result(result, out_dim=out_dim)
        torch.save(module.state_dict_with_meta(), layer_dir / f"{name}.pt")

    logger.info("Saved layer %d results to %s", layer_idx, layer_dir)


class Orchestrator:
    def __init__(
        self,
        exp_config: ExperimentConfig,
        workers_yaml: Path,
        model_config: ModelConfig,
        output_dir: Path,
        num_layers: int,
    ) -> None:
        self.exp_config = exp_config
        self.output_dir = output_dir
        self.num_layers = num_layers
        self.model_id = exp_config.model_id
        self.model_config = model_config

        self.workers_config = _load_workers_config(workers_yaml)

        self.job_queue: queue.Queue[int] = queue.Queue()
        skipped: list[int] = []
        for i in range(num_layers):
            layer_dir = output_dir / f"layer_{i}"
            if all((layer_dir / f"{n}.pt").exists() for n in ("gate", "up", "down")):
                skipped.append(i)
            else:
                self.job_queue.put(i)
        if skipped:
            logger.info(
                "Resuming: %d layers already saved, %d to build",
                len(skipped),
                num_layers - len(skipped),
            )
        # No sentinel — workers get DoneMessage when all layers complete,
        # tracked via self.completed. A sentinel would strand re-queued jobs
        # behind it in the FIFO queue after a worker timeout.

        self.results_lock = threading.Lock()
        self.completed = len(skipped)
        self.layer_results: dict[int, LayerResults] = {}
        self.worker_procs: list[subprocess.Popen[bytes]] = []
        self.shutdown = threading.Event()

    def _launch_workers(self) -> None:
        """Launch worker processes (local via subprocess, remote via SSH)."""
        for w in self.workers_config.workers:
            if w.host == "localhost" or w.host == "127.0.0.1":
                cmd = [
                    w.venv,
                    str(Path(__file__).parent.parent / "worker.py"),
                    "--orchestrator",
                    f"localhost:{self.workers_config.port}",
                    "--model-id",
                    self.model_id,
                    "--device",
                    w.device,
                    "--name",
                    w.name,
                    "--chunk-budget-mb",
                    str(w.chunk_budget_mb),
                ]
                env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
                proc = subprocess.Popen(cmd, cwd=w.workdir, env=env)
            else:
                remote_cmd = (
                    f"cd {w.workdir} && git pull && "
                    f"{w.venv} worker.py --orchestrator "
                    f"{self.workers_config.orchestrator_host}:{self.workers_config.port} "
                    f"--model-id {self.model_id} --device {w.device} "
                    f"--name {w.name} --chunk-budget-mb {w.chunk_budget_mb}"
                )
                cmd = ["ssh", w.host, remote_cmd]
                proc = subprocess.Popen(cmd)

            self.worker_procs.append(proc)
            logger.info(
                "Launched worker %s (pid=%d, host=%s, device=%s)",
                w.name,
                proc.pid,
                w.host,
                w.device,
            )

    def _handle_worker(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        """Handle one worker connection: dispatch jobs, collect results."""
        worker_name = f"{addr[0]}:{addr[1]}"
        current_job: int | None = None

        try:
            while not self.shutdown.is_set():
                msg: Message | None = recv_frame(conn)
                if msg is None:
                    logger.warning("Worker %s disconnected", worker_name)
                    if current_job is not None:
                        logger.info("Re-queueing layer %d (worker disconnect)", current_job)
                        self.job_queue.put(current_job)
                    return

                if isinstance(msg, ReadyMessage):
                    while not self.shutdown.is_set():
                        try:
                            job_idx = self.job_queue.get(timeout=1)
                        except queue.Empty:
                            if self.completed >= self.num_layers:
                                send_frame(conn, DoneMessage())
                                return
                            continue
                        current_job = job_idx
                        break
                    if self.shutdown.is_set():
                        send_frame(conn, DoneMessage())
                        return
                    job = _build_job(
                        self.exp_config,
                        self.model_id,
                        job_idx,
                        self.model_config,
                    )
                    logger.info("Dispatching layer %d to %s", job_idx, worker_name)
                    send_frame(conn, job)

                elif isinstance(msg, ResultsMessage):
                    with self.results_lock:
                        self.layer_results[msg.layer] = msg.data
                        self.completed += 1
                    _save_layer_results(
                        msg.data,
                        self.output_dir,
                        msg.layer,
                        self.model_config.hidden_size,
                        self.model_config.moe_intermediate_size,
                    )
                    current_job = None
                    logger.info(
                        "Layer %d complete (%d/%d)",
                        msg.layer,
                        self.completed,
                        self.num_layers,
                    )
                    send_frame(conn, AckMessage())

                    if self.completed >= self.num_layers:
                        logger.info(
                            "All %d layers complete — shutting down workers",
                            self.num_layers,
                        )
                        self.shutdown.set()
                        while not self.job_queue.empty():
                            self.job_queue.get()
                        return

                elif isinstance(msg, ErrorMessage):
                    logger.error(
                        "Worker %s error on layer %d: %s",
                        worker_name,
                        msg.layer,
                        msg.msg,
                    )
                    if current_job == msg.layer:
                        logger.info("Re-queueing layer %d (worker error)", msg.layer)
                        self.job_queue.put(msg.layer)
                        current_job = None

        except (ConnectionError, OSError) as e:
            logger.warning("Worker %s connection failed: %s", worker_name, e)
            if current_job is not None:
                logger.info("Re-queueing layer %d (connection lost)", current_job)
                self.job_queue.put(current_job)
        finally:
            conn.close()

    def run(self) -> None:
        """Launch workers, accept connections, distribute jobs, collect results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.workers_config.port))
        server.listen(len(self.workers_config.workers))
        logger.info("Orchestrator listening on 0.0.0.0:%d", self.workers_config.port)

        self._launch_workers()

        threads: list[threading.Thread] = []
        timeout = 7200  # 2h per worker connection (large K layers take ~30 min)

        while self.completed < self.num_layers and not self.shutdown.is_set():
            try:
                server.settimeout(1.0)
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
        logger.info(
            "Orchestrator done: %d/%d layers complete",
            self.completed,
            self.num_layers,
        )

        if self.completed < self.num_layers:
            missing = set(range(self.num_layers)) - set(self.layer_results.keys())
            logger.error("Missing layers: %s", sorted(missing))
            raise RuntimeError(f"Only {self.completed}/{self.num_layers} layers completed")
