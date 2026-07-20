#!/usr/bin/env python3
"""VQ hyperparameter sweep orchestrator + CLI entry point.

Reads `vq_results/configs.json` (from scripts/gen_vq_hyperparams.py), launches
long-lived VQ workers (local + remote SSH), distributes configs as VQJobMessage
jobs, and collects results into a single SQLite file (`vq_results/results.db`).

Workers load layer-24 weights ONCE at startup, then loop over configs in-process
— avoiding the ~35s overhead per config that a subprocess-per-config design
would incur. Integer baselines are computed once per projection by the worker
and piggy-backed on the first VQ result for that projection.

Usage:
    uv run python scripts/run_vq_hyperparams.py
    uv run python scripts/run_vq_hyperparams.py --workers workers.yaml --layer 24
    uv run python scripts/run_vq_hyperparams.py --filter gate --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shlex
import socket
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import yaml

from skcq.protocol import (
    AckMessage,
    DoneMessage,
    Message,
    ReadyMessage,
    VQErrorMessage,
    VQJobMessage,
    VQResultsMessage,
    recv_frame,
    send_frame,
)

logger = logging.getLogger("vq.orchestrator")

REPO = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = REPO / "vq_worker.py"
RESULTS_DIR = REPO / "vq_results"
CONFIGS_PATH = RESULTS_DIR / "configs.json"
DB_PATH = RESULTS_DIR / "results.db"
STATUS_PATH = RESULTS_DIR / "status.json"

# CSV schema for results — matches experiments/weight_quant_error.py output
SCHEMA = [
    "config_id TEXT PRIMARY KEY",
    "projection TEXT NOT NULL",
    "scheme TEXT NOT NULL",
    "block_size INTEGER",
    "K INTEGER",
    "n_codebooks INTEGER",
    "metric TEXT",
    "shared INTEGER",
    "sign_split INTEGER",
    "scale_dtype TEXT",
    "kmeans_iters INTEGER",
    "residual_block_sizes TEXT",
    "rel_fro_err REAL",
    "bits_per_weight REAL",
    "compression_ratio REAL",
    "worker TEXT",
    "completed_at REAL",
]


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite DB, create table if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    cols = ", ".join(SCHEMA)
    conn.execute(f"CREATE TABLE IF NOT EXISTS results ({cols})")
    return conn


def insert_row(conn: sqlite3.Connection, row: dict, config_id: str, worker: str) -> None:
    """Insert a result row, ignoring duplicates (idempotent)."""
    # Normalize types
    safe = {
        "config_id": config_id,
        "projection": str(row.get("projection", "")),
        "scheme": str(row.get("scheme", "")),
        "block_size": int(row.get("block_size", 0) or 0),
        "K": int(row.get("K", 0) or 0),
        "n_codebooks": int(row.get("n_codebooks", 0) or 0),
        "metric": str(row.get("metric", "") or ""),
        "shared": 1 if row.get("shared") else 0,
        "sign_split": 1 if row.get("sign_split") else 0,
        "scale_dtype": str(row.get("scale_dtype", "") or ""),
        "kmeans_iters": int(row.get("kmeans_iters", 0) or 0),
        "residual_block_sizes": json.dumps(row.get("residual_block_sizes", []) or []),
        "rel_fro_err": float(row.get("rel_fro_err", 0.0) or 0.0),
        "bits_per_weight": float(row.get("bits_per_weight", 0.0) or 0.0),
        "compression_ratio": float(row.get("compression_ratio", 0.0) or 0.0),
        "worker": worker,
        "completed_at": time.time(),
    }
    placeholders = ", ".join(["?"] * len(safe))
    cols = ", ".join(safe.keys())
    conn.execute(
        f"INSERT OR IGNORE INTO results ({cols}) VALUES ({placeholders})",
        list(safe.values()),
    )
    conn.commit()


def already_done(conn: sqlite3.Connection, config_id: str) -> bool:
    """Idempotency: skip configs already in DB."""
    cur = conn.execute("SELECT 1 FROM results WHERE config_id = ?", (config_id,))
    return cur.fetchone() is not None


def load_workers(yaml_path: Path) -> list[dict]:
    """Load workers.yaml into a list of worker dicts."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("workers", [])


def build_vq_job(cfg: dict) -> VQJobMessage:
    """Build a VQJobMessage from a configs.json entry."""
    return VQJobMessage(
        config_id=cfg["id"],
        projection=cfg["projection"],
        block_size=cfg["block_size"],
        K=cfg["K"],
        n_codebooks=cfg["n_codebooks"],
        residual_block_sizes=cfg.get("residual_block_sizes"),
        residual_k=cfg.get("residual_k"),
        codebook_bits=cfg["codebook_bits"],
        metric="cosine",
        scale_dtype="int8",
        kmeans_iters=50,
        shared=True,
        sign_split=True,
    )


def launch_worker(
    w: dict, port: int, orchestrator_host: str, model_id: str, layer: int
) -> subprocess.Popen:
    """Launch one worker process. Local → subprocess, remote → SSH."""
    device = w.get("device", "auto")
    chunk_mb = w.get("chunk_budget_mb", 2048)
    name = w["name"]
    venv = w["venv"]
    workdir = w.get("workdir", ".")

    common_args = [
        "--orchestrator",
        f"{orchestrator_host}:{port}",
        "--model-id",
        model_id,
        "--layer",
        str(layer),
        "--device",
        device,
        "--name",
        name,
        "--chunk-budget-mb",
        str(chunk_mb),
    ]

    if w["host"] in ("localhost", "127.0.0.1"):
        cmd = [venv, str(WORKER_SCRIPT), *common_args]
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        return subprocess.Popen(cmd, cwd=workdir, env=env)
    else:
        remote_cmd = (
            f"cd {shlex.quote(workdir)} && git pull --quiet && "
            f"{shlex.quote(venv)} vq_worker.py " + " ".join(shlex.quote(a) for a in common_args)
        )
        return subprocess.Popen(["ssh", "-o", "ConnectTimeout=10", w["host"], remote_cmd])


class VQOrchestrator:
    def __init__(
        self,
        configs: list[dict],
        workers_yaml: Path,
        model_id: str,
        layer: int,
        port: int = 5555,
        db_path: Path = DB_PATH,
    ) -> None:
        self.configs = configs
        self.workers_yaml = workers_yaml
        self.model_id = model_id
        self.layer = layer
        self.port = port
        self.db = init_db(db_path)

        # Build queue, skipping configs already done
        self.queue: queue.Queue[str] = queue.Queue()
        skipped = 0
        for cfg in configs:
            if already_done(self.db, cfg["id"]):
                skipped += 1
            else:
                self.queue.put(cfg["id"])
        self.configs_by_id = {c["id"]: c for c in configs}
        self.total = len(configs)
        self.remaining = len(configs) - skipped
        logger.info(
            "Queue: %d total, %d already done, %d to run", self.total, skipped, self.remaining
        )

        self.completed = skipped
        self.failed: list[str] = []
        self.workers_done = 0
        self.shutdown = threading.Event()
        self.db_lock = threading.Lock()

        # Orchestrator host (Tailscale hostname for remote workers)
        self.orchestrator_host = self._resolve_orchestrator_host(workers_yaml)

    @staticmethod
    def _resolve_orchestrator_host(workers_yaml: Path) -> str:
        with open(workers_yaml) as f:
            cfg = yaml.safe_load(f)
        host = cfg.get("orchestrator_host")
        if host:
            return host
        # Fall back to this machine's hostname
        import socket as _sock

        return _sock.gethostname()

    def _write_status(self) -> None:
        status = {
            "total": self.total,
            "completed": self.completed,
            "failed": len(self.failed),
            "remaining": self.remaining,
            "workers_done": self.workers_done,
            "failed_ids": self.failed[:50],
            "updated_at": time.time(),
        }
        STATUS_PATH.write_text(json.dumps(status, indent=2))

    def _handle_worker(self, conn: socket.socket, addr) -> None:
        worker_name = f"{addr[0]}:{addr[1]}"
        current_job: str | None = None
        try:
            while not self.shutdown.is_set():
                msg: Message | None = recv_frame(conn)
                if msg is None:
                    logger.warning("Worker %s disconnected", worker_name)
                    if current_job is not None:
                        self.queue.put(current_job)
                    return

                if isinstance(msg, ReadyMessage):
                    # Wait for next available config
                    while not self.shutdown.is_set():
                        try:
                            job_id = self.queue.get(timeout=1)
                            break
                        except queue.Empty:
                            if self.completed + len(self.failed) >= self.total:
                                send_frame(conn, DoneMessage())
                                return
                            continue
                    else:
                        send_frame(conn, DoneMessage())
                        return

                    current_job = job_id
                    cfg = self.configs_by_id[job_id]
                    logger.info("Dispatching %s to %s", job_id, worker_name)
                    send_frame(conn, build_vq_job(cfg))

                elif isinstance(msg, VQResultsMessage):
                    with self.db_lock:
                        insert_row(self.db, msg.row, msg.config_id, worker_name)
                        # Insert any extra rows (integer baselines, etc.) —
                        # keyed by their scheme to avoid collisions.
                        for er in msg.extra_rows or []:
                            extra_id = f"int_baseline_{er.get('scheme', 'unknown')}"
                            insert_row(self.db, er, extra_id, worker_name)
                    # Count only the main result toward sweep completion.
                    self.completed += 1
                    send_frame(conn, AckMessage())
                    self._write_status()
                    logger.info(
                        "%s done (%d/%d) — %s",
                        msg.config_id,
                        self.completed,
                        self.total,
                        worker_name,
                    )
                    current_job = None
                    if self.completed + len(self.failed) >= self.total:
                        logger.info("All configs complete — shutting down")
                        self.shutdown.set()
                        return

                elif isinstance(msg, VQErrorMessage):
                    logger.error("Worker %s error on %s: %s", worker_name, msg.config_id, msg.msg)
                    if current_job == msg.config_id:
                        self.failed.append(msg.config_id)
                        current_job = None

        except (ConnectionError, OSError) as e:
            logger.warning("Worker %s connection failed: %s", worker_name, e)
            if current_job is not None:
                self.queue.put(current_job)
        finally:
            conn.close()

    def run(self) -> None:
        workers = load_workers(self.workers_yaml)
        logger.info("Launching %d workers from %s", len(workers), self.workers_yaml)

        procs = []
        for w in workers:
            proc = launch_worker(w, self.port, self.orchestrator_host, self.model_id, self.layer)
            procs.append(proc)
            logger.info(
                "Launched %s (pid=%d, host=%s, device=%s)",
                w["name"],
                proc.pid,
                w["host"],
                w.get("device", "auto"),
            )

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(len(workers) + 4)
        logger.info("Orchestrator listening on 0.0.0.0:%d", self.port)

        threads: list[threading.Thread] = []
        try:
            while not self.shutdown.is_set() and self.completed + len(self.failed) < self.total:
                try:
                    server.settimeout(2.0)
                    conn, addr = server.accept()
                    t = threading.Thread(target=self._handle_worker, args=(conn, addr), daemon=True)
                    t.start()
                    threads.append(t)
                except TimeoutError:
                    continue
                except OSError as e:
                    if self.shutdown.is_set():
                        break
                    logger.error("Accept failed: %s", e)
        finally:
            self.shutdown.set()
            for t in threads:
                t.join(timeout=5)
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            server.close()
            self._write_status()

        logger.info(
            "Sweep complete: %d/%d done, %d failed",
            self.completed,
            self.total,
            len(self.failed),
        )
        self.db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VQ hyperparameter sweep orchestrator")
    parser.add_argument(
        "--configs",
        type=Path,
        default=CONFIGS_PATH,
        help="Path to configs.json from gen_vq_hyperparams.py",
    )
    parser.add_argument("--workers", type=Path, default=REPO / "workers.yaml")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run configs whose id matches this substring (for testing)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max configs to run (for testing)")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    if not args.configs.exists():
        print(f"Config file not found: {args.configs}")
        print("Run: uv run python scripts/gen_vq_hyperparams.py")
        return

    with open(args.configs) as f:
        configs = json.load(f)

    if args.filter:
        configs = [c for c in configs if args.filter in c["id"]]
    if args.limit:
        configs = configs[: args.limit]

    print(f"Configs: {len(configs)}")
    orch = VQOrchestrator(
        configs=configs,
        workers_yaml=args.workers,
        model_id=args.model,
        layer=args.layer,
        port=args.port,
        db_path=args.db,
    )
    orch.run()


if __name__ == "__main__":
    main()
