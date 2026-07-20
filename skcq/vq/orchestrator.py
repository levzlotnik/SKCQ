"""VQ sweep orchestrator: TCP server + job queue + worker management.

Holds a VQHyperparamRange, iterates it to populate the job queue. Accepts
worker connections, dispatches VQJobMessages, collects VQResultsMessages,
writes to SQLite. Tracks per-worker heartbeats (ring buffer) and device info.

This module is the orchestration core — no HTTP, no dashboard. The Flask
server (skcq.vq.server) wraps this class and adds the control plane API.
"""

from __future__ import annotations

import logging
import os
import queue
import shlex
import socket
import sqlite3
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import yaml

from skcq.protocol import (
    AckMessage,
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
from skcq.vq.hyperparams import VQConfig, VQHyperparamRange

logger = logging.getLogger("skcq.vq.orchestrator")

REPO = Path(__file__).resolve().parent.parent.parent
WORKER_SCRIPT = REPO / "skcq" / "vq" / "worker.py"

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

HEARTBEAT_BUFFER_SIZE = 60  # 60 samples × 5s = 5min of history


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"CREATE TABLE IF NOT EXISTS results ({', '.join(SCHEMA)})")
    return conn


def insert_row(conn: sqlite3.Connection, row: dict, config_id: str, worker: str) -> None:
    import json as _json

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
        "residual_block_sizes": _json.dumps(row.get("residual_block_sizes", []) or []),
        "rel_fro_err": float(row.get("rel_fro_err", 0.0) or 0.0),
        "bits_per_weight": float(row.get("bits_per_weight", 0.0) or 0.0),
        "compression_ratio": float(row.get("compression_ratio", 0.0) or 0.0),
        "worker": worker,
        "completed_at": time.time(),
    }
    placeholders = ", ".join(["?"] * len(safe))
    cols = ", ".join(safe.keys())
    conn.execute(
        f"INSERT OR IGNORE INTO results ({cols}) VALUES ({placeholders})", list(safe.values())
    )
    conn.commit()


def already_done(conn: sqlite3.Connection, config_id: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM results WHERE config_id = ?", (config_id,)).fetchone()
        is not None
    )


def load_workers_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_orchestrator_host(yaml_path: Path) -> str:
    cfg = load_workers_yaml(yaml_path)
    return cfg.get("orchestrator_host") or socket.gethostname()


def launch_worker_process(
    w: dict, port: int, orchestrator_host: str, model_id: str, layer: int
) -> subprocess.Popen:
    device = w.get("device", "auto")
    chunk_mb = w.get("chunk_budget_mb", 2048)
    name = w["name"]
    venv = w["venv"]
    workdir = w.get("workdir", ".")

    common = [
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
        cmd = [venv, "-m", "skcq.vq.worker", *common]
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        return subprocess.Popen(cmd, cwd=workdir, env=env)
    else:
        remote = (
            f"cd {shlex.quote(workdir)} && git pull --quiet && "
            f"{shlex.quote(venv)} -m skcq.vq.worker " + " ".join(shlex.quote(a) for a in common)
        )
        return subprocess.Popen(["ssh", "-o", "ConnectTimeout=10", w["host"], remote])


class WorkerState:
    """Per-worker state tracked by the orchestrator."""

    def __init__(self, name: str, host: str) -> None:
        self.name = name
        self.host = host
        self.enabled = True
        self.connected = False
        self.conn: socket.socket | None = None
        self.devices: list[dict] = []  # DeviceInfo as dict
        self.heartbeats: deque = deque(maxlen=HEARTBEAT_BUFFER_SIZE)
        self.current_job: str | None = None
        self.proc: subprocess.Popen | None = None


class VQOrchestrator:
    def __init__(
        self,
        range_: VQHyperparamRange,
        workers_yaml: Path,
        model_id: str,
        layer: int,
        port: int = 5555,
        db_path: Path | None = None,
    ) -> None:
        self.range = range_
        self.workers_yaml = workers_yaml
        self.model_id = model_id
        self.layer = layer
        self.port = port
        self.db_path = db_path or REPO / "vq_results" / "results.db"
        self.db = init_db(self.db_path)
        self.orchestrator_host = _resolve_orchestrator_host(workers_yaml)

        self.job_queue: queue.Queue[str] = queue.Queue()
        self.configs_by_id: dict[str, VQConfig] = {}
        self.total = 0
        self.completed = 0
        self.failed: list[str] = []
        self.shutdown = threading.Event()
        self.pause_event = threading.Event()
        self.state = "idle"
        self.db_lock = threading.Lock()
        self.workers_state: dict[str, WorkerState] = {}
        self.server_sock: socket.socket | None = None
        self.worker_procs: list[subprocess.Popen] = []

        self._rebuild_queue()
        self._init_workers_from_yaml()

    def _init_workers_from_yaml(self) -> None:
        """Pre-populate workers_state from YAML so dashboard shows workers before launch."""
        raw = load_workers_yaml(self.workers_yaml)
        for w in raw.get("workers", []):
            devices = w.get("devices", [])
            if devices:
                for dev_idx in devices:
                    name = f"{w['name']}-gpu{dev_idx}"
                    ws = WorkerState(name=name, host=w.get("host", "localhost"))
                    self.workers_state[name] = ws
            else:
                name = w["name"]
                ws = WorkerState(name=name, host=w.get("host", "localhost"))
                self.workers_state[name] = ws

    def _rebuild_queue(self) -> None:
        """Rebuild the job queue from the range, skipping already-done configs."""
        self.job_queue = queue.Queue()
        self.configs_by_id = {}
        self.total = 0
        for cfg in self.range:
            self.configs_by_id[cfg.id] = cfg
            self.total += 1
            if not already_done(self.db, cfg.id):
                self.job_queue.put(cfg.id)
        self.completed = self.total - self.job_queue.qsize() - len(self.failed)
        logger.info(
            "Queue: %d total, %d done, %d to run",
            self.total,
            self.completed,
            self.job_queue.qsize(),
        )

    # --- control plane ---

    def launch(self) -> str:
        if self.state in ("running", "paused"):
            return self.state
        self._rebuild_queue()
        self.state = "running"
        self.shutdown.clear()
        self.pause_event.clear()
        threading.Thread(target=self._run, daemon=True, name="orch").start()
        return self.state

    def pause(self) -> str:
        if self.state != "running":
            return self.state
        self.pause_event.set()
        self.state = "paused"
        return self.state

    def resume(self) -> str:
        if self.state != "paused":
            return self.state
        self.pause_event.clear()
        self.state = "running"
        return self.state

    def requeue_failed(self) -> int:
        n = len(self.failed)
        for cid in self.failed:
            self.job_queue.put(cid)
        self.failed.clear()
        return n

    def shutdown_now(self) -> str:
        self.pause_event.clear()
        self.shutdown.set()
        self.state = "stopped"
        for ws in self.workers_state.values():
            if ws.conn and ws.enabled and ws.connected:
                try:
                    send_frame(ws.conn, DisableMessage())
                except OSError:
                    ws.connected = False
                    logger.debug("Worker %s already disconnected during shutdown", ws.name)
        time.sleep(1)
        for proc in self.worker_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.state = "idle"
        return self.state

    def enable_worker(self, name: str) -> str:
        ws = self.workers_state.get(name)
        if ws is None:
            return f"unknown worker: {name}"
        ws.enabled = True
        return "enabled"

    def disable_worker(self, name: str) -> str:
        ws = self.workers_state.get(name)
        if ws is None:
            return f"unknown worker: {name}"
        ws.enabled = False
        if ws.conn and ws.connected:
            try:
                send_frame(ws.conn, DisableMessage())
            except OSError:
                ws.connected = False
                logger.debug("Worker %s already disconnected during disable", name)
        return "disabled"

    # --- TCP server ---

    def _run(self) -> None:
        raw_workers = load_workers_yaml(self.workers_yaml)
        workers = raw_workers.get("workers", [])

        # Expand multi-device workers into per-device entries
        expanded = []
        for w in workers:
            devices = w.get("devices", [])
            if devices:
                for dev_idx in devices:
                    w2 = dict(w)
                    w2["name"] = f"{w['name']}-gpu{dev_idx}"
                    w2["devices"] = [dev_idx]
                    expanded.append(w2)
            else:
                expanded.append(w)

        for w in expanded:
            proc = launch_worker_process(
                w, self.port, self.orchestrator_host, self.model_id, self.layer
            )
            self.worker_procs.append(proc)
            ws = self.workers_state.get(w["name"])
            if ws is None:
                ws = WorkerState(name=w["name"], host=w["host"])
                self.workers_state[w["name"]] = ws
            ws.proc = proc
            logger.info("Launched %s (pid=%d)", w["name"], proc.pid)

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", self.port))
        self.server_sock.listen(len(expanded) + 4)

        threads: list[threading.Thread] = []
        try:
            while not self.shutdown.is_set() and self.completed + len(self.failed) < self.total:
                try:
                    self.server_sock.settimeout(2.0)
                    conn, addr = self.server_sock.accept()
                    t = threading.Thread(target=self._handle_worker, args=(conn, addr), daemon=True)
                    t.start()
                    threads.append(t)
                except TimeoutError:
                    continue
                except OSError:
                    if self.shutdown.is_set():
                        break
        finally:
            self.shutdown.set()
            for t in threads:
                t.join(timeout=5)
            if self.server_sock:
                self.server_sock.close()
            logger.info(
                "Sweep complete: %d/%d done, %d failed",
                self.completed,
                self.total,
                len(self.failed),
            )
            self.state = "idle"

    def _handle_worker(self, conn: socket.socket, addr) -> None:
        worker_name = f"{addr[0]}:{addr[1]}"
        current_job: str | None = None
        try:
            while not self.shutdown.is_set():
                msg: Message | None = recv_frame(conn)
                if msg is None:
                    break

                if isinstance(msg, WorkerInfoMessage):
                    ws = self.workers_state.setdefault(
                        msg.worker_name, WorkerState(msg.worker_name, msg.host)
                    )
                    ws.conn = conn
                    ws.connected = True
                    ws.devices = [d.__dict__ for d in msg.devices]
                    worker_name = msg.worker_name
                    logger.info("Worker %s connected: %d devices", worker_name, len(msg.devices))

                elif isinstance(msg, HeartbeatMessage):
                    ws = self.workers_state.get(msg.worker_name)
                    if ws:
                        ws.heartbeats.append(
                            {
                                "t": time.time(),
                                "devices": [
                                    {
                                        "idx": d.index,
                                        "alloc_mb": d.allocated_mb,
                                        "reserved_mb": d.reserved_mb,
                                        "used_mb": d.used_mb,
                                        "util_pct": d.utilization_pct,
                                    }
                                    for d in msg.devices
                                ],
                            }
                        )

                elif isinstance(msg, ReadyMessage):
                    ws = self.workers_state.get(worker_name)
                    if ws and not ws.enabled:
                        send_frame(conn, DisableMessage())
                        return
                    while not self.shutdown.is_set():
                        if self.pause_event.is_set():
                            time.sleep(0.5)
                            continue
                        try:
                            job_id = self.job_queue.get(timeout=1)
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
                    ws = self.workers_state.get(worker_name)
                    if ws:
                        ws.current_job = job_id
                    send_frame(conn, VQJobMessage(config=cfg, layer=self.layer))

                elif isinstance(msg, VQResultsMessage):
                    with self.db_lock:
                        insert_row(self.db, msg.row, msg.config_id, worker_name)
                        for er in msg.extra_rows or []:
                            eid = f"int_baseline_{er.get('scheme', 'unknown')}"
                            insert_row(self.db, er, eid, worker_name)
                    self.completed += 1
                    ws = self.workers_state.get(worker_name)
                    if ws:
                        ws.current_job = None
                    send_frame(conn, AckMessage())
                    current_job = None
                    if self.completed + len(self.failed) >= self.total:
                        self.shutdown.set()
                        return

                elif isinstance(msg, VQErrorMessage):
                    logger.error("Worker %s error on %s: %s", worker_name, msg.config_id, msg.msg)
                    if current_job == msg.config_id:
                        self.failed.append(msg.config_id)
                        current_job = None
                        ws = self.workers_state.get(worker_name)
                        if ws:
                            ws.current_job = None

        except (ConnectionError, OSError) as e:
            logger.warning("Worker %s connection failed: %s", worker_name, e)
            if current_job:
                self.job_queue.put(current_job)
        finally:
            ws = self.workers_state.get(worker_name)
            if ws:
                ws.connected = False
            conn.close()
