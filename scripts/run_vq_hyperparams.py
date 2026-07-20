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

# ruff: noqa: E501  (inline JS has long lines)
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

# Global orchestrator instance — set in main(), read by Flask routes.
ORCH: VQOrchestrator | None = None

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
    conn.row_factory = sqlite3.Row  # so dict(row) works in fetch_results
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
        self.pause_event = threading.Event()  # set = paused
        self.db_lock = threading.Lock()
        self.state = "idle"  # idle | running | paused | stopped
        self.worker_procs: list[subprocess.Popen[bytes]] = []
        self.worker_threads: list[threading.Thread] = []
        self.server_sock: socket.socket | None = None
        self.configs_filter: str | None = None  # live filter substring
        self.configs_limit: int | None = None  # live cap on configs to run

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
            "state": self.state,
            "total": self.total,
            "completed": self.completed,
            "failed": len(self.failed),
            "remaining": self.remaining,
            "in_queue": self.queue.qsize(),
            "paused": self.pause_event.is_set(),
            "workers_done": self.workers_done,
            "failed_ids": self.failed[:50],
            "filter": self.configs_filter,
            "limit": self.configs_limit,
            "updated_at": time.time(),
        }
        STATUS_PATH.write_text(json.dumps(status, indent=2))

    # ------------------------------------------------------------------
    # Control plane — called from Flask handlers (or directly from main())
    # ------------------------------------------------------------------

    def launch(self) -> str:
        """Start the orchestrator in a background thread. Returns state."""
        if self.state in ("running", "paused"):
            return self.state
        self.state = "running"
        self.shutdown.clear()
        self.pause_event.clear()
        t = threading.Thread(target=self.run, daemon=True, name="orchestrator")
        t.start()
        self.worker_threads.append(t)
        self._write_status()
        return self.state

    def pause(self) -> str:
        """Stop dispatching new jobs (in-flight jobs finish)."""
        if self.state != "running":
            return self.state
        self.pause_event.set()
        self.state = "paused"
        self._write_status()
        return self.state

    def resume(self) -> str:
        """Unpause — workers start pulling jobs again."""
        if self.state != "paused":
            return self.state
        self.pause_event.clear()
        self.state = "running"
        self._write_status()
        return self.state

    def requeue_failed(self) -> int:
        """Move all failed config_ids back to the queue."""
        if not self.failed:
            return 0
        n = len(self.failed)
        for cfg_id in self.failed:
            self.queue.put(cfg_id)
        self.failed.clear()
        self._write_status()
        logger.info("Re-queued %d failed configs", n)
        return n

    def shutdown_now(self) -> str:
        """Graceful shutdown: stop dispatching, let in-flight finish, kill workers."""
        self.pause_event.clear()
        self.shutdown.set()
        self.state = "stopped"
        for proc in self.worker_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        self._write_status()
        return self.state

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
                    # Wait for next available config (respecting pause + shutdown)
                    while not self.shutdown.is_set():
                        # Pause: spin until unpaused or shutdown
                        if self.pause_event.is_set():
                            time.sleep(0.5)
                            continue
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
        """Orchestrator main loop. Runs in a background thread when launched."""
        workers = load_workers(self.workers_yaml)
        logger.info("Launching %d workers from %s", len(workers), self.workers_yaml)

        for w in workers:
            proc = launch_worker(w, self.port, self.orchestrator_host, self.model_id, self.layer)
            self.worker_procs.append(proc)
            logger.info(
                "Launched %s (pid=%d, host=%s, device=%s)",
                w["name"],
                proc.pid,
                w["host"],
                w.get("device", "auto"),
            )

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", self.port))
        self.server_sock.listen(len(workers) + 4)
        logger.info("Orchestrator listening on 0.0.0.0:%d", self.port)

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
                except OSError as e:
                    if self.shutdown.is_set():
                        break
                    logger.error("Accept failed: %s", e)
        finally:
            self.shutdown.set()
            for t in threads:
                t.join(timeout=5)
            if self.server_sock is not None:
                self.server_sock.close()
            self._write_status()

        logger.info(
            "Sweep complete: %d/%d done, %d failed",
            self.completed,
            self.total,
            len(self.failed),
        )
        self.state = "idle" if self.completed + len(self.failed) >= self.total else "stopped"
        self._write_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQ hyperparameter sweep server (orchestrator + dashboard)"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=CONFIGS_PATH,
        help="Path to configs.json from gen_vq_hyperparams.py",
    )
    parser.add_argument("--workers", type=Path, default=REPO / "workers.yaml")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--port", type=int, default=5555, help="TCP port for worker connections")
    parser.add_argument(
        "--http-port",
        type=int,
        default=8050,
        help="HTTP port for dashboard",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run configs whose id matches this substring (for testing)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max configs to run (for testing)")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--auto-launch",
        action="store_true",
        help="Automatically launch the sweep on startup (default: wait for UI)",
    )
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
    global ORCH
    ORCH = VQOrchestrator(
        configs=configs,
        workers_yaml=args.workers,
        model_id=args.model,
        layer=args.layer,
        port=args.port,
        db_path=args.db,
    )
    ORCH.configs_filter = args.filter
    ORCH.configs_limit = args.limit

    if args.auto_launch:
        ORCH.launch()

    # Import here so the global ORCH is set before Flask routes are registered.
    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    @app.route("/")
    def index() -> Response:
        return Response(_DASHBOARD_HTML, content_type="text/html")

    @app.route("/api/results")
    def api_results():
        assert ORCH is not None
        return jsonify(
            {"results": fetch_results(ORCH.db, ORCH.db_lock), "status": _orch_status(ORCH)}
        )

    @app.route("/api/status")
    def api_status():
        assert ORCH is not None
        return jsonify(_orch_status(ORCH))

    @app.route("/api/control/launch", methods=["POST"])
    def ctrl_launch():
        assert ORCH is not None
        return jsonify({"state": ORCH.launch()})

    @app.route("/api/control/pause", methods=["POST"])
    def ctrl_pause():
        assert ORCH is not None
        return jsonify({"state": ORCH.pause()})

    @app.route("/api/control/resume", methods=["POST"])
    def ctrl_resume():
        assert ORCH is not None
        return jsonify({"state": ORCH.resume()})

    @app.route("/api/control/requeue-failed", methods=["POST"])
    def ctrl_requeue():
        assert ORCH is not None
        n = ORCH.requeue_failed()
        return jsonify({"requeued": n})

    @app.route("/api/control/shutdown", methods=["POST"])
    def ctrl_shutdown():
        assert ORCH is not None
        return jsonify({"state": ORCH.shutdown_now()})

    print(f"Dashboard: http://localhost:{args.http_port}/  (worker TCP port: {args.port})")
    print("Controls: launch / pause / resume / requeue-failed / shutdown (via UI buttons)")
    app.run(host="0.0.0.0", port=args.http_port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Helpers used by Flask routes
# ---------------------------------------------------------------------------


def fetch_results(db: sqlite3.Connection, db_lock: threading.Lock) -> list[dict]:
    """Read all rows from SQLite (thread-safe)."""
    rows = []
    with db_lock:
        cur = db.execute("""
            SELECT projection, scheme, block_size, K, n_codebooks, metric,
                   shared, sign_split, scale_dtype, kmeans_iters,
                   residual_block_sizes, rel_fro_err, bits_per_weight,
                   compression_ratio, worker, completed_at
            FROM results
            ORDER BY bits_per_weight
        """)
        for r in cur.fetchall():
            d = dict(r)
            d["shared"] = bool(d["shared"])
            d["sign_split"] = bool(d["sign_split"])
            try:
                d["residual_block_sizes"] = json.loads(d.get("residual_block_sizes") or "[]")
            except TypeError, json.JSONDecodeError:
                d["residual_block_sizes"] = []
            rows.append(d)
    return rows


def _orch_status(orch: VQOrchestrator) -> dict:
    return {
        "state": orch.state,
        "total": orch.total,
        "completed": orch.completed,
        "failed": len(orch.failed),
        "remaining": orch.remaining,
        "in_queue": orch.queue.qsize(),
        "paused": orch.pause_event.is_set(),
        "filter": orch.configs_filter,
        "limit": orch.configs_limit,
    }


_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>SKCQ VQ Sweep</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #fafafa; }
    h1 { margin-bottom: 0.2em; }
    #status { font-size: 14px; color: #666; margin: 10px 0 20px; padding: 10px; background: white; border: 1px solid #ddd; }
    .controls { margin-bottom: 20px; }
    .controls button { padding: 8px 16px; margin-right: 8px; font-size: 14px; cursor: pointer; }
    .controls button:disabled { opacity: 0.4; cursor: not-allowed; }
    .chart { background: white; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
    th { background: #f4f4f4; }
    .winner-kmeans { color: #2e7d32; font-weight: bold; }
    .winner-int { color: #c62828; font-weight: bold; }
  </style>
</head>
<body>
  <h1>SKCQ VQ Hyperparameter Sweep</h1>

  <div class="controls">
    <button id="btn-launch" onclick="ctrl('launch')">Launch</button>
    <button id="btn-pause" onclick="ctrl('pause')" disabled>Pause</button>
    <button id="btn-resume" onclick="ctrl('resume')" disabled>Resume</button>
    <button id="btn-requeue" onclick="ctrl('requeue-failed')">Requeue failed</button>
    <button id="btn-shutdown" onclick="ctrl('shutdown')">Shutdown</button>
  </div>

  <div id="status">Loading...</div>

  <div id="pareto" class="chart"></div>
  <div id="heatmap" class="chart"></div>
  <div id="bucket-table" class="chart"></div>

<script>
function isKmeans(r) { return r.scheme && r.scheme.startsWith('kmeans_'); }

function paretoFrontier(rows) {
  const out = [];
  for (const r of rows) {
    const dominated = rows.some(o =>
      o !== r && o.bits_per_weight <= r.bits_per_weight &&
      o.rel_fro_err <= r.rel_fro_err &&
      (o.bits_per_weight < r.bits_per_weight || o.rel_fro_err < r.rel_fro_err)
    );
    if (!dominated) out.push(r);
  }
  out.sort((a, b) => a.bits_per_weight - b.bits_per_weight);
  return out;
}

function renderPareto(results) {
  const projections = [...new Set(results.map(r => r.projection))];
  const traces = [];
  for (const proj of projections) {
    const projRows = results.filter(r => r.projection === proj);
    const km = projRows.filter(isKmeans);
    const ints = projRows.filter(r => !isKmeans(r));
    if (km.length > 0) {
      traces.push({
        x: km.map(r => r.bits_per_weight),
        y: km.map(r => r.rel_fro_err),
        mode: 'markers',
        name: `${proj} kmeans`,
        text: km.map(r => `${r.scheme}<br>bpw=${r.bits_per_weight.toFixed(3)}<br>err=${r.rel_fro_err.toExponential(3)}`),
        marker: {
          size: km.map(r => Math.max(5, Math.min(15, r.K / 4096 + 4))),
          color: km.map(r => r.block_size),
          colorscale: 'Viridis',
          showscale: true,
          colorbar: {title: 'block_size'},
          opacity: 0.7,
        },
        hovertemplate: '%{text}<extra></extra>',
      });
    }
    if (ints.length > 0) {
      traces.push({
        x: ints.map(r => r.bits_per_weight),
        y: ints.map(r => r.rel_fro_err),
        mode: 'markers',
        name: `${proj} integer`,
        text: ints.map(r => r.scheme),
        marker: {symbol: 'x', size: 8, color: 'red', opacity: 0.6},
        hovertemplate: '%{text}<br>bpw=%{x:.3f}<br>err=%{y:.6f}<extra></extra>',
      });
    }
    const pareto = paretoFrontier(projRows);
    traces.push({
      x: pareto.map(r => r.bits_per_weight),
      y: pareto.map(r => r.rel_fro_err),
      mode: 'lines',
      name: `${proj} Pareto`,
      line: {width: 2, dash: 'dash'},
      hoverinfo: 'skip',
    });
  }
  Plotly.newPlot('pareto', traces, {
    title: 'Pareto Frontier: bits-per-weight vs reconstruction error',
    xaxis: {title: 'bits per weight'},
    yaxis: {title: 'relative Frobenius error'},
    hovermode: 'closest',
    height: 600,
  }, {responsive: true});
}

function renderHeatmap(results) {
  const km = results.filter(r => isKmeans(r) && r.n_codebooks === 1);
  if (km.length === 0) {
    document.getElementById('heatmap').innerHTML = '<p>No single-codebook results yet</p>';
    return;
  }
  const bsVals = [...new Set(km.map(r => r.block_size))].sort((a,b) => a-b);
  const kVals = [...new Set(km.map(r => r.K))].sort((a,b) => a-b);
  const errMatrix = bsVals.map(bs => kVals.map(K => {
    const r = km.find(r => r.block_size === bs && r.K === K);
    return r ? r.rel_fro_err : null;
  }));
  const bpwMatrix = bsVals.map(bs => kVals.map(K => {
    const r = km.find(r => r.block_size === bs && r.K === K);
    return r ? r.bits_per_weight.toFixed(2) : '';
  }));
  Plotly.newPlot('heatmap', [{
    z: errMatrix,
    x: kVals.map(String),
    y: bsVals.map(String),
    type: 'heatmap',
    colorscale: 'Viridis_r',
    colorbar: {title: 'error'},
    text: bpwMatrix,
    texttemplate: '%{text}',
    hovertemplate: 'bs=%{y} K=%{x}<br>err=%{z:.6f}<br>bpw=%{text}<extra></extra>',
  }], {
    title: 'Block size × K → error (single codebook, bpw annotated)',
    xaxis: {title: 'K'},
    yaxis: {title: 'block_size'},
    height: 500,
  }, {responsive: true});
}

function renderBucketTable(results) {
  const buckets = [[1.0,1.5],[1.5,2.0],[2.0,2.5],[2.5,3.0],[3.0,3.5],[3.5,4.0],[4.0,5.0],[5.0,6.0]];
  const projections = [...new Set(results.map(r => r.projection))].sort();
  let html = '<h3>Best error per bpw bucket</h3><table><thead><tr><th>bucket</th>';
  for (const p of projections) html += `<th>${p} kmeans</th><th>${p} integer</th><th>winner</th>`;
  html += '</tr></thead><tbody>';
  for (const [lo, hi] of buckets) {
    html += `<tr><td>[${lo.toFixed(1)}-${hi.toFixed(1)})</td>`;
    for (const p of projections) {
      const projRows = results.filter(r => r.projection === p && lo <= r.bits_per_weight && r.bits_per_weight < hi);
      const km = projRows.filter(isKmeans);
      const ints = projRows.filter(r => !isKmeans(r));
      const bestK = km.length > 0 ? Math.min(...km.map(r => r.rel_fro_err)) : null;
      const bestI = ints.length > 0 ? Math.min(...ints.map(r => r.rel_fro_err)) : null;
      html += `<td>${bestK !== null ? bestK.toFixed(6) : '-'}</td>`;
      html += `<td>${bestI !== null ? bestI.toFixed(6) : '-'}</td>`;
      let winner = '-';
      if (bestK !== null && bestI !== null) {
        winner = bestK < bestI ? '<span class="winner-kmeans">kmeans</span>' : '<span class="winner-int">int</span>';
      }
      html += `<td>${winner}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('bucket-table').innerHTML = html;
}

function updateButtons(state, paused) {
  const running = state === 'running' && !paused;
  const idle = state === 'idle' || state === 'stopped';
  document.getElementById('btn-launch').disabled = !idle;
  document.getElementById('btn-pause').disabled = !running;
  document.getElementById('btn-resume').disabled = state !== 'paused';
  document.getElementById('btn-shutdown').disabled = idle;
}

async function ctrl(action) {
  try {
    const resp = await fetch(`/api/control/${action}`, {method: 'POST'});
    const data = await resp.json();
    console.log(action, '->', data);
    refresh();
  } catch (e) {
    console.error(e);
  }
}

async function refresh() {
  try {
    const resp = await fetch('/api/results');
    const data = await resp.json();
    const results = data.results || [];
    const status = data.status || {};
    const progress = status.total > 0 ? (status.completed / status.total * 100).toFixed(1) : 0;
    document.getElementById('status').innerHTML =
      `<strong>State:</strong> ${status.state || 'idle'} ` +
      (status.paused ? '(paused)' : '') +
      ` &nbsp;|&nbsp; <strong>Progress:</strong> ${status.completed}/${status.total} (${progress}%)` +
      ` &nbsp;|&nbsp; <strong>Failed:</strong> ${status.failed}` +
      ` &nbsp;|&nbsp; <strong>In queue:</strong> ${status.in_queue || 0}`;
    updateButtons(status.state, status.paused);
    if (results.length > 0) {
      renderPareto(results);
      renderHeatmap(results);
      renderBucketTable(results);
    } else {
      document.getElementById('pareto').innerHTML = '<p>No results yet — click Launch to start.</p>';
    }
  } catch (e) {
    document.getElementById('status').innerHTML = `<span style="color:red">Error: ${e}</span>`;
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
