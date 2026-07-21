"""Flask server: control plane API + dashboard HTML for the VQ sweep.

Wraps VQOrchestrator and exposes:
    GET  /                         Dashboard HTML
    GET  /api/results              All SQLite rows + live status
    GET  /api/status              Orchestrator state
    GET  /api/range               Current VQHyperparamRange
    POST /api/range               Replace range (applies to next sweep)
    GET  /api/workers             Per-worker status + heartbeat ring buffer
    POST /api/workers/<name>/enable
    POST /api/workers/<name>/disable
    POST /api/control/launch|pause|resume|requeue-failed|shutdown

The dashboard HTML is served from skcq/vq/dashboard.html (static file, no
Jinja templating — the frontend fetches data via /api/* and renders with
Plotly + vanilla JS).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from skcq.vq.hyperparams import VQHyperparamRange, default_range
from skcq.vq.orchestrator import VQOrchestrator

logger = logging.getLogger("skcq.vq.server")

DASHBOARD_DIR = (Path(__file__).resolve().parent / "dashboard").absolute()


def fetch_results(db: sqlite3.Connection, db_lock: threading.Lock) -> list[dict]:
    rows = []
    with db_lock:
        cur = db.execute("""
            SELECT projection, scheme, block_size, K, n_codebooks, metric,
                   shared, sign_split, scale_dtype, kmeans_iters,
                   residual_block_sizes, rel_fro_err, bits_per_weight,
                   compression_ratio, worker, completed_at
            FROM results ORDER BY bits_per_weight
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


def _status(orch: VQOrchestrator) -> dict:
    return {
        "state": orch.state,
        "total": orch.total,
        "completed": orch.completed,
        "failed": len(orch.failed),
        "in_queue": orch.job_queue.qsize(),
        "paused": orch.pause_event.is_set(),
    }


def _workers(orch: VQOrchestrator) -> list[dict]:
    out = []
    for name, ws in orch.workers_state.items():
        last_hb = ws.heartbeats[-1] if ws.heartbeats else None
        out.append(
            {
                "name": name,
                "host": ws.host,
                "enabled": ws.enabled,
                "connected": ws.connected,
                "devices": ws.devices,
                "current_job": ws.current_job,
                "last_heartbeat": last_hb,
                "history": list(ws.heartbeats),
            }
        )
    return out


def create_app(orch: VQOrchestrator) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> Response:
        return send_file(DASHBOARD_DIR / "index.html", mimetype="text/html")

    @app.route("/styles.css")
    def styles():
        return send_file(DASHBOARD_DIR / "styles.css", mimetype="text/css")

    @app.route("/api.js")
    def api_js():
        return send_file(DASHBOARD_DIR / "api.js", mimetype="application/javascript")

    @app.route("/ui.js")
    def ui_js():
        return send_file(DASHBOARD_DIR / "ui.js", mimetype="application/javascript")

    @app.route("/api/results")
    def api_results():
        return jsonify({"results": fetch_results(orch.db, orch.db_lock), "status": _status(orch)})

    @app.route("/api/status")
    def api_status():
        return jsonify(_status(orch))

    @app.route("/api/range")
    def api_range():
        return jsonify(orch.range.to_dict())

    @app.route("/api/range", methods=["POST"])
    def api_set_range():
        orch.range = VQHyperparamRange.from_dict(json.loads(request.data))
        orch._rebuild_queue()
        return jsonify(
            {
                "applied": "now",
                "est_configs": len(orch.range),
                "total": orch.total,
                "in_queue": orch.job_queue.qsize(),
            }
        )

    @app.route("/api/workers")
    def api_workers():
        return jsonify(_workers(orch))

    @app.route("/api/workers/<name>/enable", methods=["POST"])
    def api_enable(name: str):
        return jsonify({"result": orch.enable_worker(name)})

    @app.route("/api/workers/<name>/disable", methods=["POST"])
    def api_disable(name: str):
        return jsonify({"result": orch.disable_worker(name)})

    @app.route("/api/control/launch", methods=["POST"])
    def ctrl_launch():
        return jsonify({"state": orch.launch()})

    @app.route("/api/control/pause", methods=["POST"])
    def ctrl_pause():
        return jsonify({"state": orch.pause()})

    @app.route("/api/control/resume", methods=["POST"])
    def ctrl_resume():
        return jsonify({"state": orch.resume()})

    @app.route("/api/control/requeue-failed", methods=["POST"])
    def ctrl_requeue():
        return jsonify({"requeued": orch.requeue_failed()})

    @app.route("/api/control/shutdown", methods=["POST"])
    def ctrl_shutdown():
        return jsonify({"state": orch.shutdown_now()})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VQ sweep server (orchestrator + dashboard)")
    parser.add_argument("--workers", type=Path, default=Path("workers.yaml"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--port", type=int, default=5555, help="TCP port for workers")
    parser.add_argument("--http-port", type=int, default=8050, help="HTTP port for dashboard")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )

    orch = VQOrchestrator(
        range_=default_range(),
        workers_yaml=args.workers,
        model_id=args.model,
        layer=args.layer,
        port=args.port,
    )
    orch.start()

    app = create_app(orch)
    print(f"Dashboard: http://localhost:{args.http_port}/  (worker TCP: {args.port})")
    app.run(host="0.0.0.0", port=args.http_port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
