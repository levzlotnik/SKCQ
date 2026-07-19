"""Distributed sweep runner for weight_quant_error.py experiments.

Reads `sweep_results/configs.json` (from gen_sweep.py), distributes configs
across workers (local subprocess + remote SSH), runs each as a CLI invocation
of `experiments/weight_quant_error.py --output sweep_results/<id>.csv`, and
writes a status file as it goes. Idempotent: skips configs whose output CSV
already exists with results.

Usage:
    uv run python scripts/run_sweep_distributed.py
    uv run python scripts/run_sweep_distributed.py --workers workers.yaml
    uv run python scripts/run_sweep_distributed.py --max-concurrent 4 --filter gate
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread

import yaml

REPO = Path(__file__).resolve().parent.parent
EXP_SCRIPT = REPO / "experiments" / "weight_quant_error.py"
RESULTS_DIR = REPO / "sweep_results"
STATUS_FILE = RESULTS_DIR / "status.json"
FAILED_LOG = RESULTS_DIR / "failed.jsonl"


@dataclass
class Worker:
    name: str
    host: str  # "localhost" for local subprocess
    venv_python: str  # path to python binary
    workdir: str
    device: str  # cuda / cpu / auto
    chunk_budget_mb: int | None = None
    parallelism: int = 1  # number of concurrent jobs on this worker
    remote: bool = False  # True if SSH needed
    devices: list[int] | None = None  # GPU indices to assign per parallel slot


def load_workers(yaml_path: Path) -> list[Worker]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    workers = []
    for w in cfg.get("workers", []):
        remote = w["host"] != "localhost"
        workers.append(
            Worker(
                name=w["name"],
                host=w["host"],
                venv_python=w["venv"],
                workdir=w["workdir"],
                device=w.get("device", "auto"),
                chunk_budget_mb=w.get("chunk_budget_mb"),
                parallelism=w.get("parallelism", 1),
                remote=remote,
                devices=w.get("devices"),
            )
        )
    return workers


def config_csv_path(cfg: dict) -> Path:
    """Per-config output CSV path. Used for idempotency."""
    return RESULTS_DIR / f"{cfg['id']}.csv"


def config_is_done(cfg: dict) -> bool:
    """A config is done if its CSV exists and has at least one kmeans row."""
    p = config_csv_path(cfg)
    if not p.exists():
        return False
    # Quick sanity: file should have a header + at least one kmeans row
    try:
        text = p.read_text()
        return "kmeans_" in text
    except Exception:
        return False


def build_cmd(cfg: dict, worker: Worker) -> list[str]:
    """Build the command (list of args) for one config on one worker."""
    out_csv = config_csv_path(cfg)
    cmd = [
        worker.venv_python,
        str(EXP_SCRIPT),
        *cfg["args"],
        "--output",
        str(out_csv),
        "--overwrite",
    ]
    if worker.chunk_budget_mb:
        cmd += ["--chunk-budget-mb", str(worker.chunk_budget_mb)]
    # For local workers, use absolute path; for remote, use workdir-relative
    if worker.remote:
        # Strip the repo prefix — remote will run from workdir
        cmd[1] = "experiments/weight_quant_error.py"
        # Output path should also be relative for remote (then we scp back)
        remote_out = f"sweep_results/{cfg['id']}.csv"
        # Replace the absolute --output path with relative
        for i, a in enumerate(cmd):
            if a == "--output":
                cmd[i + 1] = remote_out
        # Pass device as env var via CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES
    return cmd


def run_local(
    cfg: dict, worker: Worker, status: dict, status_lock: Lock, slot_idx: int = 0
) -> bool:
    """Run a config as a local subprocess. Returns True on success."""
    cmd = build_cmd(cfg, worker)
    env = os.environ.copy()
    # Per-slot GPU pinning: when `devices` is set, each parallel slot gets one
    # GPU. We set both CUDA_VISIBLE_DEVICES (NVIDIA torch) and HIP_VISIBLE_DEVICES
    # (ROCm torch exposes AMD GPUs as cuda:0 after HIP filtering).
    pinned_dev = (
        worker.devices[slot_idx]
        if worker.device == "cuda" and worker.devices and slot_idx < len(worker.devices)
        else None
    )
    if worker.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif pinned_dev is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(pinned_dev)
        env["HIP_VISIBLE_DEVICES"] = str(pinned_dev)

    log_path = REPO / "sweep_logs" / f"{cfg['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO,
        )
        # Wait with timeout (max 20 min per config)
        try:
            proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with status_lock:
                status["failed"].append({"id": cfg["id"], "reason": "timeout"})
                with open(FAILED_LOG, "a") as f:
                    f.write(
                        json.dumps({"id": cfg["id"], "reason": "timeout", "worker": worker.name})
                        + "\n"
                    )
            return False

    if proc.returncode != 0:
        with status_lock:
            status["failed"].append(
                {"id": cfg["id"], "reason": f"exit={proc.returncode}", "worker": worker.name}
            )
            with open(FAILED_LOG, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": cfg["id"],
                            "reason": f"exit={proc.returncode}",
                            "worker": worker.name,
                        }
                    )
                    + "\n"
                )
        return False

    return config_is_done(cfg)


def run_remote(
    cfg: dict, worker: Worker, status: dict, status_lock: Lock, slot_idx: int = 0
) -> bool:
    """Run a config on a remote worker via SSH. Returns True on success."""
    local_cmd = build_cmd(cfg, worker)  # already uses relative paths for remote
    # Per-slot GPU pinning: export CUDA/HIP_VISIBLE_DEVICES before running.
    env_prefix = ""
    if worker.devices is not None and slot_idx < len(worker.devices):
        dev = str(worker.devices[slot_idx])
        env_prefix = f"export CUDA_VISIBLE_DEVICES={dev} HIP_VISIBLE_DEVICES={dev}; "
    # On the remote side: cd to workdir, git pull, run cmd, leave output
    remote_script = (
        f"cd {shlex.quote(worker.workdir)} && git pull --quiet && {env_prefix}"
        f"{' '.join(shlex.quote(x) for x in local_cmd)}"
    )
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", worker.host, remote_script]

    log_path = REPO / "sweep_logs" / f"{cfg['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )
        try:
            proc.wait(timeout=2400)  # 40 min for remote (slower)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with status_lock:
                status["failed"].append(
                    {"id": cfg["id"], "reason": "timeout", "worker": worker.name}
                )
                with open(FAILED_LOG, "a") as f:
                    f.write(
                        json.dumps({"id": cfg["id"], "reason": "timeout", "worker": worker.name})
                        + "\n"
                    )
            return False

    if proc.returncode != 0:
        with status_lock:
            status["failed"].append(
                {"id": cfg["id"], "reason": f"exit={proc.returncode}", "worker": worker.name}
            )
            with open(FAILED_LOG, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "id": cfg["id"],
                            "reason": f"exit={proc.returncode}",
                            "worker": worker.name,
                        }
                    )
                    + "\n"
                )
        return False

    # scp the output CSV back
    remote_csv = f"{worker.workdir}/sweep_results/{cfg['id']}.csv"
    local_csv = config_csv_path(cfg)
    scp_cmd = ["scp", "-q", f"{worker.host}:{remote_csv}", str(local_csv)]
    try:
        subprocess.run(scp_cmd, check=True, timeout=60)
    except Exception as e:
        with status_lock:
            status["failed"].append(
                {"id": cfg["id"], "reason": f"scp_failed: {e}", "worker": worker.name}
            )
        return False

    return config_is_done(cfg)


def worker_loop(
    worker: Worker,
    queue: list[dict],
    status: dict,
    status_lock: Lock,
    idx_lock: Lock,
    slot_idx: int = 0,
) -> None:
    """Worker loop: pull next config from queue (shared index), run, repeat."""
    queue_idx = [0]
    while True:
        with idx_lock:
            i = queue_idx[0]
            queue_idx[0] += 1
        if i >= len(queue):
            return
        cfg = queue[i]

        if config_is_done(cfg):
            with status_lock:
                status["skipped"] += 1
                status["last_completed"] = cfg["id"]
                _persist_status(status)
            continue

        t0 = time.time()
        with status_lock:
            status["running"].append(
                {"id": cfg["id"], "worker": worker.name, "started": time.time()}
            )

        if worker.remote:
            ok = run_remote(cfg, worker, status, status_lock, slot_idx)
        else:
            ok = run_local(cfg, worker, status, status_lock, slot_idx)

        dt = time.time() - t0
        with status_lock:
            status["running"] = [r for r in status["running"] if r["id"] != cfg["id"]]
            if ok:
                status["done"].append(
                    {"id": cfg["id"], "worker": worker.name, "elapsed_s": round(dt, 1)}
                )
            else:
                status["failed_count"] = status.get("failed_count", 0) + 1
            status["last_completed"] = cfg["id"]
            _persist_status(status)


idx_lock = Lock()


def _persist_status(status: dict) -> None:
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed sweep runner")
    parser.add_argument("--configs", type=Path, default=RESULTS_DIR / "configs.json")
    parser.add_argument("--workers", type=Path, default=REPO / "workers.yaml")
    parser.add_argument(
        "--filter", type=str, default=None, help="Only run configs whose id matches this substring"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max configs to run (for testing)")
    args = parser.parse_args()

    status_lock = Lock()

    with open(args.configs) as f:
        all_configs = json.load(f)

    if args.filter:
        all_configs = [c for c in all_configs if args.filter in c["id"]]
    if args.limit:
        all_configs = all_configs[: args.limit]

    # Idempotency: skip already-done configs up front
    todo = [c for c in all_configs if not config_is_done(c)]
    done = len(all_configs) - len(todo)
    print(f"Configs: {len(all_configs)} total, {done} already done, {len(todo)} to run")

    if not todo:
        print("Nothing to do.")
        return

    workers = load_workers(args.workers)
    # Build a list of (worker) repeated by parallelism, then distribute
    worker_slots = []
    for w in workers:
        worker_slots.extend([w] * w.parallelism)
    print(f"Workers: {len(worker_slots)} slots from {len(workers)} workers")
    for w in workers:
        print(f"  {w.name}: parallelism={w.parallelism}, remote={w.remote}, host={w.host}")

    # Build per-worker queues (longest-first already sorted; assign round-robin)
    # Per-worker queue so we don't need cross-worker locking on the index.
    per_worker: dict[str, list[dict]] = {w.name: [] for w in workers}
    for i, cfg in enumerate(todo):
        # Round-robin across worker slots (already expanded by parallelism)
        slot = worker_slots[i % len(worker_slots)]
        per_worker[slot.name].append(cfg)

    for w in workers:
        print(f"  {w.name}: {len(per_worker[w.name])} configs")

    # Status tracking
    status = {
        "total": len(todo),
        "done": [],
        "failed": [],
        "failed_count": 0,
        "skipped": 0,
        "running": [],
        "last_completed": None,
        "started_at": time.time(),
    }
    _persist_status(status)

    # Spawn one thread per worker (each worker processes its queue serially in that thread,
    # but if parallelism > 1, we spawn multiple threads for that worker).
    threads = []
    for w in workers:
        for slot_idx in range(w.parallelism):
            queue = per_worker[w.name][slot_idx :: w.parallelism]
            t = Thread(
                target=worker_loop,
                args=(w, queue, status, status_lock, idx_lock, slot_idx),
                name=f"{w.name}-{slot_idx}",
            )
            t.start()
            threads.append(t)

    # Wait for all threads
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with status_lock:
                elapsed = time.time() - status["started_at"]
                print(
                    f"  done={len(status['done'])}/{status['total']} "
                    f"failed={len(status['failed'])} "
                    f"skipped={status['skipped']} "
                    f"running={len(status['running'])} "
                    f"elapsed={elapsed:.0f}s "
                    f"last={status['last_completed']}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted — status saved to", STATUS_FILE)

    for t in threads:
        t.join(timeout=1)

    elapsed = time.time() - status["started_at"]
    print(
        f"\nSweep complete: {len(status['done'])}/{status['total']} done, "
        f"{len(status['failed'])} failed, {elapsed:.0f}s total"
    )
    if status["failed"]:
        print(f"Failed configs logged to {FAILED_LOG}")


if __name__ == "__main__":
    main()
