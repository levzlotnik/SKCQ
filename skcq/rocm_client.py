from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
from typing import Any

import torch

from skcq.codebook.aqlm import AQLMCodebook
from skcq.codebook.cwr import AQLMCWR
from skcq.config import CodebookParams

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CUDA_PYTHON = os.path.join(_PROJECT_ROOT, "cuda", ".venv", "bin", "python")

_IN_PATH = "/dev/shm/skcq_worker_in.pt"
_OUT_PATH = "/dev/shm/skcq_worker_out.pt"


class RocmClient:
    """Spawns a CUDA subprocess and offloads codebook builds to the 3090."""

    def __init__(self, cuda_python: str | None = None) -> None:
        self.cuda_python = cuda_python or _DEFAULT_CUDA_PYTHON
        self._proc: subprocess.Popen[bytes] | None = None
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _ensure_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = {**os.environ, "PYTHONPATH": _PROJECT_ROOT}
        self._proc = subprocess.Popen(
            [self.cuda_python, "-m", "skcq.cuda_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        logger.info("Started CUDA server (pid=%d)", self._proc.pid)

    def _on_sigterm(self, signum: int, frame: Any) -> None:
        self.close()
        raise SystemExit(0)

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        self._ensure_server()
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        request = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        self._proc.stdin.write((json.dumps(request) + "\n").encode())
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("CUDA server closed stdout")
            msg = json.loads(line)
            if "method" in msg and msg.get("method") == "progress":
                logger.info("[cuda] %s", msg["params"]["message"])
                continue
            if "error" in msg:
                raise RuntimeError(f"RPC error from CUDA worker: {msg['error']}")
            return msg.get("result")

    def build_codebook(
        self,
        rows: torch.Tensor,
        params: CodebookParams,
        k: int,
        n_blocks: int,
        n_codebooks: int,
        num_experts: int,
        out_dim: int,
        name: str = "",
    ) -> tuple[AQLMCodebook, AQLMCWR]:
        rows_cpu = rows.cpu().contiguous()
        torch.save(rows_cpu, _IN_PATH)

        self._rpc(
            "build_codebook",
            {
                "in_path": _IN_PATH,
                "out_path": _OUT_PATH,
                "k": k,
                "n_blocks": n_blocks,
                "n_codebooks": n_codebooks,
                "num_experts": num_experts,
                "out_dim": out_dim,
                "in_dim": rows.shape[1],
                "params": params.model_dump(),
                "name": name,
            },
        )

        result = torch.load(_OUT_PATH, weights_only=False)

        os.unlink(_IN_PATH)
        os.unlink(_OUT_PATH)

        return result["aqlm"], result["cwr"]

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            self._proc = None
