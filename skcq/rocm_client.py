"""ROCm-side client: spawns a CUDA subprocess and offloads build_codebook calls.

Used by build.py / extract_and_build_codebooks to run k-means on the 3090
instead of the Strix Halo iGPU. Tensors flow through /dev/shm files or
POSIX SharedMemory, referenced by name in the RPC params.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch

from skcq.clustering import CodebookParams, CodebookResult

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CUDA_PYTHON = os.path.join(_PROJECT_ROOT, "cuda", ".venv", "bin", "python")

_IN_PATH = "/dev/shm/skcq_worker_in.pt"
_OUT_PATH = "/dev/shm/skcq_worker_out.pt"


class RocmClient:
    """Spawns and manages a CUDA subprocess for remote build_codebook calls."""

    def __init__(self, cuda_python: str | None = None) -> None:
        if cuda_python is None:
            cuda_python = _DEFAULT_CUDA_PYTHON
        if not os.path.exists(cuda_python):
            raise FileNotFoundError(
                f"CUDA Python not found: {cuda_python}. Run setup.sh to create cuda/.venv."
            )

        self.child = subprocess.Popen(
            [cuda_python, "-m", "skcq.cuda_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=_PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
            text=True,
        )
        self._id = 0
        atexit.register(self.close)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("CUDA worker spawned (pid=%d)", self.child.pid)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.close()
        raise SystemExit(128 + signum)

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        assert self.child.stdin is not None
        assert self.child.stdout is not None

        self._id += 1
        req: dict[str, Any] = {
            "method": method,
            "params": params,
            "id": self._id,
            "jsonrpc": "2.0",
        }
        self.child.stdin.write(json.dumps(req) + "\n")
        self.child.stdin.flush()

        while True:
            raw = self.child.stdout.readline()
            if not raw:
                raise RuntimeError("CUDA worker closed stdout")
            msg: dict[str, Any] = json.loads(raw)
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
        transport: str = "file",
    ) -> CodebookResult:
        if transport == "file":
            return self._build_codebook_file(
                rows, params, k, n_blocks, n_codebooks, num_experts, out_dim, name
            )
        elif transport == "shm":
            return self._build_codebook_shm(
                rows, params, k, n_blocks, n_codebooks, num_experts, out_dim, name
            )
        raise ValueError(f"unknown transport: {transport}")

    def _build_codebook_file(
        self,
        rows: torch.Tensor,
        params: CodebookParams,
        k: int,
        n_blocks: int,
        n_codebooks: int,
        num_experts: int,
        out_dim: int,
        name: str,
    ) -> CodebookResult:
        rows_cpu = rows.cpu().contiguous()
        torch.save(rows_cpu, _IN_PATH)

        self._rpc(
            "build_codebook",
            {
                "transport": "file",
                "in_path": _IN_PATH,
                "out_path": _OUT_PATH,
                "k": k,
                "n_blocks": n_blocks,
                "n_codebooks": n_codebooks,
                "num_experts": num_experts,
                "out_dim": out_dim,
                "params": params.model_dump(),
                "name": name,
            },
        )

        data = torch.load(_OUT_PATH, map_location="cpu", weights_only=True)

        os.unlink(_IN_PATH)
        os.unlink(_OUT_PATH)

        codebooks = list(data["codebooks"])
        scales = data["scales"]
        if codebooks[0].dtype == torch.float32:
            codebooks = [cb.to(torch.bfloat16) for cb in codebooks]
            scales = scales.to(torch.bfloat16)
        return CodebookResult(
            codebooks=codebooks,
            assignments=list(data["assignments"]),
            scales=scales,
            zero_mask=data["zero_mask"],
            n_blocks=data["n_blocks"],
            n_codebooks=data["n_codebooks"],
        )

    def _build_codebook_shm(
        self,
        rows: torch.Tensor,
        params: CodebookParams,
        k: int,
        n_blocks: int,
        n_codebooks: int,
        num_experts: int,
        out_dim: int,
        name: str,
    ) -> CodebookResult:
        rows_cpu = rows.cpu().contiguous()
        block_size = rows_cpu.shape[1] // n_blocks

        k_list = [max(1, int(k / params.k_residual_mult**c)) for c in range(n_codebooks)]
        cb_shapes = [(n_blocks, block_size, k_c) for k_c in k_list]
        asgn_shape = (num_experts, n_blocks, out_dim)
        scales_shape = (num_experts, n_blocks, out_dim)
        mask_shape = (num_experts * out_dim,)

        specs: list[tuple[str, str, tuple[int, ...], Any, Any]] = [
            ("rows", "skcq_rows", tuple(rows_cpu.shape), rows_cpu.numpy(), np.float32),
        ]
        for c in range(n_codebooks):
            specs.append(
                (
                    f"codebook_{c}",
                    f"skcq_cb_{c}",
                    cb_shapes[c],
                    np.zeros(cb_shapes[c], dtype=np.float32),
                    np.float32,
                )
            )
        for c in range(n_codebooks):
            specs.append(
                (
                    f"assignments_{c}",
                    f"skcq_asgn_{c}",
                    asgn_shape,
                    np.zeros(asgn_shape, dtype=np.int64),
                    np.int64,
                )
            )
        specs.append(
            (
                "scales",
                "skcq_scales",
                scales_shape,
                np.zeros(scales_shape, dtype=np.float32),
                np.float32,
            )
        )
        specs.append(
            ("zero_mask", "skcq_mask", mask_shape, np.zeros(mask_shape, dtype=np.bool_), np.bool_)
        )

        shms: list[shared_memory.SharedMemory] = []
        for shm_name, shm_id, shape, data_arr, dtype in specs:
            itemsize = np.dtype(dtype).itemsize
            nbytes = int(np.prod(shape)) * itemsize
            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_id, track=False)
            shms.append(shm)
            np.ndarray(shape, dtype=dtype, buffer=shm.buf)[:] = data_arr
            self._rpc(
                "attach_buffer",
                {
                    "name": shm_name,
                    "shm": shm_id,
                    "shape": list(shape),
                    "dtype": str(np.dtype(dtype)),
                },
            )

        try:
            self._rpc(
                "build_codebook",
                {
                    "transport": "shm",
                    "input": "rows",
                    "output_codebooks": [f"codebook_{c}" for c in range(n_codebooks)],
                    "output_assignments": [f"assignments_{c}" for c in range(n_codebooks)],
                    "output_scales": "scales",
                    "output_zero_mask": "zero_mask",
                    "k": k,
                    "n_blocks": n_blocks,
                    "n_codebooks": n_codebooks,
                    "num_experts": num_experts,
                    "out_dim": out_dim,
                    "params": params.model_dump(),
                    "name": name,
                },
            )

            codebooks = []
            assignments = []
            for c in range(n_codebooks):
                cb_shm = shms[1 + c]
                asgn_shm = shms[1 + n_codebooks + c]
                codebooks.append(
                    torch.from_numpy(
                        np.ndarray(cb_shapes[c], dtype=np.float32, buffer=cb_shm.buf).copy()
                    )
                )
                assignments.append(
                    torch.from_numpy(
                        np.ndarray(asgn_shape, dtype=np.int64, buffer=asgn_shm.buf).copy()
                    )
                )
            scales = torch.from_numpy(
                np.ndarray(
                    scales_shape, dtype=np.float32, buffer=shms[1 + 2 * n_codebooks].buf
                ).copy()
            )
            zero_mask = torch.from_numpy(
                np.ndarray(mask_shape, dtype=np.bool_, buffer=shms[2 + 2 * n_codebooks].buf).copy()
            )
        finally:
            for shm_name, _, _, _, _ in specs:
                self._rpc("detach_buffer", {"name": shm_name})
            for shm in shms:
                shm.close()
                shm.unlink()

        return CodebookResult(
            codebooks=codebooks,
            assignments=assignments,
            scales=scales,
            zero_mask=zero_mask,
            n_blocks=n_blocks,
            n_codebooks=n_codebooks,
        )

    def close(self) -> None:
        if self.child.poll() is None:
            try:
                self._rpc("quit", {})
                self.child.wait(timeout=5)
            except (RuntimeError, subprocess.TimeoutExpired):
                self.child.kill()
                self.child.wait()
        logger.info("CUDA worker shut down")
