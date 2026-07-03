#!/usr/bin/env python3
"""CUDA server: JSON-RPC over stdin/stdout, runs build_codebook on the 3090.

Launched by skcq/rocm_client.py as a subprocess with the CUDA venv's Python.
Tensors flow through /dev/shm files (torch.save/load with mmap) or POSIX
SharedMemory, referenced by name in the RPC params.

Progress notifications are emitted as JSON-RPC notifications (no id) on stdout
before the final response, so the client can stream k-means progress.

Usage (indirect — spawned by rocm_client):
    cuda/.venv/bin/python -m skcq.cuda_server

Requires PYTHONPATH to include the project root for skcq.clustering import.
"""

from __future__ import annotations

import json
import logging
import sys
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch
from jsonrpc import JSONRPCResponseManager
from jsonrpc.dispatcher import Dispatcher

from skcq.clustering import CodebookParams, build_codebook


class ProgressNotificationHandler(logging.Handler):
    """Log handler that emits JSON-RPC progress notifications to stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        notification = {
            "jsonrpc": "2.0",
            "method": "progress",
            "params": {"message": record.getMessage()},
        }
        sys.stdout.write(json.dumps(notification) + "\n")
        sys.stdout.flush()


class CudaServer:
    """Stateful server for remote build_codebook calls."""

    def __init__(self) -> None:
        self.buffers: dict[str, torch.Tensor] = {}
        self._shm_refs: dict[str, shared_memory.SharedMemory] = {}

    def attach_buffer(self, name: str, shm: str, shape: list[int], dtype: str) -> dict[str, Any]:
        shm_obj = shared_memory.SharedMemory(name=shm, track=False)
        arr = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=shm_obj.buf)
        self.buffers[name] = torch.from_numpy(arr)
        self._shm_refs[name] = shm_obj
        return {"status": "ok", "name": name, "shape": shape, "dtype": dtype}

    def detach_buffer(self, name: str) -> dict[str, Any]:
        self.buffers.pop(name, None)
        shm_obj = self._shm_refs.pop(name, None)
        if shm_obj is not None:
            shm_obj.close()
        return {"status": "ok", "name": name}

    def build_codebook(
        self,
        transport: str,
        k: int,
        n_blocks: int,
        n_codebooks: int,
        num_experts: int,
        out_dim: int,
        params: dict[str, Any] | None = None,
        name: str = "",
        in_path: str | None = None,
        out_path: str | None = None,
        input: str | None = None,
        output_codebooks: list[str] | None = None,
        output_assignments: list[str] | None = None,
        output_scales: str | None = None,
        output_zero_mask: str | None = None,
    ) -> dict[str, Any]:
        cb_params = CodebookParams(**params) if params is not None else CodebookParams()

        if transport == "file":
            assert in_path is not None
            rows = torch.load(in_path, mmap=True, map_location="cpu", weights_only=True).to("cuda")
        elif transport == "shm":
            assert input is not None
            rows = self.buffers[input].to("cuda")
        else:
            return {"status": "error", "msg": f"unknown transport: {transport}"}

        result = build_codebook(
            rows=rows,
            params=cb_params,
            k=k,
            n_blocks=n_blocks,
            n_codebooks=n_codebooks,
            num_experts=num_experts,
            out_dim=out_dim,
            device=torch.device("cuda"),
            name=name,
        )

        if transport == "file":
            assert out_path is not None
            torch.save(
                {
                    "codebooks": result.codebooks,
                    "assignments": result.assignments,
                    "scales": result.scales,
                    "zero_mask": result.zero_mask,
                    "n_blocks": result.n_blocks,
                    "n_codebooks": result.n_codebooks,
                },
                out_path,
            )
        elif transport == "shm":
            assert (
                output_codebooks is not None
                and output_assignments is not None
                and output_scales is not None
                and output_zero_mask is not None
                and len(output_codebooks) == result.n_codebooks
                and len(output_assignments) == result.n_codebooks
            )
            for i, buf_name in enumerate(output_codebooks):
                self.buffers[buf_name].copy_(result.codebooks[i])
            for i, buf_name in enumerate(output_assignments):
                self.buffers[buf_name].copy_(result.assignments[i])
            self.buffers[output_scales].copy_(result.scales)
            self.buffers[output_zero_mask].copy_(result.zero_mask)

        return {"status": "ok"}

    def quit(self) -> dict[str, Any]:
        for name in list(self.buffers.keys()):
            self.detach_buffer(name)
        return {"status": "ok", "quit": True}


def serve(dispatcher: Dispatcher) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp = JSONRPCResponseManager.handle(line, dispatcher)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp.data) + "\n")
        sys.stdout.flush()
        result = resp.data.get("result")
        if isinstance(result, dict) and result.get("quit"):
            break


def main() -> None:
    clustering_logger = logging.getLogger("skcq.clustering")
    clustering_logger.setLevel(logging.INFO)
    clustering_logger.addHandler(ProgressNotificationHandler())
    clustering_logger.propagate = False

    server = CudaServer()
    dispatcher = Dispatcher()
    dispatcher.add_method(server.build_codebook, "build_codebook")
    dispatcher.add_method(server.attach_buffer, "attach_buffer")
    dispatcher.add_method(server.detach_buffer, "detach_buffer")
    dispatcher.add_method(server.quit, "quit")
    serve(dispatcher)


if __name__ == "__main__":
    main()
