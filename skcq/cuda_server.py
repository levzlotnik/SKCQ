#!/usr/bin/env python3
"""CUDA server: JSON-RPC over stdin/stdout, runs codebook builds on the 3090.

Launched by skcq/rocm_client.py as a subprocess with the CUDA venv's Python.
Tensors flow through /dev/shm files (torch.save/load).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import torch
from jsonrpc import JSONRPCResponseManager
from jsonrpc.dispatcher import Dispatcher

from skcq.config import CodebookParams, build_aqlm_codebook


class ProgressNotificationHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        notification = {
            "jsonrpc": "2.0",
            "method": "progress",
            "params": {"message": record.getMessage()},
        }
        sys.stdout.write(json.dumps(notification) + "\n")
        sys.stdout.flush()


class CudaServer:
    def build_codebook(
        self,
        in_path: str,
        out_path: str,
        k: int,
        n_blocks: int,
        n_codebooks: int,
        num_experts: int,
        out_dim: int,
        in_dim: int,
        params: dict[str, Any] | None = None,
        name: str = "",
    ) -> dict[str, Any]:
        cb_params = CodebookParams(**params) if params is not None else CodebookParams()

        rows = torch.load(in_path, mmap=True, map_location="cuda", weights_only=True)

        aqlm = build_aqlm_codebook(
            cb_params,
            k=k,
            n_blocks=n_blocks,
            n_codebooks=n_codebooks,
            in_dim=in_dim,
            out_dim=out_dim,
            num_experts=num_experts,
            device=torch.device("cuda"),
            name=name,
        )
        cwr = aqlm.fit(rows)

        torch.save({"aqlm": aqlm, "cwr": cwr}, out_path)

        return {"status": "ok", "out_path": out_path}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.getLogger().addHandler(ProgressNotificationHandler())

    server = CudaServer()
    dispatcher = Dispatcher()
    dispatcher.add_method(server.build_codebook, "build_codebook")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = JSONRPCResponseManager.handle(line, dispatcher)
        sys.stdout.write(response.json + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
