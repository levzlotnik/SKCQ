"""Typed message protocol for orchestrator ↔ worker communication.

All messages are dataclasses pickled over TCP with length-prefixed framing.
Both sides import these types — no more untyped dicts.

Wire protocol:
    Worker → Orch: ReadyMessage, ResultsMessage, ErrorMessage
    Orch → Worker: JobMessage, AckMessage, DoneMessage
"""

from __future__ import annotations

import pickle
import socket
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from skcq.clustering import CodebookResult

# Union of all message types. recv_frame returns this or None.
Message = Union[
    "ReadyMessage",
    "JobMessage",
    "ResultsMessage",
    "ErrorMessage",
    "AckMessage",
    "DoneMessage",
    "VQJobMessage",
    "VQResultsMessage",
    "VQErrorMessage",
]


@dataclass
class ReadyMessage:
    """Worker → Orch: worker is ready for a job."""

    device: str


@dataclass
class JobMessage:
    """Orch → Worker: here's a layer to build."""

    layer: int
    model_id: str
    params: dict[str, object]
    num_experts: int
    hidden_size: int
    intermediate_size: int


@dataclass
class ResultsMessage:
    """Worker → Orch: layer build complete, here are the results."""

    layer: int
    data: dict[str, CodebookResult]


@dataclass
class ErrorMessage:
    """Worker → Orch: layer build failed."""

    layer: int
    msg: str


@dataclass
class AckMessage:
    """Orch → Worker: results received, proceed to next job."""

    pass


@dataclass
class DoneMessage:
    """Orch → Worker: no more jobs, exit."""

    pass


# ---------------------------------------------------------------------------
# VQ hyperparameter sweep messages
# ---------------------------------------------------------------------------
# A VQ job is ONE config (one projection, one set of hyperparams) — NOT a
# layer. The worker loads layer weights once at startup, then loops over
# VQJobMessages, each containing the hyperparameters for one config. Results
# are flat dicts (CSV row shape), not CodebookResult objects.


@dataclass
class VQJobMessage:
    """Orch → Worker: hyperparameters for one VQ config.

    Fields match the CLI flags of experiments/weight_quant_error.py.
    The worker calls run_one_kmeans() directly with these.
    """

    config_id: str
    projection: str  # "gate" | "up" | "down"
    block_size: int
    K: int
    n_codebooks: int
    residual_block_sizes: list[int] | None
    residual_k: list[int] | int | None
    codebook_bits: int
    # Fixed sweep hyperparams (echoed for traceability):
    metric: str  # "cosine"
    scale_dtype: str  # "int8"
    kmeans_iters: int
    shared: bool
    sign_split: bool


@dataclass
class VQResultsMessage:
    """Worker → Orch: results for one VQ config (CSV row shape).

    `extra_rows` carries integer baselines or other auxiliary rows that
    should be inserted alongside the main result. The orchestrator inserts
    each as a separate DB row (using each extra row's `scheme` as its
    config_id suffix).
    """

    config_id: str
    row: dict
    extra_rows: list[dict] | None = None


@dataclass
class VQErrorMessage:
    """Worker → Orch: VQ config failed."""

    config_id: str
    msg: str


@dataclass
class WorkerConfig:
    """One worker entry in workers.yaml."""

    name: str
    host: str
    venv: str
    workdir: str = "."
    device: str = "auto"
    chunk_budget_mb: int = 2048


@dataclass
class WorkersConfig:
    """Parsed workers.yaml."""

    orchestrator_host: str
    port: int
    workers: list[WorkerConfig]


@dataclass
class ModelConfig:
    """Model dimensions (read from HF config, no weight loading)."""

    num_experts: int
    hidden_size: int
    moe_intermediate_size: int


@dataclass
class ProjectionSpec:
    """Spec for one projection within a layer job."""

    name: str
    rows: object  # torch.Tensor, but can't import at runtime for dataclass default
    k: int
    n_blocks: int
    out_dim: int


# --- Framing ---


def send_frame(sock: socket.socket, obj: Message) -> None:
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_frame(sock: socket.socket) -> Message | None:
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    data = _recv_exactly(sock, length)
    if data is None:
        return None
    return pickle.loads(data)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
