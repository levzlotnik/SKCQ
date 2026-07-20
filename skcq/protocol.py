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
    from skcq.vq_hyperparams import VQConfig

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
    "WorkerInfoMessage",
    "HeartbeatMessage",
    "DisableMessage",
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
    """Orch → Worker: one VQConfig to run.

    The config is a structured VQConfig (from skcq.vq_hyperparams), not a
    bag of CLI args. The worker uses config.primary for the primary codebook
    and config.residuals for the residual chain. K-means iters and shared
    codebook flag are sweep-level settings, not per-codebook — they live on
    the message, not on VQConfig.
    """

    config: VQConfig  # forwarded as TYPE_CHECKING import to avoid runtime cycle
    layer: int
    kmeans_iters: int = 50
    shared: bool = True  # always shared in the sweep (single codebook per level)


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


# ---------------------------------------------------------------------------
# Worker inventory + heartbeat (for dashboard GPU monitoring)
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    """One GPU's static info (sent once on worker connect)."""

    index: int  # device index as seen by torch (after HIP/CUDA_VISIBLE_DEVICES filtering)
    name: str  # "NVIDIA RTX 3090" / "AMD Radeon AI PRO R9700"
    total_vram_mb: int


@dataclass
class WorkerInfoMessage:
    """Worker → Orch (once, on connect): worker identity + device inventory."""

    worker_name: str
    host: str
    devices: list[DeviceInfo]


@dataclass
class DeviceStats:
    """One GPU's live stats (sent in every heartbeat)."""

    index: int
    allocated_mb: int  # PyTorch's view (torch.cuda.memory_allocated)
    reserved_mb: int  # PyTorch's reserved (torch.cuda.memory_reserved)
    used_mb: int  # GPU-wide (from nvidia-smi / rocm-smi — includes non-PyTorch)
    utilization_pct: float  # 0-100, GPU-wide


@dataclass
class HeartbeatMessage:
    """Worker → Orch (every 5s): per-device live stats."""

    worker_name: str
    devices: list[DeviceStats]


@dataclass
class DisableMessage:
    """Orch → Worker: stop accepting new jobs, finish in-flight, exit.

    Sent when user clicks "disable" on a GPU in the dashboard. The worker
    finishes its current job (if any), sends the result, then exits cleanly
    to free VRAM. The orchestrator re-queues the in-flight config.
    """

    pass


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
