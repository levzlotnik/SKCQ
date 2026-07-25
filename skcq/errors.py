from __future__ import annotations


class InfrastructureError(Exception):
    """Raised for infrastructure failures (network, subprocess, disk).

    Wraps the original exception via ``raise InfrastructureError(...) from ex``.
    Caught at job-loop boundaries (worker.py, vq/worker.py) and translated to
    ErrorMessage / VQErrorMessage over the wire. Never caught silently.

    Infrastructure failures include:
    - Network: ConnectionError, OSError on sockets
    - Subprocess: TimeoutExpired on worker process terminate/kill
    - Disk: FileNotFoundError on worker launch, torch.save failures

    NOT for: ValueError, KeyError, RuntimeError (bugs — let crash).
    """

    def __init__(self, message: str = "", *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}
        self.message = message
