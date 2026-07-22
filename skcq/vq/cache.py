"""On-disk + in-memory cache for trained primary codebooks.

The primary codebook is the most expensive part of VQ training (full k-means
with iters iterations). Many configs share the same primary — e.g. a primary
with bs=8 K=8192 cosine+ssvq+int8 is shared by all configs that vary only in
residual codebooks. Caching the trained primary codebook tensor avoids
re-running k-means for every config.

Cache key: opaque string (caller constructs it — typically from
VQCodebookSpec.cache_key(layer, projection)).
Cache value: codebook tensor only (bs, K) or (n_blocks, bs, K) for non-shared.
  The assignment is re-computed on cache hit (one pass of chunk @ cb.t()
  argmax — much cheaper than full k-means).

Two implementations:
  - ``PrimaryCodebookCache``: on-disk + in-memory, lives on the orchestrator.
    Thread-safe (multiple worker connections can hit it concurrently).
  - ``RemotePrimaryCodebookCache``: wire proxy, lives on the worker. Proxies
    get/put over the TCP socket to the orchestrator's PrimaryCodebookCache
    via CacheRequestMessage / CacheResponseMessage / CacheStoreMessage.
    Requires a threading.Lock to serialize send_frame calls with the
    heartbeat thread that shares the same socket.

Storage (PrimaryCodebookCache):
  - In-memory: dict[key -> torch.Tensor] (process-local, fast)
  - On-disk: vq_cache/<key>.pt (persists across runs)
"""

from __future__ import annotations

import logging
import socket
import threading
from pathlib import Path

import torch

from skcq.protocol import (
    CacheRequestMessage,
    CacheResponseMessage,
    CacheStoreMessage,
    recv_frame,
    send_frame,
)

logger = logging.getLogger("skcq.vq_cache")


class PrimaryCodebookCache:
    """Two-tier cache for trained primary codebooks.

    Keys are opaque strings — the caller constructs them (typically via
    VQCodebookSpec.cache_key(layer, projection) from skcq.vq_hyperparams).
    Thread-safe: a threading.Lock guards the in-memory dict + disk I/O so
    multiple worker connections (each handled in its own thread on the
    orchestrator) can call get/put concurrently.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._mem: dict[str, torch.Tensor] = {}
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> torch.Tensor | None:
        """Look up a cached primary codebook. Returns the codebook tensor or None."""
        with self._lock:
            if key in self._mem:
                self._hits += 1
                return self._mem[key]
            if self._cache_dir is not None:
                disk_path = self._cache_dir / f"{key}.pt"
                if disk_path.exists():
                    try:
                        cb = torch.load(disk_path, map_location="cpu", weights_only=True)
                        self._mem[key] = cb
                        self._hits += 1
                        return cb
                    except Exception:
                        logger.warning("Cache load failed for %s, removing", key)
                        disk_path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def put(self, key: str, codebook: torch.Tensor) -> None:
        """Store a trained primary codebook in the cache."""
        with self._lock:
            self._mem[key] = codebook.detach().cpu()
            if self._cache_dir is not None:
                disk_path = self._cache_dir / f"{key}.pt"
                try:
                    torch.save(codebook.detach().cpu(), disk_path)
                except Exception:
                    logger.warning("Cache store failed for %s", key)

    def stats(self) -> dict:
        """Return cache statistics for the dashboard."""
        with self._lock:
            return {
                "in_memory_entries": len(self._mem),
                "hits": self._hits,
                "misses": self._misses,
                "cache_dir": str(self._cache_dir) if self._cache_dir else None,
            }


class RemotePrimaryCodebookCache:
    """Wire-proxy cache for workers.

    Implements the same ``get``/``put`` interface as ``PrimaryCodebookCache``
    but proxies to the orchestrator over the TCP socket. ``get`` sends a
    ``CacheRequestMessage`` and blocks on ``CacheResponseMessage``;
    ``put`` sends a ``CacheStoreMessage`` (fire-and-forget, no response).

    A ``threading.Lock`` serializes all ``send_frame`` calls on this socket
    so the heartbeat thread (which shares the same socket) can't interleave
    bytes mid-frame. The lock is shared with the worker's main loop and
    HeartbeatThread — all three must use the same lock instance.
    """

    def __init__(self, sock: socket.socket, lock: threading.Lock) -> None:
        self._sock = sock
        self._lock = lock
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> torch.Tensor | None:
        """Ask the orchestrator for a cached primary codebook."""
        with self._lock:
            send_frame(self._sock, CacheRequestMessage(key=key))
            resp = recv_frame(self._sock)
        if isinstance(resp, CacheResponseMessage):
            if resp.codebook is not None:
                self._hits += 1
                return resp.codebook
            self._misses += 1
            return None
        logger.warning("Unexpected response to cache request: %s", resp)
        self._misses += 1
        return None

    def put(self, key: str, codebook: torch.Tensor) -> None:
        """Send a freshly-trained codebook to the orchestrator for storage."""
        with self._lock:
            send_frame(self._sock, CacheStoreMessage(key=key, codebook=codebook.detach().cpu()))

    def stats(self) -> dict:
        """Return local cache statistics (hits/misses from this worker's perspective)."""
        return {
            "in_memory_entries": 0,
            "hits": self._hits,
            "misses": self._misses,
            "cache_dir": None,
        }
