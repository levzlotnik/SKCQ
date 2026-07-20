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

Storage:
  - In-memory: dict[key -> torch.Tensor] (process-local, fast)
  - On-disk: vq_cache/<key>.pt (persists across runs)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger("skcq.vq_cache")


class PrimaryCodebookCache:
    """Two-tier cache for trained primary codebooks.

    Keys are opaque strings — the caller constructs them (typically via
    VQCodebookSpec.cache_key(layer, projection) from skcq.vq_hyperparams).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._mem: dict[str, torch.Tensor] = {}
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> torch.Tensor | None:
        """Look up a cached primary codebook. Returns the codebook tensor or None."""
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
        self._mem[key] = codebook.detach().cpu()
        if self._cache_dir is not None:
            disk_path = self._cache_dir / f"{key}.pt"
            try:
                torch.save(codebook.detach().cpu(), disk_path)
            except Exception:
                logger.warning("Cache store failed for %s", key)

    def stats(self) -> dict:
        """Return cache statistics for the dashboard."""
        return {
            "in_memory_entries": len(self._mem),
            "hits": self._hits,
            "misses": self._misses,
            "cache_dir": str(self._cache_dir) if self._cache_dir else None,
        }
