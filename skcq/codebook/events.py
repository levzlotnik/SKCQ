from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


# ---------------------------------------------------------------------------
# K-means level events (emitted by KmeansExperiment)
# ---------------------------------------------------------------------------


@dataclass
class KmeansStartEvent:
    name: str
    metric: str
    k: int
    max_iters: int
    n_points: int
    block_idx: int = 0
    n_blocks: int = 1


@dataclass
class KmeansIterEvent:
    iter: int
    max_iters: int
    moved: float
    n_empty: int


@dataclass
class KmeansDoneEvent:
    iters_run: int
    final_moved: float
    converged: bool


# ---------------------------------------------------------------------------
# Codebook level events (emitted by AQLMCodebook, wrapping child events with context)
# ---------------------------------------------------------------------------


@dataclass
class CodebookStartEvent:
    n_codebooks: int
    primary_block_size: int
    primary_k: int
    metric: str


@dataclass
class CodebookIterEvent:
    codebook_idx: int
    n_codebooks: int
    block_idx: int
    n_blocks: int
    iter: int
    max_iters: int
    moved: float
    n_empty: int
    metric: str


@dataclass
class CodebookDoneEvent:
    codebook_idx: int
    n_codebooks: int


# ---------------------------------------------------------------------------
# Experiment base class — event bus with on() / _emit() / _forward_from()
# ---------------------------------------------------------------------------


class Experiment:
    """Event bus: ``on(EventType, callback)`` + internal ``_emit(event)``.

    All codebook classes extend this so listeners can subscribe before ``fit``.
    """

    def __init__(self) -> None:
        self._listeners: dict[type, list[Callable[..., None]]] = defaultdict(list)

    def on(self, event_type: type[T], callback: Callable[[T], None]) -> None:
        self._listeners[event_type].append(callback)

    def _emit(self, event: object) -> None:
        for cb in self._listeners.get(type(event), []):
            cb(event)

    def _forward_from(self, child: Experiment, event_type: type[T]) -> None:
        def handler(e: T) -> None:
            self._emit(e)

        child.on(event_type, handler)


# ---------------------------------------------------------------------------
# TqdmListener — renders k-means events to stderr
# ---------------------------------------------------------------------------


class TqdmListener:
    """Renders k-means progress to stderr.

    For non-shared codebooks (n_blocks > 1): shows a single block-level bar
    that ticks once per block's k-means completion, postfix shows last block's
    convergence. For shared codebooks (n_blocks == 1): shows the per-iteration
    bar as before.
    """

    def __init__(self) -> None:
        self._pbar: tqdm | None = None
        self._n_blocks: int = 1
        self._block_mode: bool = False

    def on_start(self, e: KmeansStartEvent) -> None:
        if e.n_blocks > 1:
            if self._pbar is not None:
                return
            self._block_mode = True
            self._n_blocks = e.n_blocks
            self._pbar = tqdm(total=e.n_blocks, desc=e.name.split(" blk=")[0], leave=True)
        else:
            self._block_mode = False
            self._pbar = tqdm(total=e.max_iters, desc=e.name, leave=True)

    def on_iter(self, e: KmeansIterEvent) -> None:
        if self._pbar is not None and not self._block_mode:
            self._pbar.update(1)
            self._pbar.set_postfix(moved=f"{e.moved:.6f}", empty=e.n_empty)

    def on_done(self, e: KmeansDoneEvent) -> None:
        if self._pbar is not None:
            if self._block_mode:
                self._pbar.update(1)
                self._pbar.set_postfix(
                    iters=e.iters_run, conv=int(e.converged), moved=f"{e.final_moved:.6f}"
                )
                if self._pbar.n >= self._n_blocks:
                    self._pbar.close()
                    self._pbar = None
            else:
                self._pbar.close()
                self._pbar = None
