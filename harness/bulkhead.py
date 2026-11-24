"""Bulkhead isolation per MCP server (design.md §1.3 Bulkheads, Gate 9 item 2).

Each pool caps in-flight calls with a slot count and runs sync handlers on its own thread pool, so a slow reranker
saturates only `rag` and never the threads that serve `db`, `llm`, health checks or triage (triage is in-process
CPU work and takes no pool). A full pool fails fast with UPSTREAM_TIMEOUT instead of queueing without bound.
After `trip_after` consecutive failures the pool opens for `open_s` and every call fails with CIRCUIT_OPEN;
Gate 11's "MCP server hang" runbook can also trip it by hand.
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from harness.errors import CIRCUIT_OPEN, INTERNAL_ERROR, UPSTREAM_TIMEOUT, HarnessError

POOL_SIZES = {"rag": 64, "db": 128, "llm": 256}
# Only infrastructure failures count toward tripping. A client error (unknown lot, bad schema, tenant violation)
# says nothing about pool health, and counting it would let one tenant's bad requests open the pool for everyone.
_INFRA_CODES = frozenset({UPSTREAM_TIMEOUT, INTERNAL_ERROR, CIRCUIT_OPEN})
_POLL_S = 0.002


class Bulkhead:
    def __init__(self, name: str, size: int, *, acquire_timeout_s: float = 0.05, trip_after: int = 5,
                 open_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.name, self.size = name, size
        self.acquire_timeout_s, self.trip_after, self.open_s, self.clock = acquire_timeout_s, trip_after, open_s, clock
        # A thread-safe slot count, not an asyncio.Semaphore: a slot is freed when the worker thread finishes,
        # even after the caller timed out, and the bulkhead works across event loops.
        self._slots = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix=f"bulkhead-{name}")
        self.in_flight = self.failures = self.rejected = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return self.clock() < self._open_until

    def trip(self, seconds: float | None = None) -> None:
        self._open_until = self.clock() + (self.open_s if seconds is None else seconds)

    def reset(self) -> None:
        self._open_until, self.failures = 0.0, 0

    def _try_acquire(self) -> bool:
        with self._slots:
            if self.in_flight >= self.size:
                return False
            self.in_flight += 1
            return True

    def _release(self, *_: Any) -> None:
        with self._slots:
            self.in_flight -= 1

    def _failed(self, exc: BaseException) -> None:
        if isinstance(exc, HarnessError) and exc.code not in _INFRA_CODES:
            return
        self.failures += 1
        if self.failures >= self.trip_after:
            self.trip()

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.is_open:
            self.rejected += 1
            raise HarnessError(CIRCUIT_OPEN, f"{self.name} bulkhead is open; try again later")
        deadline = self.clock() + self.acquire_timeout_s
        while not self._try_acquire():
            if self.clock() >= deadline:
                self.rejected += 1
                raise HarnessError(UPSTREAM_TIMEOUT, f"{self.name} pool is saturated ({self.size} in flight)")
            await asyncio.sleep(_POLL_S)
        try:
            if inspect.iscoroutinefunction(fn):
                try:
                    value = await fn(*args)
                finally:
                    self._release()
            else:
                # copy_context: the worker thread keeps the caller's OTel span. The slot is freed by the thread's
                # own completion, so a hung handler keeps holding it after the caller gives up.
                cf = self._executor.submit(contextvars.copy_context().run, fn, *args)
                cf.add_done_callback(self._release)
                value = await asyncio.wrap_future(cf)
        except (Exception, asyncio.CancelledError) as e:  # a caller timeout (cancel) counts as a failure too
            self._failed(e)
            raise
        self.failures = 0
        return value


def default_bulkheads() -> dict[str, Bulkhead]:
    return {name: Bulkhead(name, size) for name, size in POOL_SIZES.items()}
