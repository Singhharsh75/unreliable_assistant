"""Gate 17: tenant-scoped answer cache (design.md §1.3 step 3, ADR-003, KPI-27/28).

Key `cache:{tenant_id}:` + SHA256(tenant_id:norm_query). Only finished, non-personalized HYBRID_RAG answers are
stored (never Tier 2/3 routes, refusals or queries that carried PII). TTL 300 s ± 45 s. A hit is re-checked on
egress: an entry whose body names another tenant is dropped and counted as a leak.

shortcut: in-process dict, one per API worker, exact match on the normalized query. Move to Redis with an
embedding-similarity lookup inside the tenant prefix when there is more than one worker.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

log = logging.getLogger("context.cache")
TTL_S, JITTER_S = 300.0, 45.0
_PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")  # redacted PII: the answer may be about that person


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s<>-]", " ", query.lower())).strip()


def key(tenant_id: str, query: str) -> str:
    return f"cache:{tenant_id}:" + hashlib.sha256(f"{tenant_id}:{normalize(query)}".encode()).hexdigest()


class AnswerCache:
    def __init__(self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None) -> None:
        self.max_entries, self.clock, self.rng = max_entries, clock, rng or random.Random()
        self._d: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self.hits = self.misses = self.leaks_blocked = 0

    @staticmethod
    def cacheable(query: str, out: dict[str, Any]) -> bool:
        return (out["route"] == "HYBRID_RAG" and out["status"] == "DONE" and not _PLACEHOLDER.search(query)
                and bool(out["answer"]))

    def get(self, tenant_id: str, query: str) -> dict[str, Any] | None:
        k = key(tenant_id, query)
        entry = self._d.get(k)
        if entry is None or entry[0] < self.clock():
            self._d.pop(k, None)
            self.misses += 1
            return None
        body = entry[1]
        if body["tenant_id"] != tenant_id:  # egress re-check (ADR-003); cannot happen with tenant-keyed entries
            log.error("cache entry %s belongs to another tenant; dropped", k)
            self._d.pop(k, None)
            self.leaks_blocked += 1
            self.misses += 1
            return None
        self._d.move_to_end(k)
        self.hits += 1
        return dict(body)

    def put(self, tenant_id: str, query: str, out: dict[str, Any]) -> None:
        if not self.cacheable(query, out):
            return
        self._d[key(tenant_id, query)] = (self.clock() + TTL_S + self.rng.uniform(-JITTER_S, JITTER_S), dict(out))
        self._d.move_to_end(key(tenant_id, query))
        while len(self._d) > self.max_entries:
            self._d.popitem(last=False)

    @property
    def hit_ratio(self) -> float | None:
        n = self.hits + self.misses
        return self.hits / n if n else None
