"""Gate 17: tenant-scoped answer cache (design.md §1.3 step 3, ADR-003, KPI-27/28).

Key `cache:{tenant_id}:` + SHA256(tenant_id:norm_query). Only finished, non-personalized HYBRID_RAG answers are
stored (never Tier 2/3 routes, refusals or queries that carried PII). TTL 300 s ± 45 s. A hit is re-checked on
egress: an entry whose body names another tenant is dropped and counted as a leak.

`aggressive` lookups (the latency runbook) also match a question that differs only in filler words, inflection and
punctuation. The blueprint's cosine >= 0.92 is not used: opposite questions score above it ("with" / "without QA
sign-off" 0.943, "before" / "after" 0.959), so a hit could return the opposite answer (ADR-018).

shortcut: in-process dict, one per API worker. Move to Redis when there is more than one worker.
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

from mcp_servers.rag_server import bm25_tokens

log = logging.getLogger("context.cache")
TTL_S, JITTER_S = 300.0, 45.0
_PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")  # redacted PII: the answer may be about that person


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s<>-]", " ", query.lower())).strip()


# only words that cannot change what is asked; "with", "not", "before", "only", "was" must stay
_FILLER = frozenset(bm25_tokens("a an the is are do does at in of what which please"))


def key(tenant_id: str, query: str, user_id: str | None = None, aggressive: bool = False) -> str:
    """`user_id` narrows the key to one user (cache-leak runbook); `aggressive` keys the filler-free word sequence."""
    norm = "~" + " ".join(t for t in bm25_tokens(query) if t not in _FILLER) if aggressive else normalize(query)
    scope = f"{tenant_id}:{user_id}" if user_id else tenant_id
    return f"cache:{tenant_id}:" + hashlib.sha256(f"{scope}:{norm}".encode()).hexdigest()


class AnswerCache:
    def __init__(self, max_entries: int = 20_000, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None) -> None:
        self.max_entries, self.clock, self.rng = max_entries, clock, rng or random.Random()
        self._d: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self.hits = self.misses = self.leaks_blocked = 0

    @staticmethod
    def cacheable(query: str, out: dict[str, Any]) -> bool:
        return (out["route"] == "HYBRID_RAG" and out["status"] == "DONE" and not _PLACEHOLDER.search(query)
                and bool(out["answer"]))

    def get(self, tenant_id: str, query: str, *, user_id: str | None = None,
            aggressive: bool = False) -> dict[str, Any] | None:
        now, k, entry = self.clock(), "", None
        for loose in (False, True) if aggressive else (False,):
            k = key(tenant_id, query, user_id, loose)
            entry = self._d.get(k)
            if entry is not None and entry[0] >= now:
                break
            self._d.pop(k, None)
            entry = None
        if entry is None:
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

    def put(self, tenant_id: str, query: str, out: dict[str, Any], *, user_id: str | None = None,
            ttl_s: float = TTL_S, jitter_s: float = JITTER_S) -> None:
        if not self.cacheable(query, out):
            return
        entry = (self.clock() + ttl_s + self.rng.uniform(-jitter_s, jitter_s), dict(out))
        for aggressive in (False, True):  # both keys, so entries are already there when the latency runbook fires
            k = key(tenant_id, query, user_id, aggressive)
            self._d[k] = entry
            self._d.move_to_end(k)
        while len(self._d) > self.max_entries:
            self._d.popitem(last=False)

    @property
    def hit_ratio(self) -> float | None:
        n = self.hits + self.misses
        return self.hits / n if n else None
