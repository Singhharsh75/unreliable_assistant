"""Gate 11: automated incident detection and runbook mitigation (blueprint Gate 10 & 11 table).

WindowedMetrics is an OTel span processor, so the alerts read the same spans that go to Tempo and Prometheus,
with no second instrumentation path. IncidentResponder.evaluate() checks the four alert rules and applies the
mitigation for each new incident by changing RuntimeConfig, which the orchestrator reads on every request.
A mitigation stays on until an operator calls resolve(); it is never rolled back automatically.

| Incident               | Trigger                                    | Mitigation                                          |
| latency_breach         | agent.query p95 > 550 ms over 30 s          | semantic cache cosine 0.92, BM25/dense top_k 50->25 |
| hallucination_surge    | RAGAS faithfulness < 0.90 over 5 min        | rerank cutoff 0.42->0.55, extractive-quote mode     |
| mcp_server_hang        | > 1% of a pool's calls slower than 150 ms   | trip the pool's bulkhead, fail over, restart worker |
| cache_stampede_or_leak | tenant key collision, or Redis CPU > 85%    | key = SHA256(tenant:user:query), TTL 300 s ± 45 s   |

shortcut: the semantic cache and Redis (Gate 17) and a read-replica index do not exist yet, so the cache knobs,
`rag_index` failover and the worker restart are recorded in RuntimeConfig and the incident log but have no effect.
Wire them when the cache and the replica index land.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode

from harness.bulkhead import Bulkhead

log = logging.getLogger("observability.incidents")
# A call the loop's wait_for cut off ends its span with CancelledError, not UPSTREAM_TIMEOUT.
_TIMEOUT_ERRORS = frozenset({"UPSTREAM_TIMEOUT", "CancelledError", "TimeoutError"})


@dataclass
class RuntimeConfig:
    """Knobs the runbooks turn. Defaults are the design.md values."""
    top_k: int = 50                           # BM25 and dense candidates
    min_rerank_score: float = 0.42            # abstention cutoff (ADR-006)
    extractive_only: bool = False             # release only verbatim quotes, never a generated draft
    cache_cosine_threshold: float | None = None  # None = exact-key cache only
    cache_key_scope: Literal["tenant", "tenant_user"] = "tenant"
    cache_ttl_s: int = 300
    cache_ttl_jitter_s: int = 45
    rag_index: Literal["primary", "replica"] = "primary"
    request_timeout_s: float = 30.0           # whole request; the ReAct loop has its own tighter step budgets


@dataclass(frozen=True)
class Incident:
    name: str
    trigger: str
    actions: tuple[str, ...]
    at: float


@dataclass
class _Rec:
    t: float
    name: str
    ms: float
    pool: str | None
    error: str | None


class WindowedMetrics(SpanProcessor):
    """Keeps the last `horizon_s` of finished spans in memory for the alert rules."""

    def __init__(self, horizon_s: float = 300.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.horizon_s, self.clock = horizon_s, clock
        self._recs: deque[_Rec] = deque()
        self._lock = threading.Lock()  # spans end on worker threads too
        self.cache_collisions = 0
        self.redis_cpu: float | None = None

    def on_end(self, span: ReadableSpan) -> None:
        attrs = span.attributes or {}
        error = attrs.get("error.code") or ("ERROR" if span.status.status_code is StatusCode.ERROR else None)
        self.record(span.name, (span.end_time - span.start_time) / 1e6, attrs.get("pool"), error)

    def record(self, name: str, ms: float, pool: str | None = None, error: str | None = None) -> None:
        now = self.clock()
        with self._lock:
            self._recs.append(_Rec(now, name, ms, pool, error))
            while self._recs and self._recs[0].t < now - self.horizon_s:
                self._recs.popleft()

    def recent(self, name: str, within_s: float) -> list[_Rec]:
        cutoff = self.clock() - within_s
        with self._lock:
            return [r for r in self._recs if r.name == name and r.t >= cutoff]

    def p95(self, name: str, within_s: float) -> tuple[float | None, int]:
        xs = sorted(r.ms for r in self.recent(name, within_s))
        if not xs:
            return None, 0
        return xs[min(len(xs) - 1, int(0.95 * len(xs)))], len(xs)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@dataclass
class Thresholds:
    latency_p95_ms: float = 550.0
    latency_window_s: float = 30.0
    latency_min_samples: int = 20
    faithfulness_floor: float = 0.90
    faithfulness_min_samples: int = 10
    hang_ms: float = 150.0
    hang_rate: float = 0.01
    hang_window_s: float = 60.0
    hang_min_calls: int = 100
    redis_cpu: float = 0.85


class IncidentResponder:
    def __init__(
        self,
        metrics: WindowedMetrics,
        config: RuntimeConfig,
        *,
        auditor: Any = None,  # eval.ragas_auditor.RagasAuditor, for the faithfulness window
        bulkheads: dict[str, Bulkhead] | None = None,
        thresholds: Thresholds | None = None,
    ) -> None:
        self.metrics, self.config, self.auditor = metrics, config, auditor
        self.bulkheads = bulkheads or {}
        self.t = thresholds or Thresholds()
        self.active: dict[str, Incident] = {}
        self.history: list[Incident] = []
        self._defaults = replace(config)

    def evaluate(self) -> list[Incident]:
        """Run every rule once. Returns the incidents opened by this call."""
        opened = []
        for name, check in (("latency_breach", self._latency), ("hallucination_surge", self._faithfulness),
                            ("mcp_server_hang", self._hang), ("cache_stampede_or_leak", self._cache)):
            if name in self.active:
                continue
            fired = check()
            if fired is not None:
                trigger, actions = fired
                inc = Incident(name, trigger, tuple(actions), self.metrics.clock())
                self.active[name] = inc
                self.history.append(inc)
                log.warning(json.dumps({"event": "incident.opened", "incident": name, "trigger": trigger,
                                        "actions": list(actions)}))
                opened.append(inc)
        return opened

    def resolve(self, name: str) -> None:
        """Operator action: close the incident and restore the knobs its runbook changed."""
        inc = self.active.pop(name, None)
        if inc is None:
            return
        for f in _KNOBS[name]:
            setattr(self.config, f, getattr(self._defaults, f))
        if name == "mcp_server_hang":
            for b in self.bulkheads.values():
                b.reset()
        log.warning(json.dumps({"event": "incident.resolved", "incident": name}))

    # --- rules: each returns (trigger, actions) after applying the mitigation, or None -----------

    def _latency(self) -> tuple[str, list[str]] | None:
        p95, n = self.metrics.p95("agent.query", self.t.latency_window_s)
        if p95 is None or n < self.t.latency_min_samples or p95 <= self.t.latency_p95_ms:
            return None
        self.config.cache_cosine_threshold, self.config.top_k = 0.92, 25
        return f"e2e p95 {p95:.0f} ms > {self.t.latency_p95_ms:.0f} ms over {n} requests", [
            "cache_cosine_threshold=0.92", "top_k=25"]

    def _faithfulness(self) -> tuple[str, list[str]] | None:
        if self.auditor is None:
            return None
        mean, n = self.auditor.window("faithfulness")
        if mean is None or n < self.t.faithfulness_min_samples or mean >= self.t.faithfulness_floor:
            return None
        self.config.min_rerank_score, self.config.extractive_only = 0.55, True
        return f"RAGAS faithfulness {mean:.3f} < {self.t.faithfulness_floor} over {n} samples", [
            "min_rerank_score=0.55", "extractive_only=true"]

    def _hang(self) -> tuple[str, list[str]] | None:
        calls = self.metrics.recent("mcp.client", self.t.hang_window_s)
        hung_pools = []
        for pool in sorted({c.pool for c in calls if c.pool}):
            mine = [c for c in calls if c.pool == pool]
            slow = sum(c.ms > self.t.hang_ms or c.error in _TIMEOUT_ERRORS for c in mine)
            if len(mine) >= self.t.hang_min_calls and slow / len(mine) > self.t.hang_rate:
                hung_pools.append((pool, slow, len(mine)))
        if not hung_pools:
            return None
        actions = []
        for pool, _, _ in hung_pools:
            if pool in self.bulkheads:
                self.bulkheads[pool].trip()
                actions.append(f"trip_bulkhead:{pool}")
            if pool == "rag":
                self.config.rag_index = "replica"
                actions.append("rag_index=replica")
            actions.append(f"restart_worker:{pool}")
        detail = ", ".join(f"{p} {s}/{n} > {self.t.hang_ms:.0f} ms" for p, s, n in hung_pools)
        return f"MCP timeouts: {detail}", actions

    def _cache(self) -> tuple[str, list[str]] | None:
        cpu = self.metrics.redis_cpu
        if self.metrics.cache_collisions == 0 and (cpu is None or cpu <= self.t.redis_cpu):
            return None
        self.config.cache_key_scope, self.config.cache_ttl_s, self.config.cache_ttl_jitter_s = "tenant_user", 300, 45
        why = (f"{self.metrics.cache_collisions} tenant key collisions" if self.metrics.cache_collisions
               else f"Redis CPU {cpu:.0%}")
        return why, ["cache_key_scope=tenant_user", "cache_ttl=300±45s"]


_KNOBS: dict[str, tuple[str, ...]] = {
    "latency_breach": ("cache_cosine_threshold", "top_k"),
    "hallucination_surge": ("min_rerank_score", "extractive_only"),
    "mcp_server_hang": ("rag_index",),
    "cache_stampede_or_leak": ("cache_key_scope", "cache_ttl_s", "cache_ttl_jitter_s"),
}
assert all(k in {f.name for f in fields(RuntimeConfig)} for ks in _KNOBS.values() for k in ks)
