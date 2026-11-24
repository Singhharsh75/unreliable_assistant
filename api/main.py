"""Gate 15: public API (design.md §1.2.6) over the Orchestrator.

  uv run uvicorn api.main:app --port 8000                  # real LLM (ANTHROPIC_API_KEY or OPENROUTER_API_KEY)
  LLM_STUB=1 uv run uvicorn api.main:app --port 8000       # load tests: deterministic stand-in, no LLM spend

POST /api/v1/agent/query         JSON response body (§1.2.6).
POST /api/v1/agent/query/stream  text/event-stream: `segment` events (each one verified before it is sent), then one
                                 `final` event with the full response body.
GET  /healthz, GET /metrics      readiness; p50/p95 of LLM TTFT (KPI-25), first-verified-token latency (KPI-31..33), open
                                 incidents. INCIDENT_RESPONDER=0 turns the runbooks off (measurement runs).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import statistics
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from pydantic import BaseModel, ConfigDict, Field

from contracts.models import TENANT_RE, USER_RE, Chunk, GraphSubgraphIn, HybridRetrieveIn
from context.cache import AnswerCache
from harness import LoopConfig, ToolExecutionHarness, ToolSpec, default_bulkheads, plugins
from observability.incidents import IncidentResponder, RuntimeConfig, Thresholds, WindowedMetrics
from observability.tracing import setup_tracing
from triage import ClaudeGenerator, Orchestrator

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("api")
# shortcut: every API caller gets TIER_2_COMPUTE (may submit HITL requests, never mutate); map roles from the
# gateway's authN claims when the gateway lands.
CALLER_TIER = "TIER_2_COMPUTE"
INCIDENT_CHECK_S = 5.0
_TRACEPARENT = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(pattern=TENANT_RE)
    user_id: str = Field(pattern=USER_RE)
    session_id: UUID
    query: str = Field(min_length=1, max_length=4000)


class StubGenerator:
    """Load-test stand-in for the LLM: after STUB_TTFT_MS, streams the top chunk's first two sentences (grounded by
    construction). It measures this service, not Claude; never use it for real traffic."""

    def __init__(self) -> None:
        self.delay = float(os.getenv("STUB_TTFT_MS", "150")) / 1000

    async def stream(self, query: str, chunks: list[Chunk], graph: str = "") -> AsyncIterator[str]:
        await asyncio.sleep(self.delay)
        for sentence in re.split(r"(?<=[.!?])\s+", chunks[0].text)[:2]:
            yield sentence + " "

    async def generate(self, query: str, chunks: list[Chunk], graph: str = "") -> str:
        return "".join([s async for s in self.stream(query, chunks)]).strip()


class _Latency(SpanProcessor):
    """Keeps recent LLM TTFT samples (the `ttft_ms` attribute the orchestrator sets on `llm.generate`)."""

    def __init__(self) -> None:
        self.ttft: deque[float] = deque(maxlen=10_000)

    def on_end(self, span: ReadableSpan) -> None:
        if span.name == "llm.generate" and (v := (span.attributes or {}).get("ttft_ms")) is not None:
            self.ttft.append(float(v))


def build() -> tuple[Orchestrator, IncidentResponder]:
    """Composition root: in-process retriever and state store behind the tool harness and its bulkheads, and the incident
    responder that turns the RuntimeConfig the orchestrator reads."""
    from harness.manifest import verify as verify_manifest
    from mcp_servers.graphify import GraphVault
    from mcp_servers.rag_server import INDEX, Retriever
    from mcp_servers.state_store import StateStore

    db = os.getenv("STATE_DB", str(ROOT / "var" / "state.db"))  # approvals and the audit chain survive restarts
    if db != ":memory:":
        Path(db).parent.mkdir(parents=True, exist_ok=True)
    bulkheads, config, conf = default_bulkheads(), RuntimeConfig(), plugins.load()
    # the replica is its own copy of the index, from RAG_REPLICA_INDEX when there is one on another disk
    primary, replica = Retriever(), Retriever(Path(os.getenv("RAG_REPLICA_INDEX", INDEX)))
    vault = GraphVault()

    def rag_route() -> tuple[str, Any]:
        return ("rag_replica", replica.retrieve) if config.rag_index == "replica" else ("rag", primary.retrieve)

    servers = {
        "mcp-hybrid-rag": [ToolSpec("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ", primary.retrieve, pool="rag",
                                    route=rag_route)],
        "mcp-graphify-vault": [ToolSpec("graph_subgraph", GraphSubgraphIn, "TIER_1_READ", vault.subgraph)],
        "mcp-state-store": StateStore(db).specs(),
    }
    if set(conf.mcpServers) != set(servers):
        raise RuntimeError(f"mcp_config.json lists {sorted(conf.mcpServers)}, the API runs {sorted(servers)}")
    for server, specs in servers.items():  # refuse to start on an unsigned or drifted tool contract
        verify_manifest(server, specs)
    harness = ToolExecutionHarness([s for specs in servers.values() for s in specs], bulkheads=bulkheads)
    if os.getenv("LLM_STUB"):
        log.warning("LLM_STUB is set: answers come from StubGenerator, not Claude")
        generator: Any = StubGenerator()
    else:
        generator = ClaudeGenerator()
    from triage.nli import entailment

    planner = None if os.getenv("LLM_STUB") else build_planner(servers["mcp-state-store"])
    # NLI verifier on every path (ADR-013, ADR-016: KPI-24 is measured on this configuration). The loop's step budgets are
    # widened for a remote LLM planner (ADR-017): the blueprint's 250 ms step cannot hold, and the 30 s request timeout wraps it.
    breaker, ragas, otel = (conf.get(n) for n in ("circuit_breaker", "ragas_auditor", "otel_exporter"))
    cfg = LoopConfig(per_step_timeout_s=2.0, total_loop_budget_s=25.0, llm_step_timeout_s=10.0,
                     breaker_threshold=breaker.max_identical_calls)
    if otel and otel.enabled and otel.endpoint:  # read by setup_tracing
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", otel.endpoint)
    auditor, thresholds = None, Thresholds()
    if ragas and ragas.enabled:  # LLM spend: one judge run per sampled answer
        from eval.ragas_auditor import RagasAuditor, RagasScorer

        auditor, thresholds = RagasAuditor(RagasScorer()), Thresholds(faithfulness_floor=ragas.min_faithfulness)
    cache = AnswerCache()
    orchestrator = Orchestrator(harness, generator, planner, llm_pool=bulkheads["llm"], cache=cache, entail=entailment,
                                loop_config=cfg, config=config, auditor=auditor)
    return orchestrator, IncidentResponder(WindowedMetrics(), config, auditor=auditor, bulkheads=bulkheads, cache=cache,
                                           workers={"rag": primary.restart, "rag_replica": replica.restart},
                                           thresholds=thresholds)


def build_planner(state_specs: list[ToolSpec]) -> Any:
    """ADR-017: the Execution Subagent's LLM planner, bound to the state-store tools its allow-list names."""
    from triage.llm_planner import LLMPlanner
    from triage.subagents import SUBAGENTS

    models = {s.name: s.input_model for s in state_specs if s.name in SUBAGENTS["execution"].allowed_tools}
    client: list[Any] = []

    async def complete(system: str, user: str) -> str | None:
        from eval.run_ragas import GENERATOR, complete as llm_complete, make_client  # lazy: the eval module imports ragas

        client[:] = client or [make_client()]
        return await llm_complete(client[0], GENERATOR, user, system=system)

    return LLMPlanner(complete, models)


latency = _Latency()
first_token: deque[float] = deque(maxlen=10_000)  # receipt -> first verified segment, or final body without an LLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    responder: IncidentResponder | None = None
    if not hasattr(app.state, "orchestrator"):  # tests inject their own
        app.state.orchestrator, responder = build()
    app.state.responder = responder
    setup_tracing(latency, *([responder.metrics] if responder else []))
    auditor, watcher = app.state.orchestrator.auditor, None
    if auditor is not None:
        auditor.start()
    if responder is not None and os.getenv("INCIDENT_RESPONDER", "1") != "0":
        watcher = asyncio.create_task(_watch(responder))
    yield
    if watcher is not None:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher
    if auditor is not None:
        await auditor.stop()


async def _watch(responder: IncidentResponder) -> None:
    while True:
        await asyncio.sleep(INCIDENT_CHECK_S)
        try:
            responder.evaluate()
        except Exception:  # a failing rule must not end incident detection
            log.exception("incident evaluation failed")


app = FastAPI(title="unreliable-assistant", lifespan=lifespan)


def _checked(body: QueryIn, x_tenant_id: str, traceparent: str | None) -> tuple[dict[str, Any], str]:
    if x_tenant_id != body.tenant_id:
        raise HTTPException(403, "X-Tenant-ID does not match tenant_id")
    tp = traceparent if traceparent and _TRACEPARENT.match(traceparent) else f"00-{uuid4().hex}-{uuid4().hex[:16]}-01"
    return {**body.model_dump(), "session_id": str(body.session_id)}, tp


@app.post("/api/v1/agent/query")
async def query(body: QueryIn, x_tenant_id: str = Header(), traceparent: str | None = Header(None)) -> dict:
    t0 = time.perf_counter()
    req, tp = _checked(body, x_tenant_id, traceparent)
    out = await app.state.orchestrator.handle(req, traceparent=tp, caller_tier=CALLER_TIER)
    first_token.append((time.perf_counter() - t0) * 1000)
    return out


@app.post("/api/v1/agent/query/stream")
async def query_stream(body: QueryIn, x_tenant_id: str = Header(), traceparent: str | None = Header(None)):
    t0 = time.perf_counter()
    req, tp = _checked(body, x_tenant_id, traceparent)
    q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def segment(text: str) -> None:
        await q.put(("segment", {"text": text}))

    async def run() -> None:
        try:
            await q.put(("final", await app.state.orchestrator.handle(req, traceparent=tp, caller_tier=CALLER_TIER,
                                                                      on_segment=segment)))
        except Exception as e:  # handle() already turns tool failures into HALTED; this is a bug, not a request error
            log.exception("stream failed")
            await q.put(("error", {"error": type(e).__name__}))

    async def events() -> AsyncIterator[str]:
        task, first = asyncio.create_task(run()), True
        try:
            while True:
                kind, data = await q.get()
                if first:
                    first_token.append((time.perf_counter() - t0) * 1000)
                    first = False
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                if kind != "segment":
                    return
        finally:
            if not task.done():  # client went away: stop generating
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


def _pcts(xs: list[float]) -> dict[str, float | int | None]:
    if len(xs) < 2:
        return {"n": len(xs), "p50": xs[0] if xs else None, "p95": xs[0] if xs else None, "p99": xs[0] if xs else None}
    q = statistics.quantiles(xs, n=100)
    return {"n": len(xs), "p50": round(q[49], 2), "p95": round(q[94], 2), "p99": round(q[98], 2)}


@app.get("/metrics")
async def metrics() -> dict:
    cache, responder = app.state.orchestrator.cache, app.state.responder
    return {"generator": type(app.state.orchestrator.generator).__name__,  # StubGenerator TTFT is not Claude's
            "cache": {"hits": cache.hits, "misses": cache.misses, "hit_ratio": cache.hit_ratio} if cache else None,
            "incidents": {n: i.trigger for n, i in responder.active.items()} if responder else {},
            "llm_ttft_ms": _pcts(list(latency.ttft)), "first_token_ms": _pcts(list(first_token))}
