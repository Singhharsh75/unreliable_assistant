"""Gates 10-11 — W3C trace propagation across every hop, span-derived alert windows, runbook mitigations."""
import asyncio
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from contracts.models import HybridRetrieveIn, HybridRetrieveOut
from harness import Bulkhead, ToolExecutionHarness, ToolSpec
from harness.errors import CIRCUIT_OPEN, HarnessError
from mcp_servers.state_store import StateStore
from observability.incidents import IncidentResponder, RuntimeConfig, Thresholds, WindowedMetrics
from observability.tracing import setup_tracing
from tests.test_orchestrator import COLD, Gen, chunk
from triage import Orchestrator

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
TP = f"00-{TRACE}-00f067aa0ba902b7-01"
EXPORTER = InMemorySpanExporter()
setup_tracing(SimpleSpanProcessor(EXPORTER))


@pytest.fixture(autouse=True)
def clear():
    EXPORTER.clear()


class FakeRag:
    def __init__(self, out):
        self.out, self.seen = out, []

    def __call__(self, m):
        self.seen.append(m)
        return self.out


def build(rag, gen=None, config=None, **kw):
    s = StateStore()
    spec = ToolSpec("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ", rag, pool="rag")
    h = ToolExecutionHarness([*s.specs(), spec], **kw)
    return Orchestrator(h, gen or Gen("Cold-chain SKUs are stored between 2 and 8 degrees Celsius."), config=config)


def ask(o, q="What temperature are cold-chain SKUs stored at?"):
    r = {"tenant_id": "tenant_acme", "user_id": "user_1", "session_id": str(uuid4()), "query": q}
    return asyncio.run(o.handle(r, traceparent=TP, caller_tier="TIER_2_COMPUTE"))


# --- Gate 10: tracing -------------------------------------------------------------------------------

def test_every_hop_is_a_span_in_the_callers_trace():
    rag = FakeRag(HybridRetrieveOut(chunks=[chunk(COLD)], top_score=0.9, abstain=False, stage_latency_ms={}))
    r = ask(build(rag))
    spans = EXPORTER.get_finished_spans()
    names = {s.name for s in spans}
    assert {"agent.query", "triage.classify", "mcp.client", "llm.generate", "verify.nli"} <= names
    assert {format(s.context.trace_id, "032x") for s in spans} == {TRACE} and r["trace_id"] == TRACE
    root = next(s for s in spans if s.name == "agent.query")
    assert format(root.parent.span_id, "016x") == "00f067aa0ba902b7"  # child of the gateway's span
    assert root.attributes["route"] == "HYBRID_RAG"
    # the tool call carries the agent span as its parent, so a remote MCP server nests under it
    sent = rag.seen[0].ctx.traceparent
    assert sent.split("-")[1] == TRACE and sent != TP


def test_failed_tool_span_carries_the_error_code():
    db = Bulkhead("db", 2)
    db.trip()
    h = ToolExecutionHarness(StateStore().specs(), bulkheads={"db": db})
    ctx = {"tenant_id": "tenant_acme", "user_id": "user_1", "session_id": str(uuid4()),
           "idempotency_key": str(uuid4()), "traceparent": TP}
    with pytest.raises(HarnessError):
        asyncio.run(h.call("lookup_lot", {"ctx": ctx, "lot_id": "LOT-00044"}, caller_tier="TIER_1_READ"))
    (s,) = [s for s in EXPORTER.get_finished_spans() if s.name == "mcp.client"]
    assert s.attributes["error.code"] == "CIRCUIT_OPEN" and s.attributes["pool"] == "db"


def test_retriever_stages_use_design_span_names():
    from mcp_servers import rag_server

    assert rag_server._SPAN["dense"] == "biencoder"


# --- Gate 11: incident detection and runbooks ------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def responder(**kw):
    clock = Clock()
    m = WindowedMetrics(clock=clock)
    cfg = RuntimeConfig()
    return m, cfg, IncidentResponder(m, cfg, **kw), clock


def test_latency_breach_enables_aggressive_cache_and_cuts_top_k():
    m, cfg, ir, _ = responder()
    for _ in range(30):
        m.record("agent.query", 400)
    assert ir.evaluate() == []
    for _ in range(10):
        m.record("agent.query", 700)
    (inc,) = ir.evaluate()
    assert inc.name == "latency_breach" and cfg.cache_cosine_threshold == 0.92 and cfg.top_k == 25
    assert ir.evaluate() == []  # an open incident does not re-fire
    ir.resolve("latency_breach")
    assert cfg.top_k == 50 and cfg.cache_cosine_threshold is None


def test_latency_window_is_30s():
    m, cfg, ir, clock = responder()
    for _ in range(30):
        m.record("agent.query", 900)
    clock.t += 31
    assert ir.evaluate() == [] and cfg.top_k == 50


class Auditor:
    def __init__(self, mean, n):
        self.w = (mean, n)

    def window(self, metric):
        return self.w


def test_hallucination_surge_raises_cutoff_and_forces_extractive_mode():
    _, cfg, ir, _ = responder(auditor=Auditor(0.85, 12))
    (inc,) = ir.evaluate()
    assert inc.name == "hallucination_surge" and cfg.min_rerank_score == 0.55 and cfg.extractive_only
    _, cfg2, ir2, _ = responder(auditor=Auditor(0.85, 3))  # too few samples to act on
    assert ir2.evaluate() == [] and not cfg2.extractive_only


def test_orchestrator_obeys_runbook_knobs():
    rag = FakeRag(HybridRetrieveOut(chunks=[chunk(COLD)], top_score=0.9, abstain=False, stage_latency_ms={}))
    gen = Gen("anything")
    cfg = RuntimeConfig(top_k=25, min_rerank_score=0.55, extractive_only=True)
    r = ask(build(rag, gen=gen, config=cfg))
    req = rag.seen[0]
    assert (req.bm25_top_k, req.dense_top_k, req.min_rerank_score) == (25, 25, 0.55)
    assert gen.n == 0 and r["answer"] in COLD and r["citations"] == [chunk(COLD).uri]


def test_mcp_hang_trips_the_pool_bulkhead_and_fails_over():
    rag = Bulkhead("rag", 4)
    m, cfg, ir, _ = responder(bulkheads={"rag": rag})
    for i in range(200):
        m.record("mcp.client", 200 if i < 3 else 5, pool="rag")  # 1.5% slower than 150 ms
    for _ in range(200):
        m.record("mcp.client", 5, pool="db")
    (inc,) = ir.evaluate()
    assert inc.name == "mcp_server_hang" and rag.is_open and cfg.rag_index == "replica"
    assert "restart_worker:rag" in inc.actions and not any("db" in a for a in inc.actions)

    async def call():
        await rag.run(lambda: 1)
    with pytest.raises(HarnessError) as e:
        asyncio.run(call())
    assert e.value.code == CIRCUIT_OPEN
    ir.resolve("mcp_server_hang")
    assert not rag.is_open and cfg.rag_index == "primary"


def test_mcp_hang_below_one_percent_does_not_fire():
    m, _, ir, _ = responder()
    for i in range(200):
        m.record("mcp.client", 200 if i < 2 else 5, pool="rag")  # exactly 1%
    assert ir.evaluate() == []


def test_cache_leak_scopes_keys_to_user():
    m, cfg, ir, _ = responder()
    m.redis_cpu = 0.9
    (inc,) = ir.evaluate()
    assert inc.name == "cache_stampede_or_leak" and cfg.cache_key_scope == "tenant_user"


def test_span_processor_feeds_the_windows():
    clock = Clock()
    m = WindowedMetrics(clock=clock)
    setup_tracing(m)
    ask(build(FakeRag(HybridRetrieveOut(chunks=[], top_score=0.1, abstain=True, stage_latency_ms={}))))
    p95, n = m.p95("agent.query", 30)
    assert n == 1 and p95 is not None and [r.pool for r in m.recent("mcp.client", 30) if r.pool] == ["rag", "db"]


def test_thresholds_are_the_blueprint_values():
    t = Thresholds()
    assert (t.latency_p95_ms, t.latency_window_s, t.faithfulness_floor, t.hang_ms, t.hang_rate, t.redis_cpu) == \
        (550, 30, 0.90, 150, 0.01, 0.85)
