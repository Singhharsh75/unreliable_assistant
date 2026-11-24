"""Gate 17 — tenant-scoped answer cache, cache isolation probe, vault subgraph."""
from uuid import uuid4

import pytest

from context import cache as cache_mod
from context.cache import AnswerCache
from contracts.models import GraphSubgraphIn
from eval import kpis

DONE = {"tenant_id": "tenant_a", "route": "HYBRID_RAG", "status": "DONE", "answer": "x", "citations": [],
        "abstained": False, "faithfulness": 1.0}


def test_entries_are_per_tenant_and_expire():
    now = [0.0]
    c = AnswerCache(clock=lambda: now[0])
    c.put("tenant_a", "What is the SLA?", DONE)
    assert c.get("tenant_a", "  what is the SLA ") is not None  # normalized
    assert c.get("tenant_b", "What is the SLA?") is None
    now[0] = 400  # past 300 s + 45 s jitter
    assert c.get("tenant_a", "What is the SLA?") is None


@pytest.mark.parametrize("change", [{"route": "TRANSACTION_EXEC"}, {"status": "HALTED"}, {"answer": None}])
def test_only_finished_rag_answers_are_cached(change):
    c = AnswerCache()
    c.put("tenant_a", "q", {**DONE, **change})
    assert c.get("tenant_a", "q") is None


def test_queries_with_redacted_pii_are_never_cached():
    c = AnswerCache()
    c.put("tenant_a", "status for <EMAIL_1>?", DONE)
    assert c.get("tenant_a", "status for <EMAIL_1>?") is None


def test_egress_recheck_drops_another_tenants_entry():
    c = AnswerCache()
    c.put("tenant_a", "q", DONE)
    c._d[cache_mod.key("tenant_b", "q")] = c._d[cache_mod.key("tenant_a", "q")]  # a corrupted / mis-keyed entry
    assert c.get("tenant_b", "q") is None and c.leaks_blocked == 1
    assert kpis.cache_isolation_kpi().status == "PASS"


def test_isolation_probe_detects_a_leaking_cache(monkeypatch):
    def leaky_get(self, tenant_id, query, **_):  # no tenant in the key, no egress check
        entry = next((v for k, v in self._d.items() if k.endswith(cache_mod.key("", query)[-64:])), None)
        self.hits += entry is not None
        return dict(entry[1]) if entry else None

    monkeypatch.setattr(cache_mod, "key", lambda tenant_id, query, *_: cache_mod.normalize(query)[-64:].rjust(64))
    monkeypatch.setattr(AnswerCache, "get", leaky_get)
    r = kpis.cache_isolation_kpi()
    assert r.status == "FAIL" and r.value < 1.0


def test_subgraph_hides_other_tenants_notes():
    from mcp_servers.graphify import GRAPH, GraphVault

    if not GRAPH.exists():
        pytest.skip("run `uv run mcp_servers/graphify.py build`")
    v = GraphVault()

    def sub(tenant):
        ctx = {"tenant_id": tenant, "user_id": "user_1", "session_id": str(uuid4()),
               "idempotency_key": str(uuid4()), "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
        return v.subgraph(GraphSubgraphIn.model_validate({"ctx": ctx, "anchor_note_ids": ["acme-sla"]}, strict=False))

    assert {n["id"] for n in sub("tenant_acme").nodes} >= {"acme-sla"}
    assert sub("tenant_demo").nodes == []
    out = sub("tenant_acme")
    assert out.token_count <= 320 and out.raw_token_estimate >= out.token_count


def test_code_layer_maps_modules_imports_and_schemas():
    from mcp_servers.graphify import code_graph

    g = code_graph()
    ids = {n["id"] for n in g["nodes"]}
    assert {"triage.orchestrator", "contracts.models:CallContext"} <= ids
    assert ["triage.orchestrator", "imports", "contracts.models"] in [list(e) for e in g["edges"]]


def test_aggressive_lookup_matches_rewordings_but_never_opposites():
    c = AnswerCache()
    c.put("tenant_a", "Can a lot be released with QA sign-off?", DONE)
    c.put("tenant_a", "What temperature are cold-chain SKUs stored at?", DONE)
    reworded = "At what temperature are the cold-chain SKUs stored?"
    assert c.get("tenant_a", reworded) is None and c.get("tenant_a", reworded, aggressive=True) is not None
    for other in ("Can a lot be released without QA sign-off?", "What temperature are ambient SKUs stored at?",
                  "Can a lot not be released with QA sign-off?"):  # bge cosine up to 0.94 against the cached question
        assert c.get("tenant_a", other, aggressive=True) is None
    assert c.get("tenant_b", reworded, aggressive=True) is None


def test_user_scope_and_ttl_knobs():
    now = [0.0]
    c = AnswerCache(clock=lambda: now[0])
    c.put("tenant_a", "q", DONE, user_id="user_1", ttl_s=10, jitter_s=0)
    assert c.get("tenant_a", "q", user_id="user_1") and not c.get("tenant_a", "q", user_id="user_2")
    assert not c.get("tenant_a", "q")
    now[0] = 11
    assert not c.get("tenant_a", "q", user_id="user_1")


def test_the_retrieval_flow_puts_the_vault_subgraph_in_the_prompt():
    import asyncio

    from contracts.models import GraphSubgraphOut, HybridRetrieveIn, HybridRetrieveOut
    from harness import ToolExecutionHarness, ToolSpec
    from tests.test_orchestrator import COLD, REQ, TP, EventIn, chunk
    from triage import Orchestrator

    seen = {}

    class Gen:
        async def generate(self, query, chunks, graph=""):
            seen["graph"] = graph
            return "Excursions are logged as deviations."

    node = {"id": "sop-dev", "domain": "quality", "risk_tier": 2, "version": "1", "summary": "Deviations are logged."}
    h = ToolExecutionHarness([
        ToolSpec("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ", lambda m: HybridRetrieveOut(
            chunks=[chunk(COLD)], top_score=0.9, abstain=False, stage_latency_ms={})),
        ToolSpec("graph_subgraph", GraphSubgraphIn, "TIER_1_READ", lambda m: GraphSubgraphOut(
            nodes=[node], edges=[], token_count=9, raw_token_estimate=90)),
        ToolSpec("append_audit", EventIn, "TIER_1_READ", lambda m: {"seq": 1})])
    asyncio.run(Orchestrator(h, Gen(), None).handle({**REQ, "query": "How are cold-chain excursions logged?"},
                                                    traceparent=TP, caller_tier="TIER_1_READ"))
    assert "sop-dev" in seen["graph"]
