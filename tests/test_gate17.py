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
    def leaky_get(self, tenant_id, query):  # no tenant in the key, no egress check
        entry = next((v for k, v in self._d.items() if k.endswith(cache_mod.key("", query)[-64:])), None)
        self.hits += entry is not None
        return dict(entry[1]) if entry else None

    monkeypatch.setattr(cache_mod, "key", lambda tenant_id, query: cache_mod.normalize(query)[-64:].rjust(64))
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
