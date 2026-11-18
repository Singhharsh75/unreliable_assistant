import asyncio
import json

import pytest
from mcp import Client, MCPError

from mcp_servers.rag_server import VAULT, Retriever, _demo_req, bm25_tokens, build, build_server, read_chunk, rrf, windows


def test_rrf_fuses_and_dedups():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60, pool=40)
    ids = [i for i, _ in fused]
    assert ids[:2] == [1, 3]  # in both lists, best combined ranks
    assert sorted(ids) == [1, 2, 3, 4]  # each chunk once
    assert rrf([[1, 2, 3]], k=60, pool=2) == rrf([[1, 2]], k=60, pool=2)


def test_windows_cover_all_tokens_with_overlap():
    assert windows(100, 510, 64) == [(0, 100)]
    w = windows(1200, 510, 64)
    assert w[0] == (0, 510) and w[-1][1] == 1200
    assert all(b[0] == a[1] - 64 for a, b in zip(w, w[1:]))


def test_bm25_tokens_keep_ids_whole():
    assert "lot-88392" in bm25_tokens("Verify LOT-88392 safety")
    assert bm25_tokens("Deviations handled") == ["deviat", "handl"]


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("rag_index")  # built from the vault on every run, so tests never skip
    build(VAULT, d)
    return d


@pytest.fixture(scope="module")
def retriever(index_dir):
    return Retriever(index_dir)


def test_grounded_query_hits_lot_protocol(retriever):
    out = retriever.retrieve(_demo_req("Verify batch LOT-88392 safety protocol"))
    assert not out.abstain and out.chunks[0].note_id == "sop-lot-release"
    assert set(out.stage_latency_ms) == {"bm25", "dense", "rrf", "rerank"}


def test_out_of_scope_query_abstains(retriever):
    assert retriever.retrieve(_demo_req("What is the capital of France?")).abstain


def test_tenant_private_note_is_filtered(retriever):
    q = "quality escalation contacts on-call QA lead"
    other = retriever.retrieve(_demo_req(q, tenant="tenant_other"))
    acme = retriever.retrieve(_demo_req(q, tenant="tenant_acme"))
    assert all(c.note_id != "acme-escalation" for c in other.chunks)
    assert any(c.note_id == "acme-escalation" for c in acme.chunks)


def test_private_chunks_are_not_crowded_out_by_top_k(retriever):
    # acme-escalation is the global BM25 #1 for this query. A post-top-k filter would leave tenant_other with nothing.
    q = "quality escalation contacts on-call QA lead"
    other = retriever._allowed("tenant_other", None)
    bm25 = retriever._bm25(q, 1, other)
    assert bm25 and retriever.chunks[bm25[0][0]]["note_id"] != "acme-escalation"
    dense = retriever._dense(q, 1, other)
    assert dense and all(other[i] for i, _ in dense)


def test_domain_filter_is_applied_before_ranking(retriever):
    q = "quality escalation contacts on-call QA lead"
    out = retriever.retrieve(_demo_req(q, tenant="tenant_acme").model_copy(update={"domain": "sku"}))
    assert all(c.note_id != "acme-escalation" for c in out.chunks)


def test_read_chunk_enforces_tenant_acl(retriever):
    by_hash = {c["chunk_hash"]: c for c in retriever.chunks}
    private = next(c for c in retriever.chunks if c["tenants"])
    shared = next(c for c in retriever.chunks if c["tenants"] is None)
    assert json.loads(read_chunk(by_hash, shared["chunk_hash"], tenant_id=None))["note_id"] == shared["note_id"]
    assert json.loads(read_chunk(by_hash, private["chunk_hash"], tenant_id=private["tenants"][0]))["note_id"] == private["note_id"]
    with pytest.raises(MCPError) as denied:
        read_chunk(by_hash, private["chunk_hash"], tenant_id="tenant_other")
    assert denied.value.code == -32005


def _call(retriever, method, *args):
    async def go():
        async with Client(build_server(retriever)) as client:
            return await getattr(client, method)(*args)

    return asyncio.run(go())


def _mcp_error_code(retriever, method, *args) -> int:
    # the in-memory client wraps a server JSON-RPC error in a TaskGroup ExceptionGroup; unwrap to the MCPError leaf
    try:
        _call(retriever, method, *args)
    except BaseException as exc:
        leaves = [exc]
        while leaves and getattr(leaves[0], "exceptions", None):
            leaves = list(leaves[0].exceptions)
        errors = [e for e in leaves if isinstance(e, MCPError)]
        if errors:
            return errors[0].code
        raise
    raise AssertionError("expected an MCP error")


def test_mcp_hybrid_retrieve_round_trip(retriever):
    req = json.loads(_demo_req("Verify batch LOT-88392 safety protocol").model_dump_json())
    result = _call(retriever, "call_tool", "hybrid_retrieve", {"req": req})
    body = json.loads(result.content[0].text)
    assert body["abstain"] is False and body["chunks"][0]["note_id"] == "sop-lot-release"


def test_mcp_schema_failure_maps_to_code_32001(retriever):
    req = json.loads(_demo_req("LOT-88392").model_dump_json())
    req["query"] = ""  # min_length=1
    assert _mcp_error_code(retriever, "call_tool", "hybrid_retrieve", {"req": req}) == -32001


def test_mcp_resource_read_enforces_acl(retriever):
    private = next(c for c in retriever.chunks if c["tenants"])
    shared = next(c for c in retriever.chunks if c["tenants"] is None)
    body = _call(retriever, "read_resource", f"rag://chunks/{shared['chunk_hash']}")
    assert json.loads(body.contents[0].text)["note_id"] == shared["note_id"]
    assert _mcp_error_code(retriever, "read_resource", f"rag://chunks/{private['chunk_hash']}") == -32005
