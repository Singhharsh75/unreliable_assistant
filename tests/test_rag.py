import pytest

from mcp_servers.rag_server import INDEX, Retriever, _demo_req, bm25_tokens, rrf, windows


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


needs_index = pytest.mark.skipif(not (INDEX / "chunks.jsonl").exists(), reason="run: rag_server.py build")


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@needs_index
def test_grounded_query_hits_lot_protocol(retriever):
    out = retriever.retrieve(_demo_req("Verify batch LOT-88392 safety protocol"))
    assert not out.abstain and out.chunks[0].note_id == "sop-lot-release"
    assert set(out.stage_latency_ms) == {"bm25", "dense", "rrf", "rerank"}


@needs_index
def test_out_of_scope_query_abstains(retriever):
    assert retriever.retrieve(_demo_req("What is the capital of France?")).abstain


@needs_index
def test_tenant_private_note_is_filtered(retriever):
    q = "quality escalation contacts on-call QA lead"
    other = retriever.retrieve(_demo_req(q, tenant="tenant_other"))
    acme = retriever.retrieve(_demo_req(q, tenant="tenant_acme"))
    assert all(c.note_id != "acme-escalation" for c in other.chunks)
    assert any(c.note_id == "acme-escalation" for c in acme.chunks)
