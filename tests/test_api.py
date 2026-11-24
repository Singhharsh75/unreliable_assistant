"""Gate 15 — public API: tenant header check, JSON and verified-segment streaming endpoints."""
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api import main
from tests.test_orchestrator import COLD, World, chunk, out
from triage import Orchestrator

GOOD = "Cold-chain SKUs must be stored between 2 and 8 degrees Celsius."
BAD = "The plant manager signs every release form personally."


class StreamGen:
    def __init__(self, *parts):
        self.parts = parts

    async def stream(self, query, chunks):
        for p in self.parts:
            yield p

    async def generate(self, query, chunks):
        return "".join(self.parts)


def client(gen):
    main.app.state.orchestrator = Orchestrator(World(out(chunk(COLD))).harness(), gen, None)
    return TestClient(main.app)


def body(**kw):
    return {"tenant_id": "tenant_acme", "user_id": "user_1", "session_id": str(uuid4()),
            "query": "What temperature are cold-chain SKUs stored at?", **kw}


def events(resp):
    return [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:])) for b in resp.text.strip().split("\n\n")]


@pytest.fixture(autouse=True)
def _reset():
    yield
    if hasattr(main.app.state, "orchestrator"):
        del main.app.state.orchestrator


def test_tenant_header_must_match_body():
    with client(StreamGen(GOOD)) as c:
        assert c.post("/api/v1/agent/query", json=body(), headers={"X-Tenant-ID": "tenant_other"}).status_code == 403
        assert c.post("/api/v1/agent/query", json=body(tenant_id="Acme!"),
                      headers={"X-Tenant-ID": "Acme!"}).status_code == 422


def test_json_endpoint_returns_the_public_body():
    with client(StreamGen(GOOD)) as c:
        r = c.post("/api/v1/agent/query", json=body(), headers={"X-Tenant-ID": "tenant_acme"}).json()
    assert r["route"] == "HYBRID_RAG" and r["answer"] == GOOD and r["status"] == "DONE"


def test_stream_releases_only_verified_segments_and_never_retracts():
    with client(StreamGen("Cold-chain SKUs must be stored ", "between 2 and 8 degrees Celsius. ", BAD)) as c:
        ev = events(c.post("/api/v1/agent/query/stream", json=body(), headers={"X-Tenant-ID": "tenant_acme"}))
        m = c.get("/metrics").json()
    assert ev == [("segment", {"text": GOOD}), ev[-1]] and ev[-1][0] == "final"
    assert ev[-1][1]["answer"] == GOOD  # the vetoed sentence never left the server
    assert m["llm_ttft_ms"]["n"] >= 1 and m["first_token_ms"]["n"] >= 1


def test_stream_veto_before_any_release_takes_the_quote_path():
    with client(StreamGen(BAD)) as c:
        ev = events(c.post("/api/v1/agent/query/stream", json=body(), headers={"X-Tenant-ID": "tenant_acme"}))
    assert [k for k, _ in ev] == ["final"] and ev[0][1]["answer"] in COLD
