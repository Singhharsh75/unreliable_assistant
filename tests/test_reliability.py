"""Gate 9 — enterprise reliability patterns: breaker remediation, bulkheads, abstention short-circuit, HITL gate."""
import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from contracts.models import ABSTAIN_TEXT, CallContext, HybridRetrieveIn, HybridRetrieveOut
from harness import Bulkhead, FinalAnswer, ToolAction, ToolExecutionHarness, ToolSpec
from harness.errors import CIRCUIT_OPEN, RBAC_DENIED, TENANT_SCOPE_VIOLATION, UPSTREAM_TIMEOUT, HarnessError
from mcp_servers.state_store import StateStore
from triage import Orchestrator

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


def req(q, tenant="tenant_acme", user="user_op"):
    return {"tenant_id": tenant, "user_id": user, "session_id": str(uuid4()), "query": q}


def ctx(tenant="tenant_acme", user="user_op"):
    return {"tenant_id": tenant, "user_id": user, "session_id": str(uuid4()), "idempotency_key": str(uuid4()),
            "traceparent": TP}


class Script:
    def __init__(self, *actions):
        self.actions = list(actions)

    async def reason(self, prompt):
        return self.actions.pop(0)

    async def verify(self, answer, observations):
        return True


class NoGen:
    n = 0

    async def generate(self, query, chunks):
        NoGen.n += 1
        return "x"


def rag_spec(out, pool=None):
    return ToolSpec("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ", lambda m: out, pool=pool)


def orch(store, planner=None, extra=(), bulkheads=None):
    h = ToolExecutionHarness([*store.specs(), *extra], bulkheads=bulkheads or {})
    return Orchestrator(h, NoGen(), planner)


def handle(o, r, tier="TIER_2_COMPUTE"):
    return asyncio.run(o.handle(r, traceparent=TP, caller_tier=tier))


# --- 1. circuit breaker with a structured remediation prompt -------------------------

def test_breaker_halts_with_structured_remediation():
    s = StateStore()  # LOT-00044 does not exist: the same lookup fails twice
    o = orch(s, Script(*[ToolAction("lookup_lot", {"lot_id": "LOT-00044"})] * 3))
    r = handle(o, req("What is the status of LOT-00044?"))
    assert r["status"] == "HALTED_BREAKER"
    assert "check the IDs" in r["answer"] and "LOT-00044" in r["answer"]


def test_open_circuit_halts_immediately_without_calling_the_tool():
    s, db = StateStore(), Bulkhead("db", 4)
    db.trip()
    o = orch(s, Script(ToolAction("lookup_lot", {"lot_id": "LOT-00044"})), bulkheads={"db": db})
    r = handle(o, req("What is the status of LOT-00044?"))
    # the lookup and the audit write are both rejected: an open db pool loses the audit event (FMEA #6 WAL not built)
    assert r["status"] == "HALTED_BREAKER" and "temporarily unavailable" in r["answer"] and db.rejected == 2


# --- 2. bulkheads ---------------------------------------------------------------------------

class In(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext


def test_saturated_rag_pool_fails_fast_and_never_starves_db():
    release = threading.Event()

    def slow(m):
        release.wait(2)
        return {"ok": "rag"}

    rag, db = Bulkhead("rag", 2, acquire_timeout_s=0.02), Bulkhead("db", 2)
    h = ToolExecutionHarness([ToolSpec("slow_rag", In, "TIER_1_READ", slow, pool="rag"),
                              ToolSpec("db_read", In, "TIER_1_READ", lambda m: {"ok": "db"}, pool="db")],
                             bulkheads={"rag": rag, "db": db})

    async def go():
        stuck = [asyncio.create_task(h.call("slow_rag", {"ctx": ctx()}, caller_tier="TIER_1_READ")) for _ in range(2)]
        await asyncio.sleep(0.05)
        with pytest.raises(HarnessError) as e:
            await h.call("slow_rag", {"ctx": ctx()}, caller_tier="TIER_1_READ")
        t = time.perf_counter()
        db_value = (await h.call("db_read", {"ctx": ctx()}, caller_tier="TIER_1_READ")).value
        db_ms = (time.perf_counter() - t) * 1000
        release.set()
        await asyncio.gather(*stuck)
        return e.value.code, db_value, db_ms

    code, db_value, db_ms = asyncio.run(go())
    assert code == UPSTREAM_TIMEOUT and db_value == {"ok": "db"} and db_ms < 50


def test_bulkhead_trips_after_consecutive_failures_and_resets():
    b = Bulkhead("rag", 4, trip_after=2)

    def boom():
        raise RuntimeError

    async def go():
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await b.run(boom)
        with pytest.raises(HarnessError) as e:
            await b.run(lambda: 1)
        b.reset()
        return e.value.code, await b.run(lambda: 1)

    assert asyncio.run(go()) == (CIRCUIT_OPEN, 1)


def test_unknown_pool_is_a_config_error():
    with pytest.raises(ValueError):
        ToolExecutionHarness([ToolSpec("x", In, "TIER_1_READ", lambda m: 1, pool="nope")], bulkheads={"db": Bulkhead("db", 1)})


# --- 3. deterministic abstention short-circuit ------------------------------------------------

def test_abstention_short_circuit_under_90ms_and_skips_llm():
    out = HybridRetrieveOut(chunks=[], top_score=0.1, abstain=True, stage_latency_ms={})
    o, before = orch(StateStore(), extra=[rag_spec(out)]), NoGen.n
    handle(o, req("What is the cold-chain policy?"))  # warm-up
    times = []
    for _ in range(20):
        t = time.perf_counter()
        r = handle(o, req("What is the cold-chain policy?"))
        times.append((time.perf_counter() - t) * 1000)
    assert r["abstained"] and r["answer"] == ABSTAIN_TEXT and NoGen.n == before
    assert sorted(times)[18] < 90  # p95


# --- 4. asynchronous HITL gate -----------------------------------------------------------------

def submit_release(store, lot="LOT-88392", user="user_op"):
    o = orch(store, Script(ToolAction("submit_transaction", {"action": "release", "payload": {"lot_id": lot},
                                                             "justification": "QA review complete"}),
                           FinalAnswer("Release request is PENDING_APPROVAL.")))
    return handle(o, req(f"Release {lot}", user=user))


def test_tier3_writes_pending_approval_and_pauses():
    s = StateStore()
    r = submit_release(s)
    assert r["status"] == "PENDING_APPROVAL" and r["approval_id"].startswith("apr_")
    a = s.approval("tenant_acme", r["approval_id"])
    assert a["status"] == "PENDING_APPROVAL" and a["payload"] == {"lot_id": "LOT-88392"}
    assert [p["approval_id"] for p in s.pending("tenant_acme")] == [r["approval_id"]]


def test_reviewer_rules_four_eyes_role_and_finality():
    s = StateStore()
    aid = submit_release(s)["approval_id"]
    for kwargs in ({"reviewer_id": "user_op", "reviewer_role": "reviewer"},   # requester
                   {"reviewer_id": "user_rev", "reviewer_role": "operator"}):  # not a reviewer
        with pytest.raises(HarnessError) as e:
            s.decide("tenant_acme", aid, decision="APPROVED", **kwargs)
        assert e.value.code == RBAC_DENIED
    a = s.decide("tenant_acme", aid, reviewer_id="user_rev", reviewer_role="reviewer", decision="APPROVED")
    assert a["status"] == "APPROVED" and a["reviewer_id"] == "user_rev"
    with pytest.raises(HarnessError):
        s.decide("tenant_acme", aid, reviewer_id="user_rev2", reviewer_role="reviewer", decision="REJECTED")
    assert s.verify_chain("tenant_acme")
    types = [e["type"] for e in s.audit_events("tenant_acme")]
    assert types.index("hitl.requested") < types.index("hitl.decided")


def test_session_resumes_through_get_approval():
    s = StateStore()
    aid = submit_release(s)["approval_id"]
    s.decide("tenant_acme", aid, reviewer_id="user_rev", reviewer_role="reviewer", decision="REJECTED")
    h = ToolExecutionHarness(s.specs())
    res = asyncio.run(h.call("get_approval", {"ctx": ctx(), "approval_id": aid}, caller_tier="TIER_1_READ"))
    assert res.value["status"] == "REJECTED"


def test_stale_request_expires_to_rejected_never_approved():
    now = [datetime(2026, 11, 24, tzinfo=UTC)]
    s = StateStore(clock=lambda: now[0])
    aid = submit_release(s)["approval_id"]
    now[0] += timedelta(hours=25)
    assert s.approval("tenant_acme", aid)["status"] == "REJECTED"
    with pytest.raises(HarnessError):
        s.decide("tenant_acme", aid, reviewer_id="user_rev", reviewer_role="reviewer", decision="APPROVED")


def test_cross_tenant_reads_are_hard_failures_and_audited():
    s = StateStore()
    aid = submit_release(s)["approval_id"]
    s.upsert_lot("tenant_acme", "LOT-00044", "RELEASED")
    h = ToolExecutionHarness(s.specs())
    for tool, args in (("get_approval", {"approval_id": aid}), ("lookup_lot", {"lot_id": "LOT-00044"})):
        with pytest.raises(HarnessError) as e:
            asyncio.run(h.call(tool, {"ctx": ctx("tenant_globex"), **args}, caller_tier="TIER_1_READ"))
        assert e.value.code == TENANT_SCOPE_VIOLATION
    assert [e["type"] for e in s.audit_events("tenant_globex")] == ["security.tenant_scope_violation"] * 2


def test_tier3_cannot_be_reached_by_a_tier1_caller():
    s = StateStore()
    h = ToolExecutionHarness(s.specs())
    with pytest.raises(HarnessError) as e:
        asyncio.run(h.call("submit_transaction", {"ctx": ctx(), "action": "release", "payload": {"lot_id": "LOT-00044"}, "justification": "x"},
                           caller_tier="TIER_1_READ"))
    assert e.value.code == RBAC_DENIED and s.pending("tenant_acme") == []


def test_tampered_audit_chain_is_detected():
    s = StateStore()
    submit_release(s)
    s._db.execute("UPDATE audit SET event = ? WHERE seq = 1", (json.dumps({"type": "forged"}),))
    assert not s.verify_chain("tenant_acme")


# --- final review regressions ----------------------------------------------------------------

def test_client_errors_do_not_trip_the_pool():
    b = Bulkhead("db", 4, trip_after=2)

    def unknown_lot():
        raise HarnessError(-32602, "unknown lot")

    async def go():
        for _ in range(5):
            with pytest.raises(HarnessError):
                await b.run(unknown_lot)
    asyncio.run(go())
    assert not b.is_open and b.failures == 0


def test_hung_thread_keeps_its_slot_after_the_caller_times_out():
    release = threading.Event()
    b = Bulkhead("rag", 1, acquire_timeout_s=0.02)

    async def go():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(b.run(lambda: release.wait(2)), 0.02)
        with pytest.raises(HarnessError) as e:  # the thread is still busy, so the pool stays full
            await b.run(lambda: 1)
        release.set()
        await asyncio.sleep(0.05)
        return e.value.code, await b.run(lambda: 1)

    assert asyncio.run(go()) == (UPSTREAM_TIMEOUT, 1)


def test_state_store_runs_on_db_pool_threads_not_the_event_loop():
    s, seen = StateStore(), []
    s.upsert_lot("tenant_acme", "LOT-00044", "RELEASED")
    orig = s.lookup_lot

    def spy(m):
        seen.append(threading.current_thread().name)
        return orig(m)

    s.lookup_lot = spy
    h = ToolExecutionHarness(s.specs(), bulkheads={"db": Bulkhead("db", 2)})
    asyncio.run(h.call("lookup_lot", {"ctx": ctx(), "lot_id": "LOT-00044"}, caller_tier="TIER_1_READ"))
    assert seen and seen[0].startswith("bulkhead-db")


def test_sync_handler_in_a_pool_keeps_the_trace_context():
    from opentelemetry import trace

    from observability.tracing import setup_tracing, span

    setup_tracing()  # a real SDK provider, so spans have ids
    got = []

    def handler(m):
        got.append(trace.get_current_span().get_span_context().span_id)
        return 1

    h = ToolExecutionHarness([ToolSpec("t", In, "TIER_1_READ", handler, pool="rag")], bulkheads={"rag": Bulkhead("rag", 2)})

    async def go():
        with span("outer") as s:
            await h.call("t", {"ctx": ctx()}, caller_tier="TIER_1_READ")
            return s.get_span_context().span_id

    outer = asyncio.run(go())
    assert got and got[0] not in (0, outer)  # the thread sees the mcp.client span, a child of `outer`


# --- per-action payload schemas (ADR-018) --------------------------------------------------------

@pytest.mark.parametrize("action,payload,ok", [
    ("release", {"lot_id": "LOT-00044"}, True),
    ("release", {"lot_id": "lot 44"}, True),  # ID repair still applies inside the payload
    ("release", {}, False),
    ("release", {"lot_id": None}, False),
    ("release", {"lot_id": "LOT-00044", "status": "RELEASED"}, False),
    ("release", {"lot_id": "LOT-00044' OR '1'='1"}, False),
    ("open_deviation", {"lot_id": "LOT-00044", "description": "45 minutes at 11 °C"}, True),
    ("open_capa", {"lot_id": "LOT-00044"}, False),
    ("open_deviation", {"lot_id": "LOT-00044", "description": "x" * 501}, False),
    ("open_deviation", {"lot_id": "LOT-00044", "description": "ok\u202eevil"}, False),
    ("open_deviation", {"lot_id": "LOT-00044", "description": "a\x00b"}, False),
    ("update_status", {"lot_id": "LOT-00044", "status": "RELEASED"}, True),
    ("update_status", {"lot_id": "LOT-00044", "status": "SHIPPED"}, False),
])
def test_only_well_formed_payloads_become_pending_approvals(action, payload, ok):
    s = StateStore()
    h = ToolExecutionHarness(s.specs())
    call = h.call("submit_transaction", {"ctx": ctx(), "action": action, "payload": payload, "justification": "x"},
                  caller_tier="TIER_2_COMPUTE")
    if ok:
        asyncio.run(call)
        assert s.pending("tenant_acme")[0]["payload"]["lot_id"] == "LOT-00044"
    else:
        with pytest.raises(HarnessError):
            asyncio.run(call)
        assert s.pending("tenant_acme") == []


def test_restart_frees_a_hung_pool_and_ignores_the_late_release():
    release, b = threading.Event(), Bulkhead("rag", 1, acquire_timeout_s=0.02)

    async def go():
        stuck = asyncio.create_task(b.run(release.wait, 5))
        await asyncio.sleep(0.05)
        with pytest.raises(HarnessError):
            await b.run(lambda: 1)
        b.restart()
        assert await b.run(lambda: 2) == 2
        release.set()
        await stuck
    asyncio.run(go())
    assert b.in_flight == 0


def test_react_steps_reach_the_audit_log_but_not_the_response():
    s = StateStore()
    s.upsert_lot("tenant_acme", "LOT-00044", "HOLD")
    o = orch(s, Script(ToolAction("lookup_lot", {"lot_id": "LOT-00044"}, "look it up"),
                       FinalAnswer("LOT-00044 is RELEASED by Dana.", "answer")))
    r = handle(o, req("What is the status of LOT-00044?"))
    event = next(e for e in s.audit_events("tenant_acme") if e["type"] == "agent.query")
    assert r["status"] == "HALTED" and "steps" not in r
    look, final = event["steps"]
    assert (look["thought"], look["tool"]) == ("look it up", "lookup_lot") and "HOLD" in look["outcome"]
    assert final["outcome"] == "vetoed" and "Dana" not in json.dumps(event)  # the vetoed draft is not stored
