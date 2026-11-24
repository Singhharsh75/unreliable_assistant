"""LLM planner (ADR-017): parsing fails closed, ctx never comes from the model, and the real loop drives it end to end."""
import asyncio
import json
from uuid import uuid4

from context.anchor import ReActPrompt, build_anchor
from harness import BoundedLoop, FinalAnswer, LoopConfig, State, ToolAction, ToolExecutionHarness
from mcp_servers.state_store import StateStore
from triage.llm_planner import INVALID_TOOL, LLMPlanner, parse_step, tool_catalog
from triage.orchestrator import _VerifiedPlanner
from triage.subagents import SUBAGENTS, ScopedHarness

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
BASE = {"tenant_id": "tenant_test", "user_id": "user_1", "session_id": str(uuid4()), "traceparent": TP}


def _models(store):
    return {s.name: s.input_model for s in store.specs(pool=None) if s.name in SUBAGENTS["execution"].allowed_tools}


def _planner(replies, store=None):
    store = store or StateStore()
    it = iter(replies)

    async def complete(system, user):
        return next(it)
    return LLMPlanner(complete, _models(store)), store


def _prompt(goal="status of LOT-00044"):
    return ReActPrompt(anchor=build_anchor(goal), turns=())


def test_catalog_hides_ctx_and_lists_tools():
    cat = tool_catalog(_models(StateStore()))
    assert "lookup_lot" in cat and "submit_transaction" in cat and "ctx" not in cat and "append_audit" not in cat


def test_parse_step_tolerates_fences_and_rejects_garbage():
    assert parse_step('```json\n{"action": "final", "text": "ok"}\n```') == {"action": "final", "text": "ok"}
    assert parse_step("no json here") is None and parse_step("{broken") is None and parse_step(None) is None
    assert parse_step("[1, 2]") is None


def test_tool_step_strips_model_supplied_ctx():
    p, _ = _planner([json.dumps({"thought": "t", "action": "tool", "tool": "lookup_lot",
                                 "args": {"lot_id": "LOT-00044", "ctx": {"tenant_id": "tenant_other"}}})])
    a = asyncio.run(p.reason(_prompt()))
    assert a == ToolAction("lookup_lot", {"lot_id": "LOT-00044"})


def test_malformed_or_unknown_steps_fail_closed():
    bad = ["nonsense", json.dumps({"action": "tool", "tool": "drop_tables", "args": {}}),
           json.dumps({"action": "tool", "tool": "append_audit", "args": {"event": {}}}),  # not an execution tool
           json.dumps({"action": "tool", "tool": "lookup_lot", "args": "LOT-00044"}),
           json.dumps({"action": "final", "text": "  "})]
    for reply in bad:
        p, _ = _planner([reply])
        assert asyncio.run(p.reason(_prompt())) == ToolAction(INVALID_TOOL, {}), reply


def _run(replies, goal, seed=True):
    store = StateStore()
    if seed:
        store.upsert_lot("tenant_test", "LOT-00044", "QUARANTINE", "SOP-DEV-01")
    p, _ = _planner(replies, store)
    harness = ScopedHarness(ToolExecutionHarness(store.specs(pool=None)), SUBAGENTS["execution"])
    loop = BoundedLoop(harness, _VerifiedPlanner(p), LoopConfig(llm_step_timeout_s=5, total_loop_budget_s=30, per_step_timeout_s=5))
    return asyncio.run(loop.run(goal, base_ctx=BASE, caller_tier="TIER_2_COMPUTE")), store


def test_loop_lookup_then_grounded_final_is_done():
    res, _ = _run([json.dumps({"thought": "look it up", "action": "tool", "tool": "lookup_lot", "args": {"lot_id": "LOT-00044"}}),
                   json.dumps({"thought": "answer", "action": "final", "text": "LOT-00044 has status QUARANTINE."})],
                  "What is the status of LOT-00044?")
    assert res.state is State.DONE and res.iterations == 2


def test_loop_vetoes_an_invented_fact():
    res, _ = _run([json.dumps({"action": "tool", "tool": "lookup_lot", "args": {"lot_id": "LOT-00044"}}),
                   json.dumps({"action": "final", "text": "LOT-00044 is RELEASED and approved by Dana."})],
                  "What is the status of LOT-00044?")
    assert res.state is State.HALTED and res.halt_reason == "verification_veto"


def test_loop_write_files_a_pending_approval_and_nothing_else():
    res, store = _run([json.dumps({"action": "tool", "tool": "submit_transaction",
                                   "args": {"action": "release", "payload": {"lot_id": "LOT-00044"}, "justification": "QA done"}}),
                       json.dumps({"action": "final", "text": "Release of LOT-00044 is waiting for approval."})],
                      "Release LOT-00044", seed=True)
    assert any(o["tool"] == "submit_transaction" and o["ok"] and o["result"]["status"] == "PENDING_APPROVAL" for o in res.observations)
    assert store._q("SELECT status FROM lots WHERE tenant_id = ? AND lot_id = ?", ("tenant_test", "LOT-00044")) == [("QUARANTINE",)]


def test_repeated_invalid_steps_halt_instead_of_looping():
    res, _ = _run(["nonsense"] * 5, "What is the status of LOT-00044?")
    assert res.state is State.HALTED and res.iterations <= 2


# --- observation_sources (ADR-017): system-written sources the model cannot supply -------------------------------

def test_sources_cover_failures_empty_results_and_filed_requests():
    from triage.verifier import observation_sources, verify
    lookup_fail = {"tool": "lookup_lot", "ok": False, "error": "INVALID_PARAMS", "detail": "unknown lot LOT-99999"}
    assert verify("LOT-99999 was not found.", observation_sources([lookup_fail])).grounded
    empty = {"tool": "get_session_facts", "ok": True, "result": {"facts": []}, "args": {}}
    assert verify("No facts are saved for this session.", observation_sources([empty])).grounded
    sub = {"tool": "submit_transaction", "ok": True, "result": {"approval_id": "apr_" + "a" * 32, "status": "PENDING_APPROVAL"},
           "args": {"action": "release", "payload": {"lot_id": "LOT-00044"}, "justification": "QA says all checks are done"}}
    src = dict(observation_sources([sub]))
    assert "release request was submitted" in src["submit_transaction"] and "QA says" not in " ".join(src.values())
    assert "no-submission" not in src


def test_nothing_submitted_is_stated_by_the_system_not_the_model():
    from triage.verifier import observation_sources
    ok = {"tool": "lookup_lot", "ok": True, "result": {"lot_id": "LOT-00044", "status": "QUARANTINE"}, "args": {"lot_id": "LOT-00044"}}
    assert "no-submission" in dict(observation_sources([ok]))
