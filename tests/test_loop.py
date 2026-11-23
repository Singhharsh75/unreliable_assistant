"""Gate 4 — bounded ReAct loop: state machine, ADR-004 limits, breaker, no-state-change guard."""
import asyncio
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from contracts.models import CallContext, LOT_RE
from harness import BoundedLoop, FinalAnswer, LoopConfig, State, ToolAction, ToolExecutionHarness, ToolSpec
from harness.errors import UPSTREAM_TIMEOUT

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
BASE = {"tenant_id": "tenant_acme", "user_id": "user_1", "session_id": str(uuid4()), "traceparent": TP}


class LotIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    lot_id: str


class Script:
    """Planner that replays a fixed list of actions; `verify` returns `verdict`."""

    def __init__(self, actions, verdict=True):
        self.actions = list(actions)
        self.verdict = verdict
        self.seen = []

    async def reason(self, prompt):
        self.seen.append(prompt)
        nxt = self.actions.pop(0)
        if callable(nxt):
            return await nxt()
        return nxt

    async def verify(self, answer, observations):
        return self.verdict


def lot_tool(handler=None, tier="TIER_1_READ"):
    async def ok(model):
        return {"lot": model.lot_id, "status": "RELEASED"}

    return ToolSpec("lookup_lot", LotIn, tier, handler or ok)


def run_loop(planner, tools=None, config=None, caller="TIER_1_READ", clock=None):
    harness = ToolExecutionHarness(tools or [lot_tool()])
    loop = BoundedLoop(harness, planner, config or LoopConfig(), **({"clock": clock} if clock else {}))
    return asyncio.run(loop.run("verify LOT-00044", base_ctx=BASE, caller_tier=caller))


def test_tool_then_answer_reaches_done():
    p = Script([ToolAction("lookup_lot", {"lot_id": "LOT-00044"}), FinalAnswer("LOT-00044 is released [chunk]")])
    res = run_loop(p)
    assert res.state is State.DONE and res.answer.startswith("LOT-00044")
    assert res.trace == [State.INIT, State.TRIAGE, State.REASONING, State.TOOL_CALL, State.OBSERVATION,
                         State.REASONING, State.VERIFY, State.DONE]
    assert res.observations[0]["ok"] and res.observations[0]["result"]["status"] == "RELEASED"


def test_verifier_veto_blocks_release():
    res = run_loop(Script([FinalAnswer("ungrounded claim")], verdict=False))
    assert res.state is State.HALTED and res.halt_reason == "verification_veto" and res.answer is None


def test_verifier_error_never_releases_draft():
    class Broken(Script):
        async def verify(self, answer, observations):
            raise RuntimeError("NLI down")

    res = run_loop(Broken([FinalAnswer("draft")]))
    assert res.state is State.HALTED and res.halt_reason == "verify_failed" and res.answer is None


def test_max_iterations_is_five():
    # Each call uses a different lot, so no observation repeats and only the iteration cap can stop the loop.
    actions = [ToolAction("lookup_lot", {"lot_id": f"LOT-{i:05d}"}) for i in range(1, 10)]
    res = run_loop(Script(actions))
    assert res.state is State.HALTED and res.halt_reason == "max_iterations reached"
    assert res.iterations == 5


def test_breaker_trips_after_two_identical_failures():
    spy_calls = []
    async def counting(model):
        spy_calls.append(1)
        raise RuntimeError("upstream 500")

    p = Script([ToolAction("lookup_lot", {"lot_id": "LOT-00044"})] * 3)
    res = run_loop(p, tools=[lot_tool(counting)])
    assert res.state is State.HALTED_BREAKER
    assert len(spy_calls) == 2  # the third identical attempt never reaches the tool
    assert [o["error"] for o in res.observations] == ["INTERNAL_ERROR", "INTERNAL_ERROR"]


def test_step_timeout_is_an_observation_that_counts_toward_breaker():
    async def slow(model):
        await asyncio.sleep(0.5)
        return {}

    cfg = LoopConfig(per_step_timeout_s=0.02)
    p = Script([ToolAction("lookup_lot", {"lot_id": "LOT-00044"})] * 2)
    res = run_loop(p, tools=[lot_tool(slow)], config=cfg)
    assert res.state is State.HALTED_BREAKER
    assert all(o["error"] == "UPSTREAM_TIMEOUT" for o in res.observations)
    assert UPSTREAM_TIMEOUT == -32008


def test_rbac_denial_halts_immediately_and_is_not_retried():
    tier2 = lot_tool(tier="TIER_2_COMPUTE")
    p = Script([ToolAction("lookup_lot", {"lot_id": "LOT-00044"})] * 3)
    res = run_loop(p, tools=[tier2], caller="TIER_1_READ")
    assert res.state is State.HALTED and "RBAC_DENIED" in res.halt_reason
    assert res.observations == [] and res.iterations == 1


def test_repeated_identical_success_is_no_state_change():
    p = Script([ToolAction("lookup_lot", {"lot_id": "LOT-00044"})] * 3)
    res = run_loop(p)
    assert res.state is State.HALTED and res.halt_reason == "no state change since last observation"
    assert len(res.observations) == 2


def test_total_loop_budget_stops_further_passes():
    now = {"t": 0.0}

    class Ticking(Script):
        async def reason(self, prompt):
            now["t"] += 0.2  # each reasoning pass costs 200 ms of the 450 ms budget
            return await super().reason(prompt)

    actions = [ToolAction("lookup_lot", {"lot_id": f"LOT-{i:05d}"}) for i in range(1, 6)]
    res = run_loop(Ticking(actions), clock=lambda: now["t"])
    assert res.state is State.HALTED and res.halt_reason == "total_loop_budget exceeded"
    assert res.iterations < 5


def test_llm_step_timeout_halts():
    class Slow(Script):
        async def reason(self, prompt):
            await asyncio.sleep(0.5)
            return FinalAnswer("late")

    res = run_loop(Slow([]), config=LoopConfig(llm_step_timeout_s=0.02))
    assert res.state is State.HALTED and res.halt_reason == "llm_step_timeout"


def test_each_tool_call_gets_a_fresh_uuidv4_key():
    keys = []

    async def record(model):
        keys.append(model.ctx.idempotency_key)
        return {"ok": True}

    p = Script([ToolAction("lookup_lot", {"lot_id": "LOT-00001"}), ToolAction("lookup_lot", {"lot_id": "LOT-00002"})])
    run_loop(p, tools=[lot_tool(record)])
    assert len(keys) == 2 and keys[0] != keys[1]
    assert all(k.version == 4 for k in keys)
