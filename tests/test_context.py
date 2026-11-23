"""Gate 5 — goal anchor, observation compression, sliding window + dual memory."""
import asyncio
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from context import SessionMemory, Turn, build_anchor, compress_observation, estimate_tokens, normalize_entities
from contracts.models import CallContext, SessionFact
from harness import BoundedLoop, FinalAnswer, ToolAction, ToolExecutionHarness, ToolSpec

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


def base(tenant="tenant_acme", session=None):
    return {"tenant_id": tenant, "user_id": "user_1", "session_id": session or str(uuid4()), "traceparent": TP}


# --- anchor --------------------------------------------------------------------

def test_entities_are_normalized_and_deduped():
    assert normalize_entities("Check lot 44 and LOT-00044, plus sku99") == ("LOT-00044", "SKU-099")


def test_anchor_is_frozen_and_first():
    a = build_anchor("verify lot 44", ("no mutation without HITL",))
    assert a.entity_ids == ("LOT-00044",)
    try:
        a.original_intent = "other"
    except Exception:
        pass
    else:
        raise AssertionError("anchor must be immutable")
    from context import ReActPrompt
    msgs = ReActPrompt(a, (Turn("t", "x" * 5),)).messages()
    assert msgs[0].startswith("GOAL ANCHOR") and "LOT-00044" in msgs[0] and "no mutation" in msgs[0]


# --- compression ---------------------------------------------------------------

def test_boilerplate_and_empty_values_are_stripped():
    out = compress_observation({"lot_id": "LOT-00044", "status": "RELEASED", "traceparent": TP, "_meta": {"a": 1},
                                "notes": "", "tags": []})
    assert out == '{"lot_id":"LOT-00044","status":"RELEASED"}'


def test_long_observation_truncated_to_180_tokens():
    out = compress_observation({"text": "word " * 2000})
    assert estimate_tokens(out) <= 180 and out.endswith("[truncated]")


def test_short_observation_is_not_truncated():
    assert "truncated" not in compress_observation({"a": 1})


# --- memory --------------------------------------------------------------------

def test_window_keeps_last_three_turns():
    m = SessionMemory()
    for i in range(5):
        m.append_turn("tenant_a", "s1", Turn("t", str(i)))
    assert [t.observation for t in m.window("tenant_a", "s1")] == ["2", "3", "4"]


def test_window_and_facts_are_tenant_scoped():
    m = SessionMemory()
    m.append_turn("tenant_a", "s1", Turn("t", "secret"))
    m.persist_verified("tenant_a", "s1", [SessionFact(key="lot_status", value="RELEASED")])
    assert m.window("tenant_b", "s1") == () and m.facts("tenant_b", "s1") == {}
    assert m.facts("tenant_a", "s1")["lot_status"].value == "RELEASED"


def test_latest_fact_per_key_wins():
    m = SessionMemory()
    m.persist_verified("tenant_a", "s1", [SessionFact(key="k", value="old"), SessionFact(key="k", value="new")])
    assert m.facts("tenant_a", "s1")["k"].value == "new"


# --- loop integration ----------------------------------------------------------

class LotIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    lot_id: str


async def noisy(model):
    return {"lot_id": model.lot_id, "status": "RELEASED", "traceparent": TP, "blob": "x" * 5000}


class Recorder:
    def __init__(self, actions, verdict=True):
        self.actions, self.prompts, self.verdict = list(actions), [], verdict

    async def reason(self, prompt):
        self.prompts.append(prompt)
        return self.actions.pop(0)

    async def verify(self, answer, observations):
        return self.verdict


def run(planner, memory, extractor=None, ctx=None):
    h = ToolExecutionHarness([ToolSpec("lookup_lot", LotIn, "TIER_1_READ", noisy)])
    loop = BoundedLoop(h, planner, memory=memory, fact_extractor=extractor)
    return asyncio.run(loop.run("verify lot 44", base_ctx=ctx or base(), caller_tier="TIER_1_READ"))


def test_planner_sees_anchor_and_compressed_observation_only():
    p = Recorder([ToolAction("lookup_lot", {"lot_id": "LOT-00044"}), FinalAnswer("released")])
    run(p, SessionMemory())
    second = p.prompts[1]
    assert second.anchor.entity_ids == ("LOT-00044",)
    assert len(second.turns) == 1 and estimate_tokens(second.turns[0].observation) <= 180
    assert "traceparent" not in second.turns[0].observation


def test_anchor_identical_on_every_pass():
    actions = [ToolAction("lookup_lot", {"lot_id": f"LOT-{i:05d}"}) for i in (1, 2, 3)] + [FinalAnswer("ok")]
    p = Recorder(actions)
    run(p, SessionMemory())
    assert len({pr.anchor for pr in p.prompts}) == 1 and len(p.prompts) == 4
    assert len(p.prompts[-1].turns) == 3  # window capped at 3


def test_facts_persist_only_when_verified():
    mem = SessionMemory()
    ctx = base()
    extract = lambda ans, obs: [SessionFact(key="lot_status", value="RELEASED")]
    run(Recorder([FinalAnswer("x")], verdict=False), mem, extract, ctx)
    assert mem.facts(ctx["tenant_id"], ctx["session_id"]) == {}
    run(Recorder([FinalAnswer("x")], verdict=True), mem, extract, ctx)
    assert mem.facts(ctx["tenant_id"], ctx["session_id"])["lot_status"].value == "RELEASED"
