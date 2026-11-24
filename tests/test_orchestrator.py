"""Gate 7 — subagent topology end to end: Verification Subagent veto, route dispatch, audit, HITL status."""
import asyncio
import hashlib
import os
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from contracts.models import ABSTAIN_TEXT, CallContext, Chunk, HybridRetrieveIn, HybridRetrieveOut
from harness import FinalAnswer, ToolAction, ToolExecutionHarness, ToolSpec
from triage import Orchestrator, extractive_quote, verify

TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
REQ = {"tenant_id": "tenant_acme", "user_id": "user_1", "session_id": str(uuid4())}
COLD = "Cold-chain SKUs must be stored between 2 and 8 degrees Celsius. Excursions are logged as deviations."


def chunk(text: str) -> Chunk:
    h = hashlib.sha256(text.encode()).hexdigest()
    return Chunk(chunk_hash=h, uri=f"rag://chunks/{h}", text=text, note_id="n", bm25_score=1.0,
                 dense_cosine=0.9, rrf_rank=1, rerank_score=0.9)


# --- verifier -------------------------------------------------------------------

def test_grounded_claims_pass_and_cite_their_chunk():
    v = verify("Cold-chain SKUs are stored between 2 and 8 degrees Celsius [1].", [("c1", COLD), ("c2", "unrelated")])
    assert v.grounded and v.faithfulness == 1.0 and v.sources == ["c1"]


def test_wrong_number_is_vetoed_even_with_full_word_overlap():
    v = verify("Cold-chain SKUs must be stored between 2 and 18 degrees Celsius.", [("c1", COLD)])
    assert not v.grounded


def test_unsupported_sentence_vetoes_the_whole_draft():
    v = verify("Excursions are logged as deviations. The plant manager signs every release form personally.", [("c1", COLD)])
    assert not v.grounded and v.faithfulness == 0.5


def test_empty_draft_is_not_grounded():
    assert not verify("", [("c1", COLD)]).grounded


def test_extractive_quote_is_verbatim_and_bounded():
    q = extractive_quote(COLD, max_chars=80)
    assert q in COLD and len(q) <= 80 and verify(q, [("c1", COLD)]).grounded


# --- orchestrator ----------------------------------------------------------------

class EventIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    event: dict[str, Any]


class LotIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    lot_id: str


class SubmitIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    action: str
    payload: dict[str, Any]
    justification: str


class World:
    def __init__(self, retrieval: HybridRetrieveOut):
        self.retrieval, self.calls, self.audit = retrieval, [], []

    def harness(self) -> ToolExecutionHarness:
        def rag(m):
            self.calls.append("hybrid_retrieve")
            return self.retrieval

        def audit(m):
            self.audit.append(m.event)
            return {"seq": len(self.audit)}

        def lot(m):
            self.calls.append("lookup_lot")
            return {"lot_id": m.lot_id, "status": "RELEASED"}

        def submit(m):
            self.calls.append("submit_transaction")
            return {"approval_id": "apr_1", "status": "PENDING_APPROVAL"}

        return ToolExecutionHarness([
            ToolSpec("hybrid_retrieve", HybridRetrieveIn, "TIER_1_READ", rag),
            ToolSpec("append_audit", EventIn, "TIER_1_READ", audit),
            ToolSpec("lookup_lot", LotIn, "TIER_1_READ", lot),
            ToolSpec("submit_transaction", SubmitIn, "TIER_3_MUTATE", submit, min_caller_tier="TIER_2_COMPUTE", hitl_gate=True),
        ])


class Gen:
    def __init__(self, text):
        self.text, self.n = text, 0

    async def generate(self, query, chunks):
        self.n += 1
        return self.text


class Script:
    def __init__(self, *actions):
        self.actions = list(actions)

    async def reason(self, prompt):
        return self.actions.pop(0)

    async def verify(self, answer, observations):  # replaced by the Verification Subagent
        raise AssertionError("inner verify must not be used")


def run(world, query, gen=None, planner=None, tier="TIER_2_COMPUTE"):
    o = Orchestrator(world.harness(), gen or Gen("unused"), planner)
    return asyncio.run(o.handle({**REQ, "query": query}, traceparent=TP, caller_tier=tier))


def out(*chunks, abstain=False):
    return HybridRetrieveOut(chunks=list(chunks), top_score=0.9, abstain=abstain, stage_latency_ms={})


def test_canned_routes_never_reach_retrieval_or_llm():
    w, g = World(out()), Gen("x")
    for q in ("hello", "What's the weather tomorrow?"):
        r = run(w, q, gen=g)
        assert r["route"] in {"DIRECT_ANSWER", "REJECT_OOS"} and r["answer"]
    assert w.calls == [] and g.n == 0 and len(w.audit) == 2


def test_abstention_skips_the_generator():
    w, g = World(out(abstain=True)), Gen("x")
    r = run(w, "What is the policy on cold-chain storage?", gen=g)
    assert r["abstained"] and r["answer"] == ABSTAIN_TEXT and g.n == 0


def test_grounded_draft_is_released_with_citations():
    c = chunk(COLD)
    r = run(World(out(c)), "What temperature are cold-chain SKUs stored at?",
            gen=Gen("Cold-chain SKUs are stored between 2 and 8 degrees Celsius."))
    assert r["answer"].startswith("Cold-chain SKUs are stored") and r["citations"] == [c.uri] and r["faithfulness"] == 1.0


def test_ungrounded_draft_is_vetoed_and_replaced_with_a_quote():
    c, w = chunk(COLD), None
    w = World(out(c))
    r = run(w, "What temperature are cold-chain SKUs stored at?", gen=Gen("Cold-chain SKUs are stored at 25 degrees."))
    assert r["answer"] in COLD and r["citations"] == [c.uri]
    assert w.audit[-1]["vetoed"] is True


def test_execution_lookup_is_verified_against_observations():
    w = World(out())
    p = Script(ToolAction("lookup_lot", {"lot_id": "LOT-00044"}), FinalAnswer("LOT-00044 status is RELEASED."))
    r = run(w, "What is the status of LOT-00044?", planner=p)
    assert r["route"] == "TRANSACTION_EXEC" and r["status"] == "DONE" and r["answer"] == "LOT-00044 status is RELEASED."


def test_execution_answer_not_in_observations_is_halted():
    w = World(out())
    p = Script(ToolAction("lookup_lot", {"lot_id": "LOT-00044"}), FinalAnswer("LOT-00044 is QUARANTINED pending CAPA."))
    r = run(w, "What is the status of LOT-00044?", planner=p)
    assert r["status"] == "HALTED" and "verification_veto" in r["answer"]


def test_mutation_returns_pending_approval():
    w = World(out())
    p = Script(ToolAction("submit_transaction", {"action": "release", "payload": {"lot_id": "LOT-88392"}, "justification": "QA ok"}),
               FinalAnswer("Release request apr_1 is PENDING_APPROVAL."))
    r = run(w, "Release LOT-88392", planner=p)
    assert r["status"] == "PENDING_APPROVAL" and r["approval_id"] == "apr_1"
    assert w.audit[-1]["approval_id"] == "apr_1"


def test_execution_subagent_cannot_retrieve():
    w = World(out(chunk(COLD)))
    p = Script(ToolAction("hybrid_retrieve", {"query": "x"}))
    r = run(w, "What is the status of LOT-00044?", planner=p)
    assert r["status"] == "HALTED" and "RBAC_DENIED" in r["answer"] and "hybrid_retrieve" not in w.calls


# --- review fixes -------------------------------------------------------------------

def test_numbered_list_markers_are_not_claims():
    v = verify("1. Cold-chain SKUs are stored between 2 and 8 degrees Celsius.\n2. Excursions are logged as deviations.",
               [("c1", COLD)])
    assert v.grounded and len(v.claims) == 2


def test_abstention_variant_is_not_turned_into_a_quote():
    r = run(World(out(chunk(COLD))), "What is the torque spec?", gen=Gen("The knowledge base does not cover this question [1]"))
    assert r["abstained"] and r["answer"] == ABSTAIN_TEXT and r["citations"] == []


def test_veto_audit_keeps_draft_faithfulness():
    w = World(out(chunk(COLD)))
    run(w, "What temperature are cold-chain SKUs stored at?", gen=Gen("Cold-chain SKUs are stored at 25 degrees."))
    assert w.audit[-1]["vetoed"] and w.audit[-1]["draft_faithfulness"] == 0.0


class Boom:
    async def generate(self, query, chunks):
        raise TimeoutError


def test_upstream_failure_halts_and_is_still_audited():
    w = World(out(chunk(COLD)))
    r = run(w, "What temperature are cold-chain SKUs stored at?", gen=Boom())
    assert r["status"] == "HALTED" and "TimeoutError" in r["answer"] and len(w.audit) == 1


def test_missing_planner_halts_instead_of_crashing():
    w = World(out())
    r = run(w, "What is the status of LOT-00044?")
    assert r["status"] == "HALTED" and len(w.audit) == 1


def test_claim_joining_two_observations_is_grounded():
    w = World(out())
    p = Script(ToolAction("lookup_lot", {"lot_id": "LOT-00044"}),
               ToolAction("submit_transaction", {"action": "release", "payload": {"lot_id": "LOT-00044"}, "justification": "ok"}),
               FinalAnswer("LOT-00044 is RELEASED and release request apr_1 is PENDING_APPROVAL."))
    r = run(w, "Release LOT-00044", planner=p)
    assert r["status"] == "PENDING_APPROVAL" and r["answer"].startswith("LOT-00044 is RELEASED")


# --- NLI verifier ------------------------------------------------------------------

def _fake_entail(pairs):  # entailed iff every hypothesis content word of "stored"/"kept" synonyms; here: negation-aware stub
    return [(0.0, 0.9) if " not " in h else (0.0, 0.0) if "personally" in h else (0.9, 0.0) for _, h in pairs]


def test_nli_accepts_paraphrase_the_lexical_check_vetoes():
    claim = "Refrigerated products have to be kept at 2-8 C."
    assert not verify(claim, [("c1", COLD)]).grounded
    assert verify("Cold-chain products are kept chilled at 2 to 8 degrees.", [("c1", COLD)], _fake_entail).grounded


def test_nli_keeps_exact_number_precondition():
    assert not verify("Cold-chain SKUs are stored between 2 and 18 degrees Celsius.", [("c1", COLD)], lambda p: [(1.0, 0.0)] * len(p)).grounded


def test_nli_contradiction_vetoes_a_lexical_pass():
    v = verify("Cold-chain SKUs are not stored between 2 and 8 degrees Celsius.", [("c1", COLD)], _fake_entail)
    assert not v.grounded


@pytest.mark.skipif(os.getenv("SKIP_NLI") == "1", reason="downloads the NLI model")
def test_real_nli_model_separates_entailment_from_contradiction():
    from triage.nli import entailment
    (ok, _), (bad, contra) = entailment([(COLD, "Cold-chain SKUs are kept between 2 and 8 degrees Celsius."),
                                         (COLD, "Cold-chain SKUs must never be refrigerated.")])
    assert ok > 0.5 > bad and contra > 0.5
