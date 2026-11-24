"""Request orchestration across the subagent topology (design.md §1.3, Gate 7).

triage -> canned reply (DIRECT_ANSWER, REJECT_OOS)
       -> Retrieval Subagent -> generator -> Verification Subagent (HYBRID_RAG)
       -> Execution Subagent: bounded ReAct loop, verified on DONE (TRANSACTION_EXEC)
Every route ends with one audit event written by the Verification Subagent.
Each subagent reaches tools only through its own ScopedHarness, so the binding in SUBAGENTS is enforced.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Protocol
from uuid import uuid4

from eval.ragas_auditor import RagasAuditor, Sample
from contracts.models import ABSTAIN_TEXT, Chunk, HybridRetrieveOut
from context.anchor import ReActPrompt
from context.memory import SessionMemory
from harness.bulkhead import Bulkhead
from harness.errors import HarnessError
from harness.loop import BoundedLoop, FinalAnswer, LoopConfig, Planner, State, ToolAction
from harness.tool_harness import Tier, ToolExecutionHarness
from observability.incidents import RuntimeConfig
from observability.tracing import current_traceparent, parent_context, span
from triage.classifier import TriageResult, classify
from triage.subagents import SUBAGENTS, ScopedHarness, canned_response
from triage.verifier import Entail, extractive_quote, verify, with_question

log = logging.getLogger("triage.orchestrator")
_ABSTAIN_KEY = ABSTAIN_TEXT.rstrip(".").lower()


class Generator(Protocol):
    async def generate(self, query: str, chunks: list[Chunk]) -> str | None: ...


class ClaudeGenerator:
    """Production generator: the same prompt, model and client as the Gate 3 eval (eval/run_ragas.py)."""

    def __init__(self) -> None:
        self._client = None

    async def generate(self, query: str, chunks: list[Chunk]) -> str | None:
        from eval.run_ragas import GENERATOR, SYSTEM, complete, make_client  # lazy: the eval module imports ragas

        self._client = self._client or make_client()
        ctx = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
        return await complete(self._client, GENERATOR, f"Context:\n{ctx}\n\nQuestion: {query}", system=SYSTEM)


class _VerifiedPlanner:
    """The loop's VERIFY step is the Verification Subagent's claim check against the tool observations."""

    def __init__(self, inner: Planner) -> None:
        self.inner = inner

    async def reason(self, prompt: ReActPrompt) -> ToolAction | FinalAnswer:
        return await self.inner.reason(prompt)

    async def verify(self, answer: str, observations: list[dict[str, Any]]) -> bool:
        sources = [(o["tool"], json.dumps(o["result"], default=str)) for o in observations if o.get("ok")]
        # One claim may join facts from two tool results (lot status + approval id), so also check against all of them.
        sources.append(("observations", " ".join(t for _, t in sources)))
        return verify(answer, sources).grounded


class Orchestrator:
    def __init__(
        self,
        harness: ToolExecutionHarness,
        generator: Generator,
        planner: Planner | None = None,
        *,
        loop_config: LoopConfig | None = None,
        memory: SessionMemory | None = None,
        auditor: RagasAuditor | None = None,
        llm_pool: Bulkhead | None = None,
        config: RuntimeConfig | None = None,
        entail: Entail | None = None,
    ) -> None:
        self.entail = entail  # NLI scorer (triage.nli.entailment); None keeps the lexical verifier
        self.retrieval = ScopedHarness(harness, SUBAGENTS["retrieval"])
        self.execution = ScopedHarness(harness, SUBAGENTS["execution"])
        self.verification = ScopedHarness(harness, SUBAGENTS["verification"])
        self.generator, self.planner = generator, planner
        self.loop_config, self.memory = loop_config, memory or SessionMemory()
        self.auditor = auditor  # Gate 8 async RAGAS post-hook; None disables online scoring
        self.llm_pool = llm_pool  # Gate 9 bulkhead for generator calls
        self.config = config or RuntimeConfig()  # Gate 11 runbooks change it at runtime

    async def handle(self, req: dict[str, Any], *, traceparent: str, caller_tier: Tier) -> dict[str, Any]:
        """`req` is the public API body (design.md §1.2.6). Returns the response body."""
        with span("agent.query", context=parent_context(traceparent), tenant_id=req["tenant_id"]) as s:
            out = await self._handle(req, current_traceparent(traceparent), caller_tier)
            for k in ("route", "status", "abstained"):
                s.set_attribute(k, out[k])
            return out

    async def _handle(self, req: dict[str, Any], traceparent: str, caller_tier: Tier) -> dict[str, Any]:
        with span("triage.classify") as s:
            tri = classify(req["query"])
            s.set_attribute("route", tri.route)
        base = {"tenant_id": req["tenant_id"], "user_id": req["user_id"], "session_id": req["session_id"],
                "traceparent": traceparent}
        out: dict[str, Any] = {
            "tenant_id": req["tenant_id"], "trace_id": traceparent.split("-")[1], "route": tri.route,
            "answer": None, "citations": [], "abstained": False, "status": "DONE", "approval_id": None,
            "faithfulness": None,
        }
        draft_faithfulness: float | None = None  # set only when a draft was vetoed
        try:
            if (canned := canned_response(tri.route)) is not None:
                out["answer"] = canned
            elif tri.route == "HYBRID_RAG":
                draft_faithfulness = await self._rag(req["query"], base, caller_tier, out)
            else:
                await self._execute(req["query"], base, caller_tier, out)
        except Exception as e:  # upstream failure: no draft is released, and the request is still audited
            log.exception("route %s failed", tri.route)
            name = e.name if isinstance(e, HarnessError) else type(e).__name__
            out.update(status="HALTED", answer=f"I could not complete this request ({name}).", citations=[],
                       faithfulness=None)
        await self._audit(base, tri, out, draft_faithfulness)
        return out

    async def _rag(self, query: str, base: dict[str, Any], tier: Tier, out: dict[str, Any]) -> float | None:
        """Fills `out`. Returns the vetoed draft's faithfulness, or None when nothing was vetoed."""
        cfg = self.config
        res = await self.retrieval.call("hybrid_retrieve", {
            "ctx": _ctx(base), "query": query, "bm25_top_k": cfg.top_k, "dense_top_k": cfg.top_k,
            "min_rerank_score": cfg.min_rerank_score}, caller_tier=tier)
        r = res.value if isinstance(res.value, HybridRetrieveOut) else HybridRetrieveOut.model_validate(res.value)
        if r.abstain or not r.chunks:  # deterministic abstention: the LLM is skipped
            out.update(answer=ABSTAIN_TEXT, abstained=True)
            return None
        if cfg.extractive_only:  # hallucination-surge runbook: no generated text is released at all
            top = r.chunks[0]
            out.update(answer=extractive_quote(top.text), citations=[top.uri], faithfulness=1.0)
            return None
        with span("llm.generate", chunks=len(r.chunks)):
            if self.llm_pool is not None:
                draft = await self.llm_pool.run(self.generator.generate, query, r.chunks)
            else:
                draft = await self.generator.generate(query, r.chunks)
        if not draft or _ABSTAIN_KEY in draft.lower():  # tolerate a missing period or an added citation
            out.update(answer=ABSTAIN_TEXT, abstained=True)
            return None
        uri = {c.chunk_hash: c.uri for c in r.chunks}
        with span("verify.nli") as s:
            v = verify(draft, with_question(query, [(c.chunk_hash, c.text) for c in r.chunks]), self.entail)
            s.set_attribute("grounded", v.grounded)
        if v.grounded:
            out.update(answer=draft, citations=[uri[h] for h in v.sources if h in uri], faithfulness=v.faithfulness)
        elif self.entail is not None:
            # NLI veto: the draft's claims are not entailed by the context, so the context does not answer the
            # question. Abstain (KPI-23) rather than quote; a quote would count as an answer.
            out.update(answer=ABSTAIN_TEXT, abstained=True)
            return v.faithfulness
        else:
            # Veto: release only a verbatim quote of the top reranked chunk (it passed the 0.42 relevance cutoff),
            # never the ungrounded draft. The audit keeps the draft's score so KPI-24 still counts the hallucination.
            top = r.chunks[0]
            out.update(answer=extractive_quote(top.text), citations=[top.uri], faithfulness=1.0)
        if self.auditor is not None:  # scores what was released, off the request path
            self.auditor.submit(Sample(out["trace_id"], query, out["answer"], tuple(c.text for c in r.chunks),
                                       base["traceparent"]))
        return None if v.grounded else v.faithfulness

    async def _execute(self, query: str, base: dict[str, Any], tier: Tier, out: dict[str, Any]) -> None:
        if self.planner is None:
            raise RuntimeError("TRANSACTION_EXEC needs a planner")
        loop = BoundedLoop(self.execution, _VerifiedPlanner(self.planner), self.loop_config, memory=self.memory)  # type: ignore[arg-type]
        res = await loop.run(query, base_ctx=base, caller_tier=tier)
        pending = next((o["result"] for o in res.observations
                        if o["tool"] == "submit_transaction" and o.get("ok")
                        and isinstance(o["result"], dict) and o["result"].get("status") == "PENDING_APPROVAL"), None)
        if pending is not None:  # the HITL record exists even if the loop halted after writing it
            out.update(status="PENDING_APPROVAL", approval_id=pending.get("approval_id"),
                       answer=res.answer if res.state is State.DONE
                       else f"Request {pending.get('approval_id')} is waiting for reviewer approval.")
            return
        if res.remediation is not None:
            out.update(status=res.state.value, answer=res.remediation["message"])
            return
        out.update(status=res.state.value, answer=res.answer if res.state is State.DONE
                   else f"I could not complete this request ({res.halt_reason}).")

    async def _audit(self, base: dict[str, Any], tri: TriageResult, out: dict[str, Any],
                     draft_faithfulness: float | None) -> None:
        event = {
            "type": "agent.query", "route": tri.route, "intent": tri.intent, "risk_tier": tri.risk_tier,
            "status": out["status"], "abstained": out["abstained"],
            "vetoed": draft_faithfulness is not None, "draft_faithfulness": draft_faithfulness,
            "faithfulness": out["faithfulness"], "citations": out["citations"], "approval_id": out["approval_id"],
            "answer_sha256": hashlib.sha256((out["answer"] or "").encode()).hexdigest(),
        }
        try:
            await self.verification.call("append_audit", {"ctx": _ctx(base), "event": event}, caller_tier="TIER_1_READ")
        except HarnessError as e:  # a lost audit write must not drop the answer; it is logged for the OTel pipeline
            log.error("audit write failed: %s", e)


def _ctx(base: dict[str, Any]) -> dict[str, Any]:
    return {**base, "idempotency_key": str(uuid4())}
