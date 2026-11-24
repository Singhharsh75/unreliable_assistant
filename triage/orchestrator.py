"""Request orchestration across the subagent topology (design.md §1.3, Gate 7).

triage -> canned reply (DIRECT_ANSWER, REJECT_OOS)
       -> Retrieval Subagent -> generator -> Verification Subagent (HYBRID_RAG)
       -> Execution Subagent: bounded ReAct loop, verified on DONE (TRANSACTION_EXEC)
Every route ends with one audit event written by the Verification Subagent.
Each subagent reaches tools only through its own ScopedHarness, so the binding in SUBAGENTS is enforced.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol
from uuid import uuid4

from eval.ragas_auditor import RagasAuditor, Sample
from guardrails import SAFE_REFUSAL, run_prehooks
from guardrails.review import ReviewQueue
from contracts.models import ABSTAIN_TEXT, Chunk, GraphSubgraphOut, HybridRetrieveOut
from context.anchor import ReActPrompt
from context.cache import AnswerCache
from context.compress import estimate_tokens
from context.memory import SessionMemory
from harness.bulkhead import Bulkhead
from harness.errors import HarnessError
from harness.loop import BoundedLoop, FinalAnswer, LoopConfig, Planner, State, ToolAction
from harness.tool_harness import Tier, ToolExecutionHarness
from observability.incidents import RuntimeConfig
from observability.tracing import current_traceparent, parent_context, span
from triage.classifier import TriageResult, classify
from triage.subagents import SUBAGENTS, ScopedHarness, canned_response
from triage.verifier import Entail, extractive_quote, observation_sources, verify, with_question

log = logging.getLogger("triage.orchestrator")
_ABSTAIN_KEY = ABSTAIN_TEXT.rstrip(".").lower()
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")  # sentence segments, as triage.verifier splits claims


class Generator(Protocol):
    async def generate(self, query: str, chunks: list[Chunk]) -> str | None: ...
    # Generators that use the vault subgraph take `graph: str = ""` on generate (and stream); it is passed only when
    # a subgraph was found, so generators without the parameter keep working.


def build_prompt(query: str, chunks: list[Chunk], graph: str = "") -> str:
    """User turn sent to the LLM; the token counter (KPI-26) measures exactly this text."""
    ctx = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
    related = f"Related rules (vault graph):\n{graph}\n\n" if graph else ""
    return f"{related}Context:\n{ctx}\n\nQuestion: {query}"


class ClaudeGenerator:
    """Production generator: the same prompt, model and client as the Gate 3 eval (eval/run_ragas.py)."""

    def __init__(self) -> None:
        self._client = None

    _prompt = staticmethod(lambda query, chunks, graph="": build_prompt(query, chunks, graph))

    async def generate(self, query: str, chunks: list[Chunk], graph: str = "") -> str | None:
        from eval.run_ragas import GENERATOR, SYSTEM, complete, make_client  # lazy: the eval module imports ragas

        self._client = self._client or make_client()
        return await complete(self._client, GENERATOR, self._prompt(query, chunks, graph), system=SYSTEM)

    async def stream(self, query: str, chunks: list[Chunk], graph: str = "") -> AsyncIterator[str]:
        """Text deltas for the verified-segment stream (design.md §1.3 step 7)."""
        from eval.run_ragas import FALLBACK_BETA, GENERATOR, OPENROUTER, SYSTEM, complete, make_client

        self._client = self._client or make_client()
        if OPENROUTER:  # shortcut: the OpenRouter path is not streamed (one delta); stream it if it becomes the prod path
            if text := await complete(self._client, GENERATOR, self._prompt(query, chunks, graph), system=SYSTEM):
                yield text
            return
        extra: dict[str, Any] = {} if "haiku" in GENERATOR else {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        async with self._client.beta.messages.stream(
                model=GENERATOR, max_tokens=16000, output_config={"effort": "low"}, system=SYSTEM,
                messages=[{"role": "user", "content": self._prompt(query, chunks, graph)}], **extra) as st:
            async for text in st.text_stream:
                yield text


class _VerifiedPlanner:
    """The loop's VERIFY step is the Verification Subagent's claim check against the tool observations."""

    def __init__(self, inner: Planner, entail: Entail | None = None) -> None:
        self.inner, self.entail = inner, entail

    async def reason(self, prompt: ReActPrompt) -> ToolAction | FinalAnswer:
        return await self.inner.reason(prompt)

    async def verify(self, answer: str, observations: list[dict[str, Any]]) -> bool:
        # One claim may join facts from two tool results (lot status + approval id), and a failed call is a source for
        # reporting the failure (triage.verifier.observation_sources). With NLI, a correct paraphrase of a tool result passes.
        return verify(answer, observation_sources(observations), self.entail).grounded


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
        chunk_gate: Callable[[str, str], bool] | None = None,
        cache: AnswerCache | None = None,
    ) -> None:
        self.cache = cache  # tenant-scoped answer cache (ADR-003); None disables it
        # (chunk_hash, text) -> may the generator see it; index-time review verdict, live scan for unknown hashes
        self.chunk_gate = chunk_gate or ReviewQueue().allows
        self.entail = entail  # NLI scorer (triage.nli.entailment); None keeps the lexical verifier
        self.retrieval = ScopedHarness(harness, SUBAGENTS["retrieval"])
        self.execution = ScopedHarness(harness, SUBAGENTS["execution"])
        self.verification = ScopedHarness(harness, SUBAGENTS["verification"])
        self.generator, self.planner = generator, planner
        self.loop_config, self.memory = loop_config, memory or SessionMemory()
        self.auditor = auditor  # Gate 8 async RAGAS post-hook; None disables online scoring
        self.llm_pool = llm_pool  # Gate 9 bulkhead for generator calls
        self.config = config or RuntimeConfig()  # Gate 11 runbooks change it at runtime

    async def handle(self, req: dict[str, Any], *, traceparent: str, caller_tier: Tier,
                     on_segment: Callable[[str], Awaitable[None]] | None = None) -> dict[str, Any]:
        """`req` is the public API body (design.md §1.2.6). Returns the response body. With `on_segment`, a streaming
        generator's verified segments are handed over as they pass the verifier (the API streams them)."""
        with span("agent.query", context=parent_context(traceparent), tenant_id=req["tenant_id"]) as s:
            out = await self._handle(req, current_traceparent(traceparent), caller_tier, on_segment)
            for k in ("route", "status", "abstained"):
                s.set_attribute(k, out[k])
            return out

    async def _handle(self, req: dict[str, Any], traceparent: str, caller_tier: Tier,
                      on_segment: Callable[[str], Awaitable[None]] | None = None) -> dict[str, Any]:
        blocked: str | None = None
        try:  # PII redaction + injection shield; only the redacted text goes further (design.md §1.2.2 step 2)
            query = run_prehooks(req["query"]).text
        except HarnessError as e:
            query, blocked = "", e.name
        with span("triage.classify") as s:
            tri = (TriageResult("REJECT_OOS", "blocked", "none", "TIER_1_READ", "low", 1.0, blocked) if blocked
                   else classify(query))
            s.set_attribute("route", tri.route)
        base = {"tenant_id": req["tenant_id"], "user_id": req["user_id"], "session_id": req["session_id"],
                "traceparent": traceparent}
        out: dict[str, Any] = {
            "tenant_id": req["tenant_id"], "trace_id": traceparent.split("-")[1], "route": tri.route,
            "answer": None, "citations": [], "abstained": False, "status": "DONE", "approval_id": None,
            "faithfulness": None,
        }
        draft_faithfulness: float | None = None  # set only when a draft was vetoed
        cache_hit = False
        try:
            async with asyncio.timeout(self.config.request_timeout_s):  # a timeout lands in the except below: HALTED
                if blocked:
                    out.update(status="HALTED", answer=SAFE_REFUSAL)
                elif (canned := canned_response(tri.route)) is not None:
                    out["answer"] = canned
                elif tri.route == "HYBRID_RAG":
                    hit = self.cache.get(req["tenant_id"], query) if self.cache else None
                    if hit is not None:  # still audited below; a streamed hit goes out as the final event only
                        out.update({k: hit[k] for k in ("answer", "citations", "abstained", "status", "faithfulness")})
                        cache_hit = True
                    else:
                        draft_faithfulness = await self._rag(query, base, caller_tier, out, on_segment)
                        if self.cache and draft_faithfulness is None:  # vetoed or partial answers are never cached
                            self.cache.put(req["tenant_id"], query, out)
                else:
                    await self._execute(query, base, caller_tier, out)
        except Exception as e:  # upstream failure: no draft is released, and the request is still audited
            log.exception("route %s failed", tri.route)
            name = e.name if isinstance(e, HarnessError) else type(e).__name__
            out.update(status="HALTED", answer=f"I could not complete this request ({name}).", citations=[],
                       faithfulness=None)
        await self._audit(base, tri, out, draft_faithfulness, cache_hit)
        return out

    async def _rag(self, query: str, base: dict[str, Any], tier: Tier, out: dict[str, Any],
                   on_segment: Callable[[str], Awaitable[None]] | None = None) -> float | None:
        """Fills `out`. Returns the vetoed draft's faithfulness, or None when nothing was vetoed."""
        cfg = self.config
        res = await self.retrieval.call("hybrid_retrieve", {
            "ctx": _ctx(base), "query": query, "bm25_top_k": cfg.top_k, "dense_top_k": cfg.top_k,
            "min_rerank_score": cfg.min_rerank_score}, caller_tier=tier)
        r = res.value if isinstance(res.value, HybridRetrieveOut) else HybridRetrieveOut.model_validate(res.value)
        with span("hook.chunks") as s:  # indirect injection: poisoned chunks never reach the generator
            clean = [c for c in r.chunks if self.chunk_gate(c.chunk_hash, c.text)]
            s.set_attribute("poisoned", len(r.chunks) - len(clean))
        if len(clean) < len(r.chunks):
            log.warning("dropped %d poisoned chunk(s)", len(r.chunks) - len(clean))
            r = r.model_copy(update={"chunks": clean})
        if r.abstain or not r.chunks:  # deterministic abstention: the LLM is skipped
            out.update(answer=ABSTAIN_TEXT, abstained=True)
            return None
        if cfg.extractive_only:  # hallucination-surge runbook: no generated text is released at all
            top = r.chunks[0]
            out.update(answer=extractive_quote(top.text), citations=[top.uri], faithfulness=1.0)
            return None
        sources = with_question(query, [(c.chunk_hash, c.text) for c in r.chunks])
        uri = {c.chunk_hash: c.uri for c in r.chunks}
        graph, raw_graph = await self._subgraph(base, tier, r.chunks)
        kw = {"graph": graph} if graph else {}
        prompt_tokens = estimate_tokens(build_prompt(query, r.chunks, graph))
        tokens = {"prompt_tokens": prompt_tokens,  # KPI-26: vs the same prompt with every reachable note in full
                  "raw_prompt_tokens": prompt_tokens - estimate_tokens(graph) + raw_graph}
        if on_segment is not None and hasattr(self.generator, "stream"):
            with span("llm.generate", chunks=len(r.chunks), streamed=True, **tokens) as gs:
                args = (query, r.chunks, sources, on_segment, gs, kw)
                draft, released, complete = await (self.llm_pool.run(self._stream_draft, *args) if self.llm_pool
                                                   else self._stream_draft(*args))
            if released:  # verified text already reached the client; it is never retracted
                text = " ".join(released)
                v = verify(text, sources, self.entail)
                out.update(answer=text, citations=[uri[h] for h in v.sources if h in uri], faithfulness=v.faithfulness)
                if self.auditor is not None:
                    self.auditor.submit(Sample(out["trace_id"], query, text, tuple(c.text for c in r.chunks),
                                               base["traceparent"]))
                return None if complete else verify(draft, sources, self.entail).faithfulness
            # nothing passed the verifier: the whole draft takes the non-streaming abstain / veto path below
        else:
            with span("llm.generate", chunks=len(r.chunks), **tokens):
                gen = functools.partial(self.generator.generate, **kw)
                draft = await (self.llm_pool.run(gen, query, r.chunks) if self.llm_pool else gen(query, r.chunks))
        if not draft or _ABSTAIN_KEY in draft.lower():  # tolerate a missing period or an added citation
            out.update(answer=ABSTAIN_TEXT, abstained=True)
            return None
        with span("verify.nli") as s:
            v = verify(draft, sources, self.entail)
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

    async def _subgraph(self, base: dict[str, Any], tier: Tier, chunks: list[Chunk]) -> tuple[str, int]:
        """Rendered vault subgraph around the chunks' notes and its raw size; ("", 0) on any tool failure (FMEA #7:
        the answer stays grounded in the chunks alone)."""
        from mcp_servers.graphify import render

        anchors = list(dict.fromkeys(c.note_id for c in chunks))[:4]
        try:
            res = await self.retrieval.call("graph_subgraph", {"ctx": _ctx(base), "anchor_note_ids": anchors},
                                            caller_tier=tier)
        except HarnessError as e:
            log.info("graph_subgraph skipped: %s", e.name)
            return "", 0
        g = res.value if isinstance(res.value, GraphSubgraphOut) else GraphSubgraphOut.model_validate(res.value)
        return render(g.nodes, g.edges), g.raw_token_estimate

    async def _stream_draft(self, query: str, chunks: list[Chunk], sources: list[tuple[str, str]],
                            on_segment: Callable[[str], Awaitable[None]], gen_span: Any,
                            kw: dict[str, str]) -> tuple[str, list[str], bool]:
        """Steps 7-8: buffer the stream into sentences and verify each before release. A sentence with no checkable
        claim is held and joined to the next. Stops at the first veto or abstention. Returns (draft so far, released
        segments, whether the whole draft was released)."""
        t0, draft, pending, hold, released = time.perf_counter(), "", "", "", []

        async def check(seg: str) -> bool | None:  # True released, None held, False stop
            nonlocal hold
            hold = f"{hold} {seg}".strip()
            if _ABSTAIN_KEY in hold.lower():
                return False
            v = verify(hold, sources, self.entail)
            if not v.claims:
                return None
            if not v.grounded:
                return False
            if not released:
                gen_span.set_attribute("ttft_ms", (time.perf_counter() - t0) * 1000)  # first verified segment
            released.append(hold)
            hold = ""
            await on_segment(released[-1])
            return True

        async for delta in self.generator.stream(query, chunks, **kw):
            draft += delta
            *done, pending = _BOUNDARY.split(pending + delta)
            for seg in filter(str.strip, done):
                if await check(seg) is False:
                    return draft, released, False
        if pending.strip() and await check(pending) is False:
            return draft, released, False
        if hold and not released:  # nothing checkable at all: fail closed through the veto path
            return draft, released, False
        if hold:  # trailing text with no claim ("Let me know if ...") cannot be ungrounded
            released.append(hold)
            await on_segment(hold)
        return draft, released, True

    async def _execute(self, query: str, base: dict[str, Any], tier: Tier, out: dict[str, Any]) -> None:
        if self.planner is None:
            raise RuntimeError("TRANSACTION_EXEC needs a planner")
        loop = BoundedLoop(self.execution, _VerifiedPlanner(self.planner, self.entail), self.loop_config, memory=self.memory,  # type: ignore[arg-type]
                           observation_gate=_tool_output_ok)
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
                     draft_faithfulness: float | None, cache_hit: bool = False) -> None:
        event = {
            "type": "agent.query", "route": tri.route, "intent": tri.intent, "risk_tier": tri.risk_tier,
            "blocked": tri.reason if tri.intent == "blocked" else None,  # PII_BLOCKED | INJECTION_BLOCKED
            "cache_hit": cache_hit,
            "status": out["status"], "abstained": out["abstained"],
            "vetoed": draft_faithfulness is not None, "draft_faithfulness": draft_faithfulness,
            "faithfulness": out["faithfulness"], "citations": out["citations"], "approval_id": out["approval_id"],
            "answer_sha256": hashlib.sha256((out["answer"] or "").encode()).hexdigest(),
        }
        try:
            await self.verification.call("append_audit", {"ctx": _ctx(base), "event": event}, caller_tier="TIER_1_READ")
        except HarnessError as e:  # a lost audit write must not drop the answer; it is logged for the OTel pipeline
            log.error("audit write failed: %s", e)


def _tool_output_ok(text: str) -> bool:
    """Indirect injection through tool output. Regex shield only: the chunk classifier misreads short structured
    text ("id: sop-returns" scores 0.997), which would halt ordinary lookups."""
    from guardrails import injection
    return not injection.scan(text, indirect=True).blocked


def _ctx(base: dict[str, Any]) -> dict[str, Any]:
    return {**base, "idempotency_key": str(uuid4())}
