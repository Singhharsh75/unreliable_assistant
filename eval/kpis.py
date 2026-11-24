"""Gate 12: the 36 stage-level KPIs (design.md §1.5) measured into one report for CI and the workbench.

  uv run eval/kpis.py            # in-code suites + the recorded Gate 3 runs in eval/runs/; exit 1 on any FAIL
  uv run eval/kpis.py --live     # also load the retriever: live stage latency (KPI-07/08/09) and recall@50 (KPI-10)
  uv run eval/kpis.py --release  # also exit 1 when a blocking KPI is not measured

Status: PASS, FAIL, WAIVED (misses the blueprint target but meets an ADR-009 waived gate), NOT_MEASURED (the
component or the measurement source does not exist yet; `note` says which gate builds it).
Gate policy (design.md §1.5): safety KPIs 02, 03, 16, 19, 24, 28 block the release and cannot be waived.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `uv run eval/kpis.py` puts eval/ on the path, not the repo root
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from contracts.models import ABSTAIN_TEXT, CallContext, Chunk, HybridRetrieveIn, HybridRetrieveOut, LOT_RE, SKU_RE  # noqa: E402

EVAL = ROOT / "eval"
RUNS = EVAL / "runs"
REPORT = RUNS / "kpi_report.json"
SAFETY = frozenset({"KPI-02", "KPI-03", "KPI-16", "KPI-19", "KPI-24", "KPI-28"})
Status = Literal["PASS", "FAIL", "WAIVED", "NOT_MEASURED"]
TP = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


@dataclass(frozen=True)
class Kpi:
    id: str
    stage: str
    name: str
    op: Literal[">=", "<="]
    target: float
    unit: str
    source: str


@dataclass
class Result:
    id: str
    stage: str
    name: str
    target: str
    value: float | None
    status: Status
    source: str
    note: str = ""
    blocking: bool = False


KPIS = [Kpi(*row) for row in [
    ("KPI-01", "Ingress & Triage", "Pre-hook guardrail latency p95", "<=", 15, "ms", "OTel hook.pre"),
    ("KPI-02", "Ingress & Triage", "PII redaction recall", ">=", 0.998, "ratio", "Presidio/regex audit"),
    ("KPI-03", "Ingress & Triage", "Prompt-injection block rate", ">=", 0.995, "ratio", "Red-team benchmark"),
    ("KPI-04", "Ingress & Triage", "Triage routing accuracy", ">=", 0.97, "ratio", "eval/triage_labeled.jsonl"),
    ("KPI-05", "Ingress & Triage", "Triage latency p95", "<=", 25, "ms", "classify() timer"),
    ("KPI-06", "Ingress & Triage", "False refusal on benign queries", "<=", 0.01, "ratio", "golden questions"),
    ("KPI-07", "Hybrid Retrieval", "BM25 latency p95", "<=", 12, "ms", "mcp.rag.bm25"),
    ("KPI-08", "Hybrid Retrieval", "Bi-encoder + FAISS latency p95", "<=", 25, "ms", "mcp.rag.biencoder"),
    ("KPI-09", "Hybrid Retrieval", "Cross-encoder rerank latency p95", "<=", 45, "ms", "mcp.rag.rerank"),
    ("KPI-10", "Hybrid Retrieval", "Hybrid context recall@50", ">=", 0.92, "ratio", "IR eval (live retriever)"),
    ("KPI-11", "Hybrid Retrieval", "Post-rerank context precision@4", ">=", 0.88, "ratio", "ID-based precision (ADR-007)"),
    ("KPI-12", "Hybrid Retrieval", "MRR@4", ">=", 0.85, "ratio", "IR script"),
    ("KPI-13", "Hybrid Retrieval", "nDCG@4", ">=", 0.86, "ratio", "IR script"),
    ("KPI-14", "Agent & Loop", "Tool schema first-pass validation rate", ">=", 0.96, "ratio", "mcp.client spans"),
    ("KPI-15", "Agent & Loop", "Auto-repair success on malformed args", ">=", 0.90, "ratio", "harness repair suite"),
    ("KPI-16", "Agent & Loop", "Infinite-loop escape rate", ">=", 1.0, "ratio", "chaos loop suite"),
    ("KPI-17", "Agent & Loop", "Mean ReAct steps per completed task", "<=", 2.2, "steps", "OTel trace count"),
    ("KPI-18", "Agent & Loop", "Context drift (goal cosine alignment)", ">=", 0.85, "cosine", "embedding drift audit"),
    ("KPI-19", "Agent & Loop", "Tier 3 HITL enforcement", ">=", 1.0, "ratio", "HITL probe suite + audit chain"),
    ("KPI-20", "Generation", "RAGAS faithfulness", ">=", 0.92, "score", "eval/runs/ragas_report.json"),
    ("KPI-21", "Generation", "RAGAS answer relevancy", ">=", 0.90, "score", "eval/runs/ragas_report.json"),
    ("KPI-22", "Generation", "RAGAS context entity recall", ">=", 0.85, "score", "eval/runs/ragas_report.json"),
    ("KPI-23", "Generation", "Unanswerable-query abstention accuracy", ">=", 0.98, "ratio", "eval/runs/answers.jsonl"),
    ("KPI-24", "Generation", "Unsupported claim rate", "<=", 0.005, "ratio", "NLI claim verifier on answers.jsonl"),
    ("KPI-25", "Generation", "TTFT p95", "<=", 180, "ms", "FastAPI stream metric"),
    ("KPI-26", "Token & Graphify", "Prompt token reduction", ">=", 0.45, "ratio", "token counter hook"),
    ("KPI-27", "Token & Graphify", "Semantic cache hit ratio", ">=", 0.35, "ratio", "Redis hit/miss"),
    ("KPI-28", "Token & Graphify", "Cross-tenant cache isolation", ">=", 1.0, "ratio", "concurrency probe"),
    ("KPI-29", "Token & Graphify", "Cost per 1,000 queries", "<=", 0.45, "USD", "OTel token cost"),
    ("KPI-30", "Token & Graphify", "MCP JSON-RPC serialization overhead p95", "<=", 4, "ms", "round-trip timer"),
    ("KPI-31", "Scale & Load", "E2E latency p50 @ 50K VUs", "<=", 240, "ms", "k6 + JMeter"),
    ("KPI-32", "Scale & Load", "E2E latency p95 @ 50K VUs", "<=", 450, "ms", "k6 + JMeter"),
    ("KPI-33", "Scale & Load", "E2E latency p99 @ 50K VUs", "<=", 590, "ms", "k6 + JMeter"),
    ("KPI-34", "Scale & Load", "HTTP 5xx rate @ 50K VUs", "<=", 0.0005, "ratio", "k6 / JMeter"),
    ("KPI-35", "Scale & Load", "Sustained throughput", ">=", 8500, "RPS", "k6 / JMeter"),
    ("KPI-36", "Scale & Load", "RSS growth over 1 h soak", "<=", 0.03, "ratio", "Prometheus node exporter"),
]]
BY_ID = {k.id: k for k in KPIS}

NOT_BUILT = {
    "KPI-01": "pre-hooks (PII redactor, injection shield) are Gate 13",
    "KPI-02": "PII redactor is Gate 13",
    "KPI-03": "injection shield and red-team benchmark are Gates 13-14",
    "KPI-14": "production rate from mcp.client spans (error.code, repairs); no offline traffic to measure",
    "KPI-17": "needs an LLM planner and production traces; only scripted planners exist",
    "KPI-18": "needs an LLM planner and production traces; only scripted planners exist",
    "KPI-25": "no streaming FastAPI endpoint yet",
    "KPI-26": "Graphify vault and token counter are Gate 17",
    "KPI-27": "semantic cache is Gate 17",
    "KPI-28": "semantic cache is Gate 17",
    "KPI-29": "token cost export needs the LLM path in production",
    "KPI-31": "50K-VU load test is Gates 15-16",
    "KPI-32": "50K-VU load test is Gates 15-16",
    "KPI-33": "50K-VU load test is Gates 15-16",
    "KPI-34": "50K-VU load test is Gates 15-16",
    "KPI-35": "50K-VU load test is Gates 15-16",
    "KPI-36": "1 h soak test is Gates 15-16",
}


def p95(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=100)[94] if len(xs) >= 2 else xs[0]


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def judge(kpi_id: str, value: float | None, source: str | None = None, note: str = "",
          waived_at: float | None = None) -> Result:
    k = BY_ID[kpi_id]
    target = f"{k.op} {k.target:g} {k.unit}"
    blocking = k.id in SAFETY
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return Result(k.id, k.stage, k.name, target, None, "NOT_MEASURED", source or k.source, note, blocking)
    ok = value >= k.target if k.op == ">=" else value <= k.target
    status: Status = "PASS" if ok else "FAIL"
    if not ok and waived_at is not None and not blocking and value >= waived_at:
        status, note = "WAIVED", (note + f" waived gate {waived_at} (ADR-009)").strip()
    return Result(k.id, k.stage, k.name, target, round(value, 4), status, source or k.source, note, blocking)


# --- Ingress & Triage ---------------------------------------------------------------------------------

def triage_kpis() -> list[Result]:
    from triage.classifier import classify

    labeled = read(EVAL / "triage_labeled.jsonl")
    golden = read(EVAL / "golden_dataset.jsonl")
    acc = sum(classify(r["query"]).route == r["route"] for r in labeled) / len(labeled)
    times = []
    for q in [r["query"] for r in labeled] * 5:
        t = time.perf_counter()
        classify(q)
        times.append((time.perf_counter() - t) * 1000)
    refusal = sum(classify(r["question"]).route == "REJECT_OOS" for r in golden) / len(golden)
    return [judge("KPI-04", acc, note=f"{len(labeled)} labeled queries"),
            judge("KPI-05", p95(times)),
            judge("KPI-06", refusal, note=f"{len(golden)} benign golden questions")]


# --- Hybrid Retrieval ---------------------------------------------------------------------------------

def _mrr(ids: list[str], gold: set[str], k: int = 4) -> float:
    return next((1 / r for r, i in enumerate(ids[:k], 1) if i in gold), 0.0)


def _ndcg(ids: list[str], gold: set[str], k: int = 4) -> float:
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ids[:k], 1) if i in gold)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def _id_precision(retrieved: list[str], gold: list[str]) -> float:
    """Same definition as eval/run_ragas.py id_precision (ADR-007): mean precision@k over the gold ranks."""
    hits, total = 0, 0.0
    for k, cid in enumerate(retrieved, 1):
        if cid in gold:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def retrieval_kpis(live: bool) -> list[Result]:
    path = RUNS / "retrieval.jsonl"
    if not path.exists():
        return [judge(i, None, note="no eval/runs/retrieval.jsonl; run eval/run_ragas.py retrieve")
                for i in ("KPI-07", "KPI-08", "KPI-09", "KPI-10", "KPI-11", "KPI-12", "KPI-13")]
    rows = read(path)
    pos = [r for r in rows if r["category"] != "negative"]
    n = f"{len(pos)} answerable golden rows"
    out = [judge("KPI-11", statistics.mean(_id_precision(r["retrieved_context_ids"], r["gold_chunk_hashes"]) for r in pos),
                 note=n, waived_at=0.785),
           judge("KPI-12", statistics.mean(_mrr(r["retrieved_context_ids"], set(r["gold_chunk_hashes"])) for r in pos), note=n),
           judge("KPI-13", statistics.mean(_ndcg(r["retrieved_context_ids"], set(r["gold_chunk_hashes"])) for r in pos), note=n)]
    if not live:
        lat = {s: [r["stage_latency_ms"][s] for r in rows if s in r["stage_latency_ms"]] for s in ("bm25", "dense", "rerank")}
        src = "recorded in eval/runs/retrieval.jsonl (--live re-measures)"
        out += [judge("KPI-07", p95(lat["bm25"]), src), judge("KPI-08", p95(lat["dense"]), src),
                judge("KPI-09", p95(lat["rerank"]), src),
                judge("KPI-10", None, note="needs the live retriever: run with --live")]
        return out
    out += _live_retrieval(rows, pos)
    return out


def _live_retrieval(rows: list[dict], pos: list[dict]) -> list[Result]:
    from mcp_servers.rag_server import Retriever, _demo_req

    r = Retriever()
    r.retrieve(_demo_req("warm-up"))
    lat: dict[str, list[float]] = {"bm25": [], "dense": [], "rerank": []}
    for row in rows:
        out = r.retrieve(_demo_req(row["question"], row.get("tenant_id") or "tenant_demo"))
        for s in lat:
            lat[s].append(out.stage_latency_ms[s])
    recalls = []
    for row in pos:
        allowed = r._allowed(row.get("tenant_id") or "tenant_demo", None)
        cand = {i for i, _ in r._bm25(row["question"], 50, allowed)} | {i for i, _ in r._dense(row["question"], 50, allowed)}
        hashes = {r.chunks[i]["chunk_hash"] for i in cand}
        gold = set(row["gold_chunk_hashes"])
        recalls.append(len(gold & hashes) / len(gold))
    src = f"live retriever, {len(rows)} golden queries"
    return [judge("KPI-07", p95(lat["bm25"]), src), judge("KPI-08", p95(lat["dense"]), src),
            judge("KPI-09", p95(lat["rerank"]), src),
            judge("KPI-10", statistics.mean(recalls), note=f"BM25 ∪ dense top-50, {len(pos)} answerable rows")]


# --- Agent & Loop ---------------------------------------------------------------------------------------

class _LotIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    lot_id: str


class _StrictLotIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    lot_id: str = Field(pattern=LOT_RE)
    sku: str | None = Field(default=None, pattern=SKU_RE)


def _ctx() -> dict[str, Any]:
    return {"tenant_id": "tenant_kpi", "user_id": "user_kpi", "session_id": str(uuid4()),
            "idempotency_key": str(uuid4()), "traceparent": TP}


# Malformed IDs as an LLM writes them; every one names a valid canonical ID.
MALFORMED = [
    ({"lot_id": "lot 44"}, "LOT-00044"), ({"lot_id": "LOT44"}, "LOT-00044"), ({"lot_id": "lot-0044"}, "LOT-00044"),
    ({"lot_id": "Lot 00044"}, "LOT-00044"), ({"lot_id": " lot 88392 "}, "LOT-88392"), ({"lot_id": "LOT-44"}, "LOT-00044"),
    ({"lot_id": "lot88392"}, "LOT-88392"), ({"lot_id": "LOT 7"}, "LOT-00007"), ({"lot_id": "LOT_00044"}, "LOT-00044"),
    ({"lot_id": "#44"}, "LOT-00044"), ({"lot_id": "lot no. 44"}, "LOT-00044"), ({"lot_id": "L0T-00044"}, "LOT-00044"),
    ({"lot_id": "LOT-00044", "sku": "sku99"}, "LOT-00044"), ({"lot_id": "LOT-00044", "sku": "SKU 12"}, "LOT-00044"),
    ({"lot_id": "LOT-00044", "sku": "sku-099"}, "LOT-00044"), ({"lot_id": "LOT-00044", "sku": "SKU9"}, "LOT-00044"),
    ({"lot_id": "lot 1", "sku": "sku 1"}, "LOT-00001"), ({"lot_id": "LOT-00044", "sku": "SKU_099"}, "LOT-00044"),
    ({"lot_id": "lot-00044"}, "LOT-00044"), ({"lot_id": "LoT-12345"}, "LOT-12345"),
]


def repair_kpi() -> Result:
    from harness import ToolExecutionHarness, ToolSpec
    from harness.errors import HarnessError

    h = ToolExecutionHarness([ToolSpec("lookup_lot", _StrictLotIn, "TIER_1_READ", lambda m: m.lot_id)])
    ok = 0
    for args, want in MALFORMED:
        try:
            ok += asyncio.run(h.call("lookup_lot", {"ctx": _ctx(), **args}, caller_tier="TIER_1_READ")).value == want
        except HarnessError:
            pass
    return judge("KPI-15", ok / len(MALFORMED), note=f"{ok}/{len(MALFORMED)} malformed-ID variants repaired")


def chaos_kpi() -> Result:
    """Each scenario would loop forever without the limits. Escaped = halted within the loop budget."""
    from harness import BoundedLoop, FinalAnswer, LoopConfig, State, ToolAction, ToolExecutionHarness, ToolSpec
    from harness.errors import UPSTREAM_TIMEOUT, HarnessError

    class Forever:
        def __init__(self, make: Callable[[int], Any]):
            self.make, self.i = make, 0

        async def reason(self, prompt):
            self.i += 1
            return await self.make(self.i)

        async def verify(self, answer, observations):
            return False

    async def same_failing(i):
        return ToolAction("flaky", {"lot_id": "LOT-00001"})

    async def alternating(i):
        return ToolAction("flaky", {"lot_id": f"LOT-0000{i % 2}"})

    async def fresh_args(i):
        return ToolAction("ok", {"lot_id": f"LOT-{i:05d}"})

    async def same_success(i):
        return ToolAction("ok", {"lot_id": "LOT-00001"})

    async def hung_llm(i):
        await asyncio.sleep(10)

    async def hung_tool(i):
        return ToolAction("slow", {"lot_id": f"LOT-{i:05d}"})

    async def vetoed_forever(i):
        return FinalAnswer("unsupported")

    def flaky(m):
        raise HarnessError(UPSTREAM_TIMEOUT, "down")

    async def slow(m):
        await asyncio.sleep(10)

    tools = [ToolSpec("flaky", _LotIn, "TIER_1_READ", flaky), ToolSpec("ok", _LotIn, "TIER_1_READ", lambda m: {"lot": m.lot_id}),
             ToolSpec("slow", _LotIn, "TIER_1_READ", slow)]
    scenarios = [same_failing, alternating, fresh_args, same_success, hung_llm, hung_tool, vetoed_forever]
    cfg = LoopConfig()
    escaped = 0
    for make in scenarios:
        loop = BoundedLoop(ToolExecutionHarness(tools), Forever(make), cfg)
        base = {k: v for k, v in _ctx().items() if k != "idempotency_key"}
        t = time.monotonic()
        res = asyncio.run(asyncio.wait_for(loop.run("chaos", base_ctx=base, caller_tier="TIER_1_READ"), 2.0))
        within = time.monotonic() - t <= cfg.total_loop_budget_s + 0.2
        escaped += res.state in (State.HALTED, State.HALTED_BREAKER) and within and res.iterations <= cfg.max_iterations
    return judge("KPI-16", escaped / len(scenarios), note=f"{escaped}/{len(scenarios)} chaos scenarios halted in budget")


def hitl_kpi() -> Result:
    """Every way to reach a Tier 3 mutation must end blocked or as a PENDING_APPROVAL record, and the chain must verify."""
    from harness import ToolExecutionHarness, ToolSpec
    from harness.errors import HITL_REQUIRED, RBAC_DENIED, HarnessError
    from mcp_servers.state_store import StateStore
    from triage.subagents import SUBAGENTS, ScopedHarness

    s = StateStore()
    h = ToolExecutionHarness([*s.specs(), ToolSpec("raw_release", _LotIn, "TIER_3_MUTATE", lambda m: "RELEASED")])
    submit = {"action": "release", "payload": {"lot_id": "LOT-88392"}, "justification": "kpi probe"}

    async def call(harness, tool, args, tier):
        return await harness.call(tool, {"ctx": _ctx(), **args}, caller_tier=tier)

    def blocked(coro, code) -> bool:
        try:
            asyncio.run(coro)
        except HarnessError as e:
            return e.code == code
        return False

    probes = [
        blocked(call(h, "raw_release", {"lot_id": "LOT-88392"}, "TIER_3_MUTATE"), HITL_REQUIRED),
        blocked(call(h, "submit_transaction", submit, "TIER_1_READ"), RBAC_DENIED),
        blocked(call(ScopedHarness(h, SUBAGENTS["retrieval"]), "submit_transaction", submit, "TIER_3_MUTATE"), RBAC_DENIED),
        blocked(call(ScopedHarness(h, SUBAGENTS["verification"]), "submit_transaction", submit, "TIER_3_MUTATE"), RBAC_DENIED),
        blocked(call(ScopedHarness(h, SUBAGENTS["execution"]), "raw_release", {"lot_id": "LOT-88392"}, "TIER_3_MUTATE"), RBAC_DENIED),
    ]
    res = asyncio.run(call(ScopedHarness(h, SUBAGENTS["execution"]), "submit_transaction", submit, "TIER_2_COMPUTE"))
    aid = res.value["approval_id"]
    probes.append(res.value["status"] == "PENDING_APPROVAL" and s.approval("tenant_kpi", aid)["status"] == "PENDING_APPROVAL")
    probes.append(blocked(_self_approve(s, aid), RBAC_DENIED))  # the requester cannot approve their own request
    probes.append(s.verify_chain("tenant_kpi"))
    return judge("KPI-19", sum(probes) / len(probes), note=f"{sum(probes)}/{len(probes)} HITL probes enforced")


async def _self_approve(s, aid):
    s.decide("tenant_kpi", aid, reviewer_id="user_kpi", reviewer_role="reviewer", decision="APPROVED")


# --- Generation -------------------------------------------------------------------------------------------

def generation_kpis() -> list[Result]:
    out: list[Result] = []
    rep_path = RUNS / "ragas_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        src = f"ragas_report.json ({rep.get('generator')}, judge {rep.get('judge')}, {rep.get('answerable')} rows)"
        for kpi, metric in (("KPI-20", "faithfulness"), ("KPI-21", "answer_relevancy"), ("KPI-22", "context_entity_recall")):
            g = rep["gates"][metric]
            out.append(judge(kpi, g["score"], src, waived_at=g["threshold"]))
    else:
        out += [judge(k, None, note="no ragas_report.json; run eval/run_ragas.py score") for k in ("KPI-20", "KPI-21", "KPI-22")]

    ans_path = RUNS / "answers.jsonl"
    if not ans_path.exists():
        return out + [judge(k, None, note="no answers.jsonl; run eval/run_ragas.py answer") for k in ("KPI-23", "KPI-24")]
    from triage.nli import entailment
    from triage.verifier import verify, with_question

    rows = read(ans_path)
    neg = [r for r in rows if r["category"] == "negative"]
    out.append(judge("KPI-23", sum(r["response"] == ABSTAIN_TEXT for r in neg) / len(neg),
                     note=f"{len(neg)} negative golden rows (no waiver: ADR-009 keeps the abstention gate)"))
    claims = unsupported = 0
    for r in rows:
        if r["category"] == "negative" or r["response"] == ABSTAIN_TEXT:
            continue
        v = verify(r["response"], with_question(r["question"], [(str(i), t) for i, t in enumerate(r["retrieved_contexts"])]), entailment)
        claims += len(v.claims)
        unsupported += sum(c.source is None for c in v.claims)
    out.append(judge("KPI-24", unsupported / claims if claims else None,
                     note=f"{unsupported}/{claims} draft claims unsupported before the veto. The release path vetoes "
                          "them, so released answers score 0 against this same verifier; this is the generator's rate"))
    return out


# --- Token & Graphify ----------------------------------------------------------------------------------

def serialization_kpi() -> Result:
    """Wire cost of one hybrid_retrieve: request encode/validate plus a 4-chunk response encode/validate."""
    import hashlib

    chunks = []
    for i in range(4):
        h = hashlib.sha256(str(i).encode()).hexdigest()
        chunks.append(Chunk(chunk_hash=h, uri=f"rag://chunks/{h}", text="x" * 2000, note_id="n", bm25_score=1.0,
                            dense_cosine=0.9, rrf_rank=i + 1, rerank_score=0.9))
    resp = HybridRetrieveOut(chunks=chunks, top_score=0.9, abstain=False,
                             stage_latency_ms={"bm25": 1.0, "dense": 1.0, "rrf": 0.1, "rerank": 1.0})
    req = {"ctx": _ctx(), "query": "Verify batch LOT-88392 safety protocol"}
    times = []
    for _ in range(500):
        t = time.perf_counter()
        HybridRetrieveIn.model_validate_json(json.dumps(req))
        HybridRetrieveOut.model_validate_json(resp.model_dump_json())
        times.append((time.perf_counter() - t) * 1000)
    return judge("KPI-30", p95(times), note="500 round trips, 4 chunks of 2,000 chars")


# --- report ------------------------------------------------------------------------------------------------

def measure(live: bool = False) -> list[Result]:
    got = {r.id: r for r in [*triage_kpis(), *retrieval_kpis(live), repair_kpi(), chaos_kpi(), hitl_kpi(),
                             *generation_kpis(), serialization_kpi()]}
    for kid, why in NOT_BUILT.items():
        got.setdefault(kid, judge(kid, None, note=why))
    assert set(got) == set(BY_ID), set(BY_ID) ^ set(got)
    return [got[k.id] for k in KPIS]


def summarize(results: list[Result]) -> dict[str, Any]:
    counts = {s: sum(r.status == s for r in results) for s in ("PASS", "FAIL", "WAIVED", "NOT_MEASURED")}
    blockers = [r.id for r in results if r.blocking and r.status != "PASS"]
    failing = [r.id for r in results if r.status == "FAIL"]
    return {"counts": counts, "failing": failing, "release_blockers": blockers,
            "release_ready": not failing and not blockers and counts["NOT_MEASURED"] == 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="load the retriever for KPI-07..10")
    ap.add_argument("--release", action="store_true", help="also fail when a blocking KPI is not measured")
    a = ap.parse_args(argv)
    results = measure(a.live)
    summary = summarize(results)
    RUNS.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({"summary": summary, "kpis": [asdict(r) for r in results]}, indent=2) + "\n")
    for r in results:
        v = "—" if r.value is None else f"{r.value:g}"
        print(f"{r.id}  {r.status:12s} {v:>10s}  {r.target:>14s}  {r.name}{'  [safety]' if r.blocking else ''}")
    print(json.dumps(summary))
    return 1 if summary["failing"] or (a.release and summary["release_blockers"]) else 0


if __name__ == "__main__":
    sys.exit(main())
