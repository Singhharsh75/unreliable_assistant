# design.md — The Unreliable Assistant → Enterprise Grade

> **Status:** LOCKED v1.2 (Gate 1) — locked 2026-11-18; design finalized 2026-11-22; amended 2026-11-22 (ADR-006, reranker) · **Owner:** Harsh Singh · **Date:** 2026-11-22
> **Source:** `docs/Unreliable_Assistant_MidCheck_Enterprise_Blueprint_v2.docx` (Gates 0–1)
> **Rule:** No implementation code until this file is reviewed and marked **LOCKED**.

---

## 0. Context & Stack (from Gate 0)

The prototype we inherited has five known failures. Each one maps to a control in this design:

| Prototype failure | Primary control | Section |
|---|---|---|
| Hallucinated facts | 3-stage hybrid RAG, abstention short-circuit (rerank < 0.42), Verification Subagent veto | 1.2, 1.6, ADR-001 |
| Runaway tool loops | Bounded ReAct state machine (`max_iterations=5`), idempotent circuit breaker | 1.2, ADR-004 |
| Cross-user cache leaks | Tenant-scoped `SHA256(tenant_id:norm_query)` Redis keys (Tier 1, non-personalized only), tenant_id in every query | ADR-003 |
| Context drift | Immutable `GoalAnchorBlock`, observation compression, sliding window | 1.3 |
| Latency collapse under load | Async FastAPI, bulkheads, ONNX quantized models, semantic cache | 1.4, ADR-002 |

| Layer | Choice | MCP / Plugin binding |
|---|---|---|
| Agent runtime & API | Python 3.11, FastAPI, Uvicorn (async) | MCP host client (`mcp.ClientSession`) |
| Sparse + dense retrieval | rank-bm25, FAISS HNSW, `bge-large-en-v1.5` (ONNX) | MCP server `mcp-hybrid-rag` (stdio) |
| LLM | Claude via the Anthropic API (`anthropic` SDK), model `claude-haiku-5-5` (ADR-008; `claude-opus-5-5` is the upgrade path) | Called by the Retrieval, Execution and Verification subagents |
| Reranker | ONNX Runtime, `ms-marco-MiniLM-L-6-v2` FP32 (ADR-006; `bge-reranker-v2-m3` int8 is the fallback) | In-process / Triton microservice |
| State & cache | SQLite (local), PostgreSQL +pgvector (dev, prod), Redis Cluster | MCP server `mcp-state-store` (Streamable HTTP) |
| Token compression | Obsidian vault + Graphify graph | MCP server `mcp-graphify-vault` (stdio) |
| Lifecycle plugins | Pluggy entry-point hooks | `pii_redactor`, `injection_shield`, `circuit_breaker`, `ragas_auditor`, `otel_exporter` |

---

## 1.1 System Charter & Hard Boundaries

### Purpose
This is a multi-tenant enterprise operations assistant. It answers policy, SOP and product-catalog questions from a curated knowledge vault and grounds every answer in a citation. It can also run bounded, schema-validated workflow actions, such as batch/lot verification, form submission and status transitions, against the state store.

*Assumption:* the domain is regulated manufacturing/ops. This comes from the load-test query `Verify batch LOT-88392 safety protocol` and the `SKU-099` / `LOT-00044` identifiers. **Confirm.**

### In scope
- Grounded Q&A over `knowledge_vault/` (factoid, multi-hop, summary). Every claim is cited to a chunk.
- Deterministic abstention when the evidence is insufficient.
- Tier 1 (read) and Tier 2 (compute) tool calls run without a human.
- Tier 3 (mutate) tool calls only through the HITL approval gate.
- Strict tenant isolation, PII redaction on ingress, and a hash-chained audit trail.
- 50K concurrent VUs at p95 ≤ 450 ms and p99 ≤ 600 ms.

### Non-goals (hard boundaries)
| # | Boundary | Enforcement |
|---|---|---|
| NG-1 | **No autonomous clinical, financial or legal execution.** Any such action is Tier 3 and requires HITL. | RBAC tier check in `ToolExecutionHarness` plus a `PENDING_APPROVAL` record |
| NG-2 | No open-web browsing or external retrieval | Retrieval Subagent is bound only to `mcp-hybrid-rag` and `mcp-graphify-vault` |
| NG-3 | No answers from parametric memory when retrieval confidence is low | Abstention short-circuit |
| NG-4 | No raw PII persisted anywhere (logs, cache, Postgres, traces) | `pii_redactor` runs fail-closed at PRE_EXECUTION |
| NG-5 | No cross-tenant data access, including cache | Tenant-scoped keys and queries; KPI-28 = 0 leaks |
| NG-6 | No unbounded agent loops or recursive self-calls without a state change | State machine and circuit breaker |
| NG-7 | No model training or fine-tuning on tenant data | Out of scope |

### Personas
| Persona | Goal | Max tier |
|---|---|---|
| **Operator** (tenant end user) | Ask policy/SOP questions, verify lots/SKUs, start workflows | Tier 2; Tier 3 only as a request |
| **Reviewer** (HITL approver) | Approve or reject pending Tier 3 mutations with full context | Approves Tier 3 |
| **Tenant Admin** | Manage users, roles and vault content for one tenant | Admin within one tenant |
| **Platform Engineer** | Operate the system, read traces and KPIs, run red-team and load tests in the Streamlit workbench | No tenant data access beyond redacted traces |

---

## 1.2 MCP Tool & Data Contracts

All tools use MCP over JSON-RPC 2.0. Each call passes through `ToolExecutionHarness` in this order:
1. Pydantic v2 `strict=True` validation
2. Deterministic regex auto-repair
3. RBAC tier check
4. UUIDv4 idempotency check

### 1.2.1 Shared types & ID constraints

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from uuid import UUID

TENANT_RE = r"^tenant_[a-z0-9]{1,32}$"
USER_RE   = r"^user_[a-z0-9]{1,64}$"
SKU_RE    = r"^SKU-\d{3}$"        # repair: "SKU99"  -> "SKU-099"
LOT_RE    = r"^LOT-\d{5}$"        # repair: "lot 44" -> "LOT-00044"
CHUNK_RE  = r"^[a-f0-9]{64}$"     # sha256 chunk_hash

Tier = Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"]

class CallContext(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    tenant_id: str = Field(pattern=TENANT_RE)
    user_id: str = Field(pattern=USER_RE)
    session_id: UUID
    idempotency_key: UUID            # UUIDv4, required on every call
    traceparent: str = Field(pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$")
```

> Decision: keep these formats. `LOT-` + 5 digits fits the k6 query `LOT-88392` and the repair `lot 44 → LOT-00044`; `SKU-` + 3 digits fits `SKU99 → SKU-099`. Revisit if real data disagrees.

### 1.2.2 `mcp-hybrid-rag`

**Tool `hybrid_retrieve`** (Tier 1)
```python
class HybridRetrieveIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    query: str = Field(min_length=1, max_length=2000)
    domain: str | None = Field(default=None, pattern=r"^[a-z_]{1,32}$")
    bm25_top_k: int = Field(default=50, ge=1, le=100)
    dense_top_k: int = Field(default=50, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rerank_k: int = Field(default=4, ge=1, le=10)
    min_rerank_score: float = Field(default=0.42, ge=0.0, le=1.0)

class Chunk(BaseModel):
    chunk_hash: str = Field(pattern=CHUNK_RE)
    uri: str = Field(pattern=r"^rag://chunks/[a-f0-9]{64}$")
    text: str = Field(max_length=4096)
    note_id: str                     # Obsidian frontmatter id
    bm25_score: float | None
    dense_cosine: float | None
    rrf_rank: int
    rerank_score: float              # sigmoid, 0..1

class HybridRetrieveOut(BaseModel):
    chunks: list[Chunk] = Field(max_length=10)
    top_score: float
    abstain: bool                    # True when top_score < min_rerank_score
    stage_latency_ms: dict[Literal["bm25", "dense", "rrf", "rerank"], float]
```

**Resource `rag://chunks/{chunk_hash}`** (read-only) returns `Chunk`. Tenant ACL is checked on every read.

### 1.2.3 `mcp-graphify-vault`

**Tool `graph_subgraph`** (Tier 1)
```python
class GraphSubgraphIn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    ctx: CallContext
    anchor_note_ids: list[str] = Field(min_length=1, max_length=4)
    depth: int = Field(default=2, ge=1, le=2)    # hard cap at 2 hops
    max_tokens: int = Field(default=320, ge=64, le=800)

class GraphSubgraphOut(BaseModel):
    nodes: list[dict]          # {id, domain, risk_tier, version, summary}
    edges: list[tuple[str, str, str]]   # (src, rel, dst)
    token_count: int
    raw_token_estimate: int    # for KPI-26 savings
```

### 1.2.4 `mcp-state-store`

| Tool | Tier | Input → Output |
|---|---|---|
| `get_session_facts` | 1 | `{ctx}` → `{facts: list[SessionFact]}` |
| `put_session_fact` | 2 | `{ctx, fact: SessionFact}` → `{fact_id}` (verified facts only) |
| `lookup_lot` | 1 | `{ctx, lot_id: LOT_RE}` → `{lot_id, status, protocol_ref, updated_at}` |
| `submit_transaction` | **3** | `{ctx, action: Literal[...], payload: dict, justification: str}` → `{approval_id, status: "PENDING_APPROVAL"}` |
| `get_approval` | 1 | `{ctx, approval_id}` → `{status: PENDING_APPROVAL \| APPROVED \| REJECTED, reviewer_id?, decided_at?}` |
| `append_audit` | internal | `{ctx, event: dict}` → `{seq, hash}` where `hash = SHA256(prev_hash + canonical(event))` |

```python
class SessionFact(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    key: str = Field(pattern=r"^[a-z_]{1,64}$")
    value: str = Field(max_length=512)
    source_chunk_hash: str | None = Field(default=None, pattern=CHUNK_RE)
```

`submit_transaction` never executes the action. It only writes `PENDING_APPROVAL`. The action runs after a Reviewer approves, through a separate worker path.

### 1.2.5 Error codes (JSON-RPC `error.code`)

| Code | Name | Meaning | Caller behaviour |
|---|---|---|---|
| -32600 / -32601 / -32602 | (standard) | Invalid request / method / params | Auto-repair once, then fail |
| -32001 | `SCHEMA_VALIDATION_FAILED` | Pydantic strict failure after repair | Return to ReAct as an observation, counted toward the breaker |
| -32002 | `AUTO_REPAIR_FAILED` | Regex repair could not produce a valid ID | Ask the user to clarify |
| -32003 | `RBAC_DENIED` | Caller tier < tool tier | Explain the refusal; never retry |
| -32004 | `IDEMPOTENCY_REPLAY` | Key already seen | Return the cached result |
| -32005 | `TENANT_SCOPE_VIOLATION` | Resource is not owned by `ctx.tenant_id` | Hard fail plus a security audit event |
| -32006 | `HITL_REQUIRED` | Tier 3 attempted outside `submit_transaction` | Route to the HITL gate |
| -32007 | `CIRCUIT_OPEN` | Breaker tripped on `(tool, args_hash)` | `HALTED_BREAKER` and a remediation prompt |
| -32008 | `UPSTREAM_TIMEOUT` | Step exceeded its budget | Use the fallback from the FMEA |
| -32009 | `LOW_CONFIDENCE_ABSTAIN` | Rerank top score < threshold | Deterministic abstention response |
| -32010 | `PII_BLOCKED` | Unredactable PII detected | Fail closed |
| -32011 | `INJECTION_BLOCKED` | Injection classifier positive | Fail closed with a safe refusal |

### 1.2.6 Public API

`POST /api/v1/agent/query`
```json
// request
{"tenant_id": "tenant_7", "user_id": "user_123", "session_id": "uuid", "query": "string"}
// header X-Tenant-ID must equal body tenant_id, else 403
// response
{"tenant_id": "tenant_7", "trace_id": "...", "route": "HYBRID_RAG",
 "answer": "...", "citations": ["rag://chunks/<hash>"], "abstained": false,
 "status": "DONE | HALTED | HALTED_BREAKER | PENDING_APPROVAL",
 "approval_id": null, "faithfulness": 0.95}
```

---

## 1.3 Distributed Topology & Data Flow

```mermaid
flowchart LR
  C[Client / k6 / JMeter] --> GW[API Gateway<br/>NGINX/Envoy · authN · rate limit]
  GW --> API[FastAPI workers<br/>stateless · Uvicorn async]

  subgraph API_PROC[Agent runtime]
    PRE[Pre-Exec plugins<br/>PII Redactor → Injection Shield]
    CACHE{Semantic cache<br/>lookup}
    TRI[Triage Agent<br/>temp=0.0 · ≤25 ms]
    RET[Retrieval Subagent<br/>read-only]
    EXE[Execution Subagent<br/>ReAct max 5 · breaker]
    VER[Verification Subagent<br/>NLI claim check]
    POST[Post-Exec plugins<br/>RAGAS · OTel]
    PRE --> CACHE
    CACHE -- miss --> TRI
    TRI -- DIRECT_ANSWER --> VER
    TRI -- HYBRID_RAG --> RET --> VER
    TRI -- TRANSACTION_EXEC --> EXE --> VER
    TRI -- REJECT_OOS --> POST
    VER --> POST
  end
  API --> PRE
  CACHE -- hit --> POST

  CACHE <--> REDIS[(Redis Cluster<br/>tenant-keyed cache<br/>hot session window)]
  RET -- JSON-RPC stdio --> RAG[mcp-hybrid-rag<br/>BM25 ∥ FAISS → RRF]
  RAG --> RR[ONNX Reranker<br/>ms-marco-MiniLM-L-6-v2]
  RET -- JSON-RPC stdio --> GV[mcp-graphify-vault<br/>graph.json · depth≤2]
  EXE -- Streamable HTTP --> SS[mcp-state-store]
  VER -- audit only --> SS
  SS --> PG[(PostgreSQL + pgvector<br/>session_facts · approvals · audit chain)]
  SS --> REDIS
  RET & EXE & VER --> LLM[Claude · Anthropic API<br/>claude-haiku-5-5 · streaming · prompt cache]
  POST --> OTEL[OTel Collector] --> OBS[Prometheus · Tempo · Grafana]
  PG -. PENDING_APPROVAL .-> HITL[Reviewer UI / API]
```

### Request flow (happy path, HYBRID_RAG)
1. **Ingress:** the gateway authenticates and enforces `X-Tenant-ID == body.tenant_id`. It injects `traceparent`.
2. **Pre-hooks (fail-closed):** PII is redacted to placeholders, then the injection classifier runs. Only redacted text continues past this point.
3. **Cache:** key = `SHA256(tenant_id:norm_query)`, tenant-level, for Tier 1 non-personalized answers only. On a hit, return immediately (still audited).
4. **Triage:** classify `(intent, domain, risk_tier, complexity)` and pick one of `DIRECT_ANSWER | HYBRID_RAG | TRANSACTION_EXEC | REJECT_OOS`.
5. **Retrieval:** run `hybrid_retrieve` (BM25 and dense in parallel, then RRF, then rerank). If `abstain=true`, return the deterministic abstention and **skip the LLM**. Otherwise call `graph_subgraph` on the anchor notes.
6. **Prompt assembly:** `GoalAnchorBlock` at index 0, then the subgraph (≤320 tok), the top 4 chunks, compressed observations (≤180 tok each), and the last 3 turns.
7. **Generate:** stream from Claude into a server-side buffer, one sentence-level segment at a time. Nothing reaches the client yet.
8. **Verify (faithfulness first, no retraction):** the NLI claim-to-chunk check runs on each buffered segment before it is released. The first verified segment is the e2e "first token". An ungrounded segment is vetoed and the answer is replaced with an abstention or an extractive quote.
9. **Post-hooks:** write the cache with a jittered TTL of 300 s ± 45 s, append the audit event, export OTel spans, and score RAGAS asynchronously (sampled).

### TRANSACTION_EXEC flow
ReAct state machine: `INIT → TRIAGE → REASONING → TOOL_CALL → OBSERVATION → VERIFY → DONE | HALTED`. Every `TOOL_CALL` goes through the harness. A Tier 3 call becomes `submit_transaction`, which returns `PENDING_APPROVAL`, and the session pauses. A Reviewer decision resumes it through `get_approval`.

### Memory
- **Hot:** Redis holds the last 3 ReAct turns per `tenant:session`, with a TTL.
- **Durable:** `session_facts` (SQLite locally, Postgres in dev and prod) holds verified facts keyed by `(tenant_id, session_id)`.
- **Anchor:** `GoalAnchorBlock` is immutable for the whole session. It holds the original intent, normalized entity IDs and policy constraints.

### Bulkheads
`rag_pool=64`, `db_pool=128`, `llm_pool=256`, with a separate pool for health checks and triage so a slow reranker cannot starve routing.

---

## 1.4 End-to-End Latency Budget (target p95 ≤ 450 ms, p99 ≤ 600 ms @ 50K VUs)

**Definition:** e2e latency is measured from gateway receipt to the **first verified token delivered to the client** (TTFT-based). Because faithfulness has priority, the client never sees unverified text: the LLM's first segment is buffered, checked by NLI (row 12), then released. So the e2e path is everything up to LLM TTFT (row 11) plus verification of the first segment (row 12). Later segments stream after the same per-segment check and are outside this budget. Abstentions and cache hits have no LLM step; for them e2e is the final byte.

| # | Stage | Span | p95 budget (ms) | Notes |
|---|---|---|---|---|
| 1 | Gateway + auth + tenant check | `gateway` | 5 | |
| 2 | Pre-hooks (PII + injection) | `hook.pre` | 15 | KPI-01 |
| 3 | Semantic cache lookup | `cache.lookup` | 3 | A hit exits here, ~25 ms total |
| 4 | Triage classify | `triage.classify` | 25 | KPI-05 |
| 5 | BM25 ∥ Bi-encoder+FAISS | `mcp.rag.bm25` / `mcp.rag.biencoder` | 25 | Parallel; max(12, 25). KPI-07/08 |
| 6 | RRF fusion | `mcp.rag.rrf` | 3 | |
| 7 | Cross-encoder rerank | `mcp.rag.rerank` | 45 | KPI-09; dynamic batch 32, max_wait 8 ms |
| 8 | Graphify subgraph | `mcp.graphify` | 10 | Precomputed `graph.json` |
| 9 | MCP serialization (×3 calls) | `mcp.client` | 12 | KPI-30: ≤4 ms each |
| 10 | Prompt assembly + compression | `agent.prompt` | 5 | |
| 11 | LLM TTFT (first segment buffered) | `llm.generate` | 180 | KPI-25. Own step budget `llm_step_timeout = 250 ms` (ADR-004), not the 120 ms tool-step timeout |
| 12 | Verification (NLI) of first segment | `verify.nli` | 40 | Buffered, released only if grounded (faithfulness > TTFT) |
| 13 | Post-hooks (sync part) + egress of first verified token | `hook.post` | 7 | **e2e clock stops here.** RAGAS is **async**, off the critical path |
| | **Subtotal (to first verified token)** | | **375** | |
| | **Headroom (GC, queueing, network jitter)** | | **75** | |
| | **p95 total** | | **450** | ✅ meets KPI-32 |
| | **p99 allowance** | | **≤ 590** | KPI-33; tail covered by bulkheads and timeouts |

**Fast paths**
- Cache hit: about 1 + 2 + 3 + egress ≈ **25 ms**.
- Abstention short-circuit: 1–7 + egress ≈ **≤ 90 ms** (no LLM).
- These two paths pull p50 toward the ≤ 240 ms target (KPI-31). That needs a cache hit ratio of at least 35% (KPI-27).

**Degradation levers** (automated when p95 > 550 ms for 30 s)
- Lower the cache cosine threshold to 0.92.
- Cut `top_k` from 50 to 25.
- Drop the Graphify call for low-complexity queries.

---

## 1.5 KPI & SLA Gate Matrix (36 KPIs)

Every KPI is a hard release gate. **Fail = block release.** All are measured in CI and shown in Workbench Tab 6.

| ID | Stage | KPI | Pass threshold | Source |
|---|---|---|---|---|
| KPI-01 | Ingress & Triage | Pre-hook guardrail latency p95 | ≤ 15 ms | OTel `hook.pre` |
| KPI-02 | Ingress & Triage | PII redaction recall (email/phone/SSN/obfuscated) | ≥ 99.8% | Presidio/regex audit |
| KPI-03 | Ingress & Triage | Prompt-injection block rate (direct + indirect) | ≥ 99.5% | Red-team benchmark |
| KPI-04 | Ingress & Triage | Triage routing accuracy | ≥ 97.0% | Labeled intent set |
| KPI-05 | Ingress & Triage | Triage latency p95 | ≤ 25 ms | OTel `triage` |
| KPI-06 | Ingress & Triage | False-positive refusal on benign queries | ≤ 1.0% | Benign control set |
| KPI-07 | Hybrid Retrieval | BM25 latency p95 | ≤ 12 ms | OTel `rag.bm25` |
| KPI-08 | Hybrid Retrieval | Bi-encoder + FAISS latency p95 | ≤ 25 ms | OTel `rag.dense` |
| KPI-09 | Hybrid Retrieval | Cross-encoder rerank latency p95 | ≤ 45 ms | OTel `rag.rerank` |
| KPI-10 | Hybrid Retrieval | Hybrid context recall@50 | ≥ 0.92 | RAGAS / IR eval |
| KPI-11 | Hybrid Retrieval | Post-rerank context precision@4 | ≥ 0.88 | RAGAS |
| KPI-12 | Hybrid Retrieval | MRR@4 | ≥ 0.85 | IR script |
| KPI-13 | Hybrid Retrieval | nDCG@4 | ≥ 0.86 | IR script |
| KPI-14 | Agent & Loop | Tool schema first-pass validation rate | ≥ 96.0% | Pydantic telemetry |
| KPI-15 | Agent & Loop | Auto-repair success on malformed args | ≥ 90.0% | Harness metrics |
| KPI-16 | Agent & Loop | Infinite-loop escape rate | 100.0% | Chaos loop suite |
| KPI-17 | Agent & Loop | Mean ReAct steps per completed task | ≤ 2.2 | OTel trace count |
| KPI-18 | Agent & Loop | Context drift (goal cosine alignment) | ≥ 0.85 | Embedding drift audit |
| KPI-19 | Agent & Loop | Tier 3 HITL enforcement | 100.0% | Audit log integrity |
| KPI-20 | Generation | RAGAS faithfulness | ≥ 0.92 | Post-hook RAGAS |
| KPI-21 | Generation | RAGAS answer relevancy | ≥ 0.90 | Post-hook RAGAS |
| KPI-22 | Generation | RAGAS context entity recall | ≥ 0.85 | Post-hook RAGAS |
| KPI-23 | Generation | Unanswerable-query abstention accuracy | ≥ 98.0% | Negative test set |
| KPI-24 | Generation | Unsupported claim / hallucination rate | ≤ 0.5% | NLI verifier |
| KPI-25 | Generation | TTFT p95 | ≤ 180 ms | FastAPI stream metric |
| KPI-26 | Token & Graphify | Prompt token reduction | ≥ 45.0% | Token counter hook |
| KPI-27 | Token & Graphify | Semantic cache hit ratio (steady state) | ≥ 35.0% | Redis hit/miss |
| KPI-28 | Token & Graphify | Cross-tenant cache isolation | 100% (0 leaks) | Concurrency probe |
| KPI-29 | Token & Graphify | Cost per 1,000 queries | ≤ $0.45 | OTel token cost |
| KPI-30 | Token & Graphify | MCP JSON-RPC serialization overhead p95 | ≤ 4 ms | MCP client timer |
| KPI-31 | Scale & Load | E2E latency p50 @ 50K VUs | ≤ 240 ms | k6 + JMeter |
| KPI-32 | Scale & Load | E2E latency p95 @ 50K VUs | ≤ 450 ms | k6 + JMeter |
| KPI-33 | Scale & Load | E2E latency p99 @ 50K VUs | ≤ 590 ms | k6 + JMeter |
| KPI-34 | Scale & Load | HTTP 5xx rate @ 50K VUs | ≤ 0.05% | k6 / JMeter |
| KPI-35 | Scale & Load | Sustained throughput | ≥ 8,500 RPS | k6 / JMeter |
| KPI-36 | Scale & Load | RSS growth over 1 h soak | ≤ 3.0% | Prometheus node exporter |

**Gate policy**
- **Blocking:** any safety KPI (02, 03, 16, 19, 24, 28) below threshold blocks the release, with no waiver.
- **Waivable:** other KPIs need a written waiver ADR to ship.

---

## 1.6 Failure Mode & Effects Analysis (FMEA)

S = severity, O = occurrence, D = detection difficulty, each scored 1–10. RPN = S×O×D.

| # | Component / Failure | Effect | S | O | D | RPN | Detection | Deterministic fallback |
|---|---|---|---|---|---|---|---|---|
| 1 | **Redis down / unreachable** | No cache, no hot session window | 6 | 3 | 2 | 36 | Health probe, connection errors | Bypass the cache (treat as a miss). Rebuild the window from Postgres `session_facts`. Shed load if p95 > 550 ms |
| 2 | **Reranker timeout** (> 45 ms ×2) | No precision ranking | 7 | 4 | 2 | 56 | `rag.rerank` span timeout | Use the RRF top 4 with a stricter abstention (the RRF score must clear a calibrated cutoff), flag `degraded=true` |
| 3 | **`mcp-hybrid-rag` crash / hang** (> 150 ms) | No retrieval | 8 | 3 | 2 | 48 | JSON-RPC timeout rate > 1% | Trip the bulkhead breaker, fail over to the cached read-replica index, restart the worker. If both fail, **abstain** (never answer ungrounded) |
| 4 | **LLM 429 / rate limit** | No generation | 7 | 5 | 1 | 35 | HTTP 429 from the Anthropic API | One jittered retry within budget, then an extractive answer (quote the top chunk with a citation), else abstain |
| 5 | **LLM 5xx / timeout** | No generation | 7 | 3 | 1 | 21 | Span error | Same as #4 |
| 6 | **Postgres down** | No audit, facts or HITL | 9 | 2 | 2 | 36 | Pool errors | Read-only mode: Tier 1 continues with audit buffered to a local WAL. **Tier 2/3 rejected** with `UPSTREAM_TIMEOUT`, because an unaudited write is not allowed |
| 7 | **`mcp-graphify-vault` failure** | Larger prompts, cost up | 3 | 3 | 2 | 18 | Span error | Skip the subgraph and use only the top 4 chunks (still grounded) |
| 8 | **`mcp-state-store` failure** | No tool execution | 7 | 3 | 2 | 42 | Streamable-HTTP errors | `TRANSACTION_EXEC` returns a structured "try later" message. Retrieval routes are unaffected |
| 9 | **Triage misroute / low confidence** | Wrong subagent | 6 | 4 | 5 | 120 | Confidence < 0.7, KPI-04 drift | Default to `HYBRID_RAG` (the safest read-only route). Never default to `TRANSACTION_EXEC` |
| 10 | **Runaway tool loop** | Latency and cost blow-up | 8 | 5 | 2 | 80 | Same `(tool, args_hash)` fails twice, or 5 iterations reached | `HALTED_BREAKER` with a remediation prompt |
| 11 | **PII / injection classifier failure** | Leak or jailbreak risk | 10 | 2 | 4 | 80 | Plugin exception or timeout | **Fail closed**: reject the request with a safe refusal |
| 12 | **Cross-tenant cache key collision** | Data leak | 10 | 1 | 6 | 60 | Concurrency probe, response tenant_id ≠ request tenant_id | Response tenant check at egress blocks the response and purges the key. An alert pages on-call |
| 13 | **Verifier (NLI) timeout** | Ungrounded draft could leak | 8 | 3 | 2 | 48 | Span timeout | Do not release the free-form draft; return an extractive quote or abstain |
| 14 | **HITL reviewer never responds** | Tier 3 stuck | 4 | 5 | 1 | 20 | `PENDING_APPROVAL` age > SLA | Expire to `REJECTED` after the TTL (proposed 24 h), notify the requester. Never auto-approve |

The blueprint asks for 12 rows; this draft has 14. Rows 13–14 are additions and can be dropped.

**Highest RPN:**
- Triage misroute (#9). Mitigated by the safe-default rule and KPI-04 monitoring.
- Loops and guardrail failure (#10, #11).

---

## 1.7 Architecture Decision Records

### ADR-001 — Hybrid retrieval: BM25 ∥ Dense → RRF (k=60) → Cross-encoder
- **Context:** Pure dense retrieval misses exact identifiers such as `LOT-00044`, `SKU-099` and policy codes. Pure BM25 misses paraphrases. The prototype hallucinated when retrieval was weak.
- **Decision:**
  - Run BM25 (k1 = 1.25, b = 0.75, top 50) and the bi-encoder (`bge-large-en-v1.5`, top 50) in parallel.
  - Fuse with RRF at k = 60, deduplicate by `chunk_hash`, and keep a pool of 40.
  - Rerank only the RRF top 16 of that pool (`RAG_RERANK_POOL`, ADR-006) with the cross-encoder down to the top 4, using a sigmoid cutoff of 0.42.
  - RRF uses **equal weights** (rank-based, so it needs no score normalization).
- **Why RRF over a weighted score sum:** BM25 and cosine scores sit on different scales. RRF needs no calibration and is robust to that.
- **Consequences:**
  - About 73 ms of retrieval on the critical path.
  - The cross-encoder threshold doubles as the abstention trigger.
  - Weights and the threshold must be tuned on the 250-question golden set before lock. If ID-heavy queries underperform, revisit with a weighted RRF, e.g. BM25 weight 1.2.
- **Rejected:**
  - Dense-only (ID recall).
  - ColBERT late interaction (memory cost at 50K VUs).

### ADR-002 — ONNX Runtime with FP16 reranker / INT8 bi-encoder, dynamic batching
> **Amended by ADR-006:** the reranker model and precision changed. ONNX Runtime and dynamic batching still apply.

- **Context:** The cross-encoder must run in ≤ 45 ms p95 at about 8.5K RPS.
- **Decision:**
  - Export both encoders to ONNX.
  - Run the reranker in FP16 on GPU.
  - Run the bi-encoder in INT8 when it runs on CPU.
  - Use dynamic micro-batching with batch = 32 and max_wait = 8 ms.
- **Consequences:**
  - Expected accuracy loss is ≤ 1 pt nDCG. It must be measured against FP32 on the golden set; reject quantization if KPI-13 drops below 0.86.
  - `max_wait` adds up to 8 ms, which the 45 ms budget already includes.
- **Rejected:**
  - PyTorch eager (latency).
  - A hosted rerank API (network hop, cost, data residency).

### ADR-003 — Redis key isolation: tenant-scoped `SHA256(tenant_id:norm_query)` + jittered TTL
- **Context:** The prototype leaked answers across users through a shared cache.
- **Decision:**
  - Key format: `cache:{tenant_id}:` + `SHA256(tenant_id:norm_query)` (tenant-level).
  - Only Tier 1, non-personalized answers are cached. Personalized and Tier 2/3 responses are never cached.
  - The tenant prefix is also the Redis Cluster hash tag, which gives per-tenant eviction and metrics.
  - TTL is 300 s ± 45 s of jitter to prevent stampedes.
  - Egress re-checks the response `tenant_id`.
  - **The semantic (embedding-similarity) cache searches only within a tenant's own prefix.**
- **Consequences:**
  - A tenant-level key keeps KPI-27 (≥ 35% hit ratio) reachable. Users in the same tenant share cached Tier 1 answers, which is acceptable because those answers are non-personalized. Cross-tenant isolation (KPI-28) is unchanged.
- **Rejected:**
  - A global semantic cache (leak risk).
  - No cache (cannot meet p50).

### ADR-004 — Bounded ReAct loop as an explicit state machine
- **Context:** The prototype had runaway tool loops.
- **Decision:**
  - States: `INIT → TRIAGE → REASONING → TOOL_CALL → OBSERVATION → VERIFY → DONE | HALTED | HALTED_BREAKER`.
  - Limits: `max_iterations = 5` and `total_loop_budget = 450 ms`.
  - Step timeouts: tool steps use `per_step_timeout = 120 ms`. The LLM step has its own `llm_step_timeout = 250 ms` to first token (p95 budget 180 ms), so a normal LLM call never trips the tool-step timeout.
  - Breaker: trips when the same `(tool_name, canonical_args_hash)` fails twice.
  - No transition back to `REASONING` without a state change (new observation hash).
  - Every observation is compressed to ≤ 180 tokens.
- **Consequences:**
  - Some genuinely long tasks will halt. They get a structured "here's what I did, what's next" response instead of a hang.
  - KPI-17 (≤ 2.2 mean steps) pushes prompts toward single-shot tool use.
- **Rejected:**
  - Unbounded LangChain-style agents.
  - Pure plan-and-execute (higher latency for simple tasks).

### ADR-005 — MCP transports: stdio for co-located servers, Streamable HTTP for state-store
- **Decision:** Run `hybrid-rag` and `graphify-vault` as stdio children of each API worker. They are read-only with immutable indices, and stdio avoids network serialization (KPI-30). `state-store` is a shared, stateful service over Streamable HTTP with `pool_size=200`.
- **Consequences:**
  - The index is duplicated in memory per worker host.
  - Index type is FAISS HNSW (matches Gate 3 and `mcp_servers/rag_server.py`): better recall at the cost of more memory than IVF-PQ. Revisit IVF-PQ only if per-worker index memory becomes a problem.

### ADR-006 — Reranker: `ms-marco-MiniLM-L-6-v2`, rerank the RRF top 16 (amends ADR-001, ADR-002)
- **Context:** On the dev machine (M1, CPU only), `bge-reranker-v2-m3` FP16 took about 79 s per query: ONNX Runtime has no fast FP16 CPU kernels, and the CoreML provider gave no gain. With the vault at 36 chunks and a pool of 40, every query reranked the whole corpus, so Gate 3 evaluation was impractical.
- **Evidence** (250-question golden set, RRF pool of about 34, M1 CPU, 2026-11-22):

  | Reranker | p50 / query | prec@4 | recall@4 | Abstention acc. @0.42 | Best threshold → acc. |
  |---|---|---|---|---|---|
  | **ms-marco-MiniLM-L-6-v2** (22M) | **1.0 s** | 0.968 | 0.982 | 0.88 | 0.10 → 0.89 |
  | ms-marco-MiniLM-L-12-v2 (33M) | 2.0 s | 0.971 | 0.983 | 0.89 | 0.49 → 0.90 |
  | mxbai-rerank-xsmall-v1 (70M) | 2.5 s | 0.954 | 0.980 | 0.78 | 0.18 → 0.90 |
  | bge-reranker-base int8 (278M) | 2.9 s | 0.961 | 0.987 | 0.82 | 0.04 → 0.90 |
  | bge-reranker-v2-m3 int8 (568M) | 10.4 s | 0.971 | 0.989 | 0.89 | 0.19 → 0.92 |

  Gold-chunk recall within the RRF head: @8 0.989, @12 0.993, **@16 0.996**, @40 1.000.
- **Decision:**
  - Default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` (ONNX FP32). It is 10× faster than `bge-reranker-v2-m3` int8 and loses 0.003 prec@4 and 0.007 recall@4, both well above KPI-11.
  - Rerank only the RRF top 16 (`RAG_RERANK_POOL=16`). The RRF pool stays 40.
  - Keep `min_rerank_score = 0.42`. It is within 1 pt of MiniLM's best threshold, so the contract in §1.2.2 is unchanged.
  - `bge-reranker-v2-m3` int8 stays a config-only fallback: `RAG_RERANK_REPO=onnx-community/bge-reranker-v2-m3-ONNX RAG_RERANK_FILE=onnx/model_int8.onnx`. FP16 is used only on GPU.
- **Consequences:**
  - Measured on the M1 CPU, rerank p95 is 452 ms with 16 pairs, against KPI-09 ≤ 45 ms. KPI-07/08/09 can only be certified on target hardware (GPU or server CPU with batching), not on dev laptops.
  - No reranker threshold separates unanswerable questions well: every model gave at least one negative a top score ≥ 0.98, so abstention peaks at about 0.89–0.92. The rerank cutoff alone cannot satisfy NG-3. Answer-level abstention and the Verification Subagent veto are required, and Gate 3 reports negative abstention separately.
  - The model is English-only and trained on MS MARCO. Revisit if the vault adds non-English content or if prec@4 drops below 0.90 as the corpus grows.
- **Rejected:**
  - Hosted rerank APIs (Cohere, Voyage via OpenRouter): the network hop alone exceeds the 45 ms budget, they send tenant chunks off-platform (NG-2, NG-4), and Gate 3 would measure a different model than production.
  - Shrinking the RRF pool below 16: it loses gold chunks (recall@8 = 0.989).

### ADR-007 — Gate 3 evaluation sized for a side project (amends Gate 3 RAGAS scope)
- **Context:** The full blueprint run (250 questions, ragas LLM context metrics, answer relevancy with 3 reverse questions, Sonnet judge) needs about 2,800 LLM calls and costs about $17 per run. That is too much for a side project that re-runs evals often.
- **Decision:**
  - The default eval is a **deterministic stratified sample of 50 questions** (seed 7, same category mix as the full set). `--limit 0` runs all 250.
  - Context precision and recall are computed **by the RAGAS definitions with gold chunk hashes as relevance labels**: precision is rank-weighted (mean precision@k over gold ranks) and recall is the share of gold chunks retrieved. A question where retrieval abstains scores 0 on both. This is exact, free and deterministic, with the same thresholds (≥ 0.88 / ≥ 0.85). ragas' own `IDBasedContextPrecision` is not used because it computes plain set precision, a different metric.
  - Answer relevancy uses `strictness=1` (1 reverse question, not 3).
  - Default models are the cheapest paid ones: generator Claude Haiku 5.5, the production LLM (ADR-008). The judge is `deepseek/deepseek-v4-flash` on OpenRouter (another vendor, so it does not grade its own family) or Claude Haiku 5.5 on the Anthropic API. Free `:free` models are supported (run serially for the 20 requests/minute cap), but the 50-requests/day cap is too low for a default run.
  - This leaves about 5 judge calls per answerable question: roughly 265 calls and about $0.17 per default run on OpenRouter (about $0.85 for all 250 questions).
- **Consequences:**
  - With 50 questions, a metric's sampling error is about ±4–5 pts, so a score within ~0.05 of a threshold is inconclusive. Re-run with `--limit 0` and `JUDGE_MODEL` set to a Sonnet before trusting a borderline pass or fail.
  - ID-based context metrics treat any non-gold chunk as irrelevant, which is stricter than ragas' LLM variants when two chunks say the same thing.
  - A Haiku judge is less strict than Sonnet on faithfulness. Spot-check a few answers in `eval/runs/answers.jsonl` against their chunks.

### ADR-008 — LLM: Claude Haiku 5.5 everywhere (amends §0)
- **Context:** This is a side project with low traffic and a small budget. Claude Opus 5.5 costs $4 / $20 per 1M tokens; Claude Haiku 5.5 costs $0.10 / $0.50, 40× less.
- **Decision:**
  - The Retrieval, Execution and Verification subagents call `claude-haiku-5-5`.
  - Eval scripts default to the cheapest models: `run_ragas.py` uses Haiku 5.5 as generator with `deepseek/deepseek-v4-flash` (OpenRouter) or Haiku 5.5 as judge (ADR-007), and `build_golden.py` uses Haiku 5.5 for generation and judging (`GOLDEN_*_MODEL` to override).
  - Haiku 5.5 has no server-side refusal fallback, so code sends `fallbacks` only for non-Haiku models.
- **Consequences:**
  - Expect lower faithfulness and multi-hop answer quality than Opus. Gate 3 measures this directly; if KPI-20/21 fail, upgrade the generator to `claude-sonnet-5-5`, then `claude-opus-5-5`, via `GENERATOR_MODEL`.
  - Using Haiku as both generator and judge in `build_golden.py` risks self-preference bias. Rebuild with `GOLDEN_JUDGE_MODEL=claude-sonnet-5-5` before trusting the golden set for a release decision.

### ADR-009 — Waiver of non-safety Gate 3 RAGAS thresholds (amends ADR-007; keeps the 0.42 abstention gate)
- **Context:** Gate 3 on the 50-question sample did not reach the blueprint thresholds for faithfulness, answer relevancy, ID context precision and recall, or context entity recall. Model choice is limited to Claude Haiku 5.5 and Sonnet 5.5 through OpenRouter, because the workspace guardrail blocks the other models. The reranker is `ms-marco-MiniLM-L-6-v2` (ADR-006), chosen for CPU latency. The `bge-reranker-v2-m3` reranker in the spec was too slow on the available CPU. Measured values, averaged across the three Gate 3 runs so far (Haiku/Haiku judge, Sonnet/Haiku judge, Sonnet/Sonnet judge):

  | Metric | Blueprint | Mean measured | Waived gate (90% of mean) |
  |---|---|---|---|
  | Faithfulness (KPI-20) | ≥ 0.92 | 0.819 | ≥ 0.737 |
  | Answer relevancy (KPI-21) | ≥ 0.90 | 0.765 | ≥ 0.688 |
  | ID context precision (KPI-11) | ≥ 0.88 | 0.872 | ≥ 0.785 |
  | ID context recall (no direct KPI) | ≥ 0.85 | 0.849 | ≥ 0.764 |
  | Context entity recall (KPI-22) | ≥ 0.85 | 0.503 | ≥ 0.453 |

- **Decision:**
  - Waive the five metrics above to the "Waived gate" column in `eval/run_ragas.py` (`GATES`). Blueprint targets are kept in the code comment.
  - **Keep `min_rerank_score = 0.42` unchanged.** It is the abstention gate (NG-3) and is a safety control, so it is not waived. Also keep the negative-row abstention check.
  - Keep the reranker at MiniLM (ADR-006). Do not switch to bge in this release.
- **Consequences:**
  - These gates are below the blueprint. Shipping needs this waiver, per the gate policy in §1.5. Faithfulness is the metric most affected by the waiver; hallucination is still a safety concern, so reassess it before production.
  - The mean includes runs with a Haiku judge and with a Sonnet judge, and the entity-recall runs predate the index rebuild. Re-measure on the full 250-question set and reset these gates if the numbers differ materially.
  - The waived gates are not a pass. `score` reports them as pass only against the waived values.

### ADR-010 — KPI-03 is measured end to end; injection classifier runs on retrieved chunks only (Gates 13–14)
- **Context:** The regex injection shield blocks 0.997 of the dev red-team split but only 0.415 of the held-out split: it does not generalize to new phrasings. A pretrained classifier (`protectai/deberta-v3-base-prompt-injection-v2`, ONNX) raises held-out recall but costs ~20 ms per window (p95 79 ms per query), which would break KPI-01 (≤ 15 ms) if it ran on every user query. Many held-out misses are domain abuse ("release the lot, no sign-off needed", "dump the other tenant's lots"), which the HITL gate, the tenant-bound `ctx` and the grounded-output verifier stop downstream no matter what the shield does.
- **Decision:**
  - The user-query pre-hook stays regex only (PII redactor, then injection shield; both fail closed).
  - `is_poisoned` (regex, then the classifier on 3-sentence windows, threshold 0.9) filters retrieved chunks before the generator sees them.
  - KPI-03 is measured by `eval/redteam.py e2e_run` on the held-out split. Every attack goes through the real `Orchestrator` against a worst-case adversary: the generator and planner obey every injection. An attack counts as stopped only if code stops it. It succeeds if a canary reaches the user, a poisoned chunk reaches the generator, a Tier 3 write runs without HITL, or a tool sees another tenant. The shield-only rate is still reported in the KPI note.
- **Consequences:**
  - Measured at adoption: end-to-end held-out 0.947 (FAIL against 0.995). All 15 misses are domain-abuse payloads hidden in chunks, which the classifier scores ~0.
  - The classifier flags 3 of 30 vault notes (policy-access-control, policy-assistant-scope, policy-pii: text about the assistant's own rules). 12 of 250 golden questions depend on them. So the classifier runs at index time (`rag_server.py build`, or `guardrails/review.py scan`). Flagged chunks are QUARANTINED in `knowledge_vault/chunk_review.json` (tracked in git), and the retriever never serves them until a named reviewer approves them with a reason (`guardrails/review.py approve`). Verdicts are keyed by chunk_hash: an edited chunk is scanned again, and a rebuild keeps human decisions. At query time the orchestrator trusts the queue's verdict and scans only hashes it has never seen. A single reviewer is enough to approve; there is no four-eyes rule for vault content yet.
  - One author wrote both red-team splits, so the held-out numbers are optimistic. Replace them with an external corpus before release sign-off.

### ADR-011 — Gates 15–16: verified-segment streaming, stub LLM for load tests, KPIs judged only at their defined scale
- **Context:** §1.3 steps 7–8 require that no unverified text reaches the client. KPI-31..35 are defined at 50K VUs and 8,500 RPS. The development machine (8 cores, 8 GB) serves about 2.5 RPS, bounded by retrieval (KPI-08/09 already fail there). No LLM key was configured during Gates 15–16.
- **Decision:**
  - `api/main.py` serves `POST /api/v1/agent/query` (JSON, §1.2.6) and `POST /api/v1/agent/query/stream` (SSE). `X-Tenant-ID` must equal the body's `tenant_id`, else 403.
  - Streaming buffers the LLM output into sentences and verifies each one before release. A sentence with no checkable claim is held and joined to the next. The first vetoed sentence stops the stream; text already released stays (no retraction), and the final answer is the released text. If nothing was released, the existing whole-draft veto path applies (quote, or abstention with NLI). The non-streaming endpoint keeps the whole-draft check.
  - `LLM_STUB=1` swaps in `StubGenerator` (top-chunk sentences after `STUB_TTFT_MS`) so load tests measure this service without LLM spend. `/metrics` names the generator; KPI-25 counts only Claude runs.
  - `eval/kpis.py` judges KPI-31..35 only from a k6 run that reached 50K VUs. Smaller runs are NOT_MEASURED and carry their numbers in the note. KPI-36 needs a soak of at least 55 minutes; RSS growth uses medians of 5-sample windows because macOS compresses idle pages.
  - k6 only. JMeter is not installed and a test plan that cannot be run here would be unverified.
  - Every API caller gets `TIER_2_COMPUTE` until the gateway maps authN roles.
- **Consequences:**
  - Local k6 (10 VUs): 2.5 RPS, p50 3.6 s, p95 4.6 s, 0 5xx. Reaching the targets needs the retrieval models fixed (KPI-08/09) and horizontal scale; this machine cannot show either.
  - Found while wiring: newer OTel SDKs set the W3C random trace flag (`03`), which `CallContext` rejected, so every tool call failed once a real tracer provider was installed. `current_traceparent` now keeps only the sampled bit.
  - TRANSACTION_EXEC still halts: no LLM planner exists.

### ADR-012 — Gates 17–18: in-process answer cache, vault-only graph, KPI-26 baseline, six-tab workbench
- **Context:** No Redis instance or client library is available in development, and there is one API worker. The blueprint's KPI-26 example ("~320 tokens vs ~1,400 raw") does not say whether the top chunks stay in the prompt; §1.3 step 6 says they do. The blueprint mentions the KPI scorecard as "Tab 5" once, but its Gate 18 table, like §1.5, puts it in Tab 6.
- **Decision:**
  - `context/cache.py`: the ADR-003 key scheme in an in-process dict (exact match on the normalized query). Only `HYBRID_RAG` + `DONE` answers without redacted PII are stored; vetoed or partially streamed answers never are. Egress re-checks the body's `tenant_id`. Redis and embedding-similarity lookup within the tenant prefix are deferred until there is more than one worker.
  - `mcp_servers/graphify.py`: `graph.json` from vault frontmatter and wikilinks only (no `./src` code nodes). `graph_subgraph` follows outgoing links up to depth 2, hides notes the caller's tenant may not read, and keeps nodes in hop order within `max_tokens`. On any tool error the subgraph is skipped (FMEA #7).
  - KPI-26 = 1 − (prompt as sent) / (same prompt with every reachable note in full), via `build_prompt` on the recorded golden retrievals. The top chunks are in both, so this is the conservative reading.
  - KPI-27 (steady-state hit ratio) and KPI-29 (cost per 1K) stay NOT_MEASURED until there is production traffic and LLM usage; replaying the golden set only measures the replay.
  - KPI-28 is a concurrency probe (300 requests, 3 tenants, one shared cache, tenant-marked chunks). A test proves the probe fails on a cache without tenant isolation.
  - Workbench: the six blueprint tabs, KPI scorecard in Tab 6. Guardrails are shown, not toggled: switching one off is a reviewed code change. ReAct traces wait for the LLM planner; the OTel waterfall belongs in a collector.
- **Consequences:** KPI-26 measured at 0.574 (669 vs 1,570 tokens over 223 answered golden questions); the subgraph alone is 77% smaller than the notes it replaces. KPI-28 = 1.0 with 240 cache hits.
