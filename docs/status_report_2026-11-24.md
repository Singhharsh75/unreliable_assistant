# Unreliable Assistant: Status Report, 2026-11-24 (updated)

**Scope:** state of the build against the 18-gate blueprint (`Unreliable_Assistant_MidCheck_Enterprise_Blueprint_v2.docx`).
**Sources:** `eval/runs/kpi_report.json` (`uv run eval/kpis.py --live`) and `design.md` (ADR-001 to ADR-022).
**Test suite:** 226 passed (`uv run pytest`).

## 1. Summary

The work that can be done in this environment is done, except the KPI-24 run through the API, which needs an LLM key. The system is **not release-ready**: 8 KPIs are deferred, 7 are waived, and the sign-off conditions in section 4 are open.

| Status | Count | KPIs |
|---|---|---|
| PASS | 21 | the rest |
| FAIL | 0 | none |
| WAIVED | 7 | 08, 09 (ADR-020), 17, 18 (ADR-019), 20, 21, 22 (ADR-009) |
| DEFERRED | 8 | 25, 27, 29, 31 to 35 (ADR-021) |
| NOT_MEASURED | 0 | none |

Release blockers (safety KPIs 02, 03, 16, 19, 24, 28): none open. A waiver is not a pass: those seven KPIs miss the blueprint target.

## 2. Waived KPIs

| KPI | Measured | Target | Waived gate | Why |
|---|---|---|---|---|
| KPI-08 bi-encoder p95 | 124 ms | 25 ms | 1,000 ms, no GPU only | CPU-only machine; varies run to run (116 to 483 ms seen) |
| KPI-09 rerank p95 | 508 ms | 45 ms | 2,000 ms, no GPU only | Same (up to 1,007 ms seen) |
| KPI-17 mean ReAct steps | 2.54 | 2.2 | 3.0 | Keeps the read-before-write rule |
| KPI-18 goal alignment AUC | 0.808 | 0.85 | 0.75 | Redefined metric; the gate was chosen after seeing the result |
| KPI-20/21/22 RAGAS | 0.853 / 0.771 / 0.587 | 0.92 / 0.90 / 0.85 | 0.737 / 0.688 / 0.453 | ADR-009 |

## 3. Closed in this update

- **Payload schemas** per action, so junk requests no longer become PENDING_APPROVAL records. `mcp-state-store` re-signed.
- **Idempotency cache** bounded, locks dropped after use.
- **ReAct trace** in the audit log and workbench Tab 1.
- **Plugin registry** and `mcp_config.json`; the config cannot weaken a guardrail.
- **Incident runbooks** run in the API and change behaviour (aggressive cache, per-user keys, replica failover, worker restart). The blueprint's 0.92 cosine match was replaced because it matches opposite questions (ADR-018).
- **JMeter** formally replaced by k6 (ADR-022).
- **`graph_subgraph`** was already in the retrieval flow; a test now proves it.
- **KPI-24 judge:** an unreadable judge reply no longer counts as a clean answer.

## 4. Open before release

| Item | Needs |
|---|---|
| KPI-24 through the API (`eval/released_judge.py --api URL`) | LLM key |
| KPI-24 with a judge from another model family; more claims than 548 | LLM key |
| KPI-14, 17, 18 again: recorded before the payload schemas changed the planner's tool catalog | LLM key |
| KPI-25 | Streamed Claude run |
| KPI-20 to 22 re-measured | LLM key |
| KPI-08, 09 on target hardware | GPU or target host |
| KPI-31 to 35 | Distributed 50K-user environment |
| KPI-27, 29 | Production traffic |
| KPI-03 sign-off | Red-team corpus by another author |
| Roles: every caller is `TIER_2_COMPUTE`; incident `resolve()` has no endpoint | Auth gateway |
| Postgres/pgvector, Redis Cluster, NGINX/Envoy, vLLM, Prometheus/Tempo/Grafana | Infrastructure |

Two earlier gaps remain: an approved Tier 3 request is recorded but never applied, and audit events are lost while the db pool is open (`docs/gates_7-12_fixes_and_next_steps.md`, section 4). 37 of 198 answerable drafts (19%) are vetoed to abstention, which lowers answer coverage (ADR-016).
