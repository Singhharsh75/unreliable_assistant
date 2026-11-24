# Gates 7–12: Recommended Fixes and Next Steps

**Date:** 2026-11-24 · **Scope:** Gates 7–12 (subagent topology, continuous RAGAS, reliability patterns, observability, incident response, KPI matrix) · **Test suite:** 139 passed · **Source of KPI numbers:** `eval/runs/kpi_report.json` (live run, `uv run eval/kpis.py --live`)

## 1. Summary

Gates 7–12 are built and reviewed, but the system is **not release-ready**.

| Status | Count | KPIs |
|---|---|---|
| PASS | 12 | 04, 05, 06, 07, 10, 11, 12, 13, 15, 16, 19, 30 |
| FAIL | 4 | 08, 09, 23, 24 |
| WAIVED (ADR-009) | 3 | 20, 21, 22 |
| NOT_MEASURED | 17 | 01, 02, 03, 14, 17, 18, 25, 26, 27, 28, 29, 31–36 |

**Release blockers** (safety KPIs that cannot be waived, design.md §1.5): KPI-24 fails. KPI-02, KPI-03 and KPI-28 are not measured yet.

## 2. Failing KPIs and recommended fixes

### KPI-24: Unsupported claim rate (safety, blocking). Measured 35.5%, target ≤ 0.5%

**Cause:** the Verification Subagent (`triage/verifier.py`) checks claims by word overlap (≥ 80% of stemmed content words, plus exact numbers and IDs) rather than by meaning. It rejects paraphrases that are correct, so 35.5% of the generator's draft sentences are flagged. Rejected drafts are never released: they are replaced by a verbatim quote of the top chunk. Users therefore never see unsupported text, but about a third of answers become quotes instead of answers.

**Recommended fixes, in order:**
1. Replace the lexical check with an ONNX NLI cross-encoder, for example a DeBERTa-v3 MNLI model, run through the existing `onnxruntime` and `rag_server._session` helpers. Keep the exact-match rule for numbers and IDs as a hard precondition. The latency budget for this is `verify.nli` ≤ 40 ms (design.md §1.4, row 12).
2. Until then, measure how the lexical check compares with the RAGAS judge. Draft faithfulness from RAGAS is 0.853, which implies about 15% unsupported claims, against 35.5% from the verifier. The gap of roughly 20 points is the verifier's false-veto rate.
3. Make the generator prompt (`eval/run_ragas.py` `SYSTEM`) ask for answers phrased close to the chunk wording, with one claim per sentence.
4. Re-run `eval/run_ragas.py answer` and then `eval/kpis.py --live`.

### KPI-23: Unanswerable-query abstention. Measured 94.6%, target ≥ 98%

**Cause:** ADR-006 already found that no reranker cutoff separates unanswerable questions well, and some negatives score ≥ 0.98. The generator still answers 2 of the 37 negative questions.

**Recommended fixes:**
1. Inspect the failing negative rows in `eval/runs/answers.jsonl` (`category == "negative"`, `response != ABSTAIN_TEXT`).
2. Once the NLI verifier exists, add an answer-level check: an answer whose claims are not entailed becomes an abstention, not a quote. Today a vetoed draft becomes a quote, which counts as answering.
3. Add a "does the context answer the question?" step before generation, either the NLI model on (question, top chunk) or a short LLM call.
4. Do not lower the 0.42 cutoff to fix this. ADR-009 keeps it as the abstention gate.

### KPI-08 and KPI-09: Retrieval stage latency. Bi-encoder 105 ms (target ≤ 25 ms); rerank 460 ms (target ≤ 45 ms)

**Cause:** these were measured on the CPU-only M1 development machine. `bge-large-en-v1.5` (335M parameters) and the cross-encoder over a pool of 16 are too slow for the budget on this hardware.

**Recommended fixes:**
1. Measure on the target hardware first. Run `uv run mcp_servers/rag_server.py bench` on the deployment instance class.
2. Bi-encoder: use the INT8-quantized ONNX model (ADR-002 already specifies INT8), or move to `bge-small-en-v1.5` or `bge-base-en-v1.5` and re-check KPI-10 and KPI-11.
3. Reranker: lower `RAG_RERANK_POOL` from 16 to 8 and check precision@4 (KPI-11 is 0.925 today, so there is headroom). Batch reranker calls across concurrent requests (ADR-002 dynamic batching, max wait 8 ms).
4. If neither meets the budget, write an ADR that waives KPI-08 and KPI-09 for CPU-only deployments, as ADR-009 did for RAGAS.

## 3. Waived KPIs (ADR-009): re-measure before production

| KPI | Measured | Blueprint target | Waived gate |
|---|---|---|---|
| KPI-20 Faithfulness | 0.853 | ≥ 0.92 | ≥ 0.737 |
| KPI-21 Answer relevancy | 0.771 | ≥ 0.90 | ≥ 0.688 |
| KPI-22 Context entity recall | 0.587 | ≥ 0.85 | ≥ 0.453 |

ADR-009 asks for faithfulness to be reassessed before production. The NLI verifier above should raise released faithfulness directly.

**Conflict:** the Gate 11 "hallucination surge" runbook fires when online faithfulness is < 0.90 over 5 minutes, and the measured mean is 0.853. Once online RAGAS is enabled, the runbook will fire almost at once and switch the system to quote-only mode. Either raise faithfulness above 0.90 first, or set `Thresholds.faithfulness_floor` to a waived value in an ADR.

## 4. Known gaps and shortcuts in the Gate 7–12 code

| # | Gap | Where | Risk | Fix |
|---|---|---|---|---|
| 1 | No LLM planner. The `TRANSACTION_EXEC` route needs a planner to be injected, and only scripted test planners exist | `triage/orchestrator.py` | Transactions can't run in production; KPI-17 and KPI-18 can't be measured | Build a Claude planner that emits `ToolAction` or `FinalAnswer` from `ReActPrompt.messages()` |
| 2 | Approved Tier 3 actions are recorded but never applied | `mcp_servers/state_store.py` (`shortcut:` comment) | An approved release or quarantine has no effect | Add the worker path design.md §1.2.4 describes, idempotent, with status `APPLIED` and an audit event |
| 3 | Audit events are lost when the db pool is open or the database is down | `triage/orchestrator.py` `_audit` | Gaps in a regulated audit trail | Local write-ahead buffer that replays to the store (FMEA #6); reject Tier 2 and 3 while it is in use |
| 4 | The idempotency cache and its locks grow without limit (pre-existing, Gate 4) | `harness/tool_harness.py` `_results`, `_locks` | Memory growth on a long-running server; affects KPI-36 | Cache with a TTL or LRU limit; drop each lock when its call finishes |
| 5 | The runbook's cache, Redis and read-replica settings change config but do nothing | `observability/incidents.py` (`shortcut:` comment) | The latency and cache-leak runbooks are only partly effective | Wire them when the semantic cache (Gate 17) and a replica index exist |
| 6 | The state store runs in-process on SQLite, not as a Streamable-HTTP MCP server on Postgres | `mcp_servers/state_store.py` | Not deployable as ADR-005 specifies | Wrap it in an MCP server as `rag_server.build_server` does, and port the schema to Postgres |
| 7 | The graph tool (`graph_subgraph`) is skipped in the retrieval flow | `triage/orchestrator.py` `_rag` | Larger prompts | Gate 17 |
| 8 | The real `RagasScorer` and `ClaudeGenerator` are constructed in tests but never called | `eval/ragas_auditor.py`, `triage/orchestrator.py` | An integration bug could surface only in production | One smoke test with an API key in CI |
| 9 | The production generator imports the eval module (`eval.run_ragas`) | `ClaudeGenerator` | The runtime depends on eval-only code | Move `SYSTEM`, `complete` and `make_client` into a shared module |
| 10 | The KPI-15 repair suite still fails "#44" and "L0T-00044" (a zero for the letter O) | `harness/tool_harness.py` `_REPAIR_RULES` | 2 of 20 malformed forms are not repaired | Field-aware repair: a bare number in a `lot_id` field becomes `LOT-` plus the padded number |

## 5. Next steps

### 5.1 Fix before continuing (blocks release)
1. **NLI verifier** (KPI-24, and helps KPI-23 and KPI-20).
2. **Answer-level abstention** (KPI-23).
3. **Measure latency on the target hardware**, then optimize the models or write a waiver ADR (KPI-08 and KPI-09).
4. **LLM planner** (gap 1). Without it the transaction route doesn't work.

### 5.2 Remaining gates
| Gate | Work | KPIs it unblocks |
|---|---|---|
| 13–14 | Pre-hooks: PII redactor and injection shield, both failing closed; red-team benchmark | 01, **02**, **03** (safety) |
| 15–16 | FastAPI streaming endpoint, k6 and JMeter at 50K users, 1-hour soak | 25, 31–36 |
| 17 | Graphify vault, semantic cache (Redis, tenant-keyed), token counter | 26, 27, **28** (safety), 29 |
| 18 | Streamlit workbench: the KPI Matrix tab exists and must move to its blueprint position (Tab 5; design.md says Tab 6) | — |

### 5.3 Housekeeping
- Commit Gates 7–12. Nothing is committed yet.
- Add `uv run eval/kpis.py` to CI. It exits 1 on any FAIL; add `--release` for release builds.
- Write an ADR that records the decisions taken in Gates 7–12:
  - lexical verifier as a stopgap;
  - a vetoed draft becomes a quote;
  - bulkheads trip only on infrastructure errors;
  - 24-hour HITL expiry and the four-eyes rule.
- Resolve the Tab 5 versus Tab 6 conflict between the blueprint and design.md §1.5.

## 6. How to reproduce

```bash
uv run pytest -q                      # 139 tests
uv run eval/kpis.py --live            # writes eval/runs/kpi_report.json; exit 1 on FAIL
uv run streamlit run app.py           # KPI Matrix tab reads the report
```
