"""Developer workbench (Gate 18): the six blueprint tabs. Heavy work (models, benchmarks) runs only behind a button.

  uv run streamlit run app.py
"""
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
RUNS = ROOT / "eval" / "runs"
DIAGRAM = ROOT / "docs" / "workflow_orchestration.png"
STATE_DB = os.getenv("STATE_DB", str(ROOT / "var" / "state.db"))  # the API's store: approvals, audit, session facts


def load(name: str):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def retriever():
    from mcp_servers.rag_server import Retriever
    return Retriever()


@st.cache_resource
def vault():
    from mcp_servers.graphify import GraphVault
    return GraphVault()


@st.cache_resource
def store():
    from mcp_servers.state_store import StateStore
    Path(STATE_DB).parent.mkdir(parents=True, exist_ok=True)
    return StateStore(STATE_DB)


def ctx(tenant: str) -> dict:
    from uuid import uuid4
    return {"tenant_id": tenant, "user_id": "user_workbench", "session_id": str(uuid4()),
            "idempotency_key": str(uuid4()), "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}


st.set_page_config(page_title="Unreliable Assistant Workbench", layout="wide")
tenant = st.sidebar.selectbox("Tenant", ["tenant_demo", "tenant_acme", "tenant_globex"])
t1, t2, t3, t4, t5, t6 = st.tabs(["1 Architecture & Workflow", "2 Hybrid RAG Lab", "3 MCP & Plugin Inspector",
                                  "4 Memory & Graphify Tokens", "5 Red Team & Circuit Breaker",
                                  "6 36-KPI Scorecard & Load"])

# --- Tab 1 -------------------------------------------------------------------------------------------------------
with t1:
    st.header("Workflow orchestration")
    if DIAGRAM.exists():
        st.image(str(DIAGRAM), width="stretch")
    else:
        st.error("Diagram missing. Run: uv run --with graphviz --with pillow python scripts/render_workflow.py")
    with st.expander("design.md contract"):
        st.markdown((ROOT / "design.md").read_text())

    st.subheader("Triage console")
    q = st.text_input("Query", "Release LOT-88392 once QA has approved it", key="triage_q")
    if q:
        from guardrails import run_prehooks
        from harness.errors import HarnessError
        from triage import classify
        try:
            pre = run_prehooks(q)
            st.json({"redacted": pre.text, "pii": pre.pii, "injection_score": pre.injection_score,
                     **asdict(classify(pre.text))})
        except HarnessError as e:
            st.error(f"Blocked by pre-hook: {e.name}")
    st.caption("ReAct step trace: needs the LLM planner, which does not exist yet.")

    st.subheader(f"Tier 3 HITL queue ({tenant})")
    reviewer = st.text_input("Reviewer id", "user_reviewer")
    pending = store().pending(tenant)
    if not pending:
        st.info(f"No pending approvals in {STATE_DB}.")
    for a in pending:
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.write(f"`{a['approval_id']}` {a['action']} {a['payload']}")
        for col, decision in ((c2, "APPROVED"), (c3, "REJECTED")):
            if col.button(decision.title(), key=f"{decision}-{a['approval_id']}"):
                from harness.errors import HarnessError
                try:  # four-eyes and RBAC are enforced by the store, not by this page
                    store().decide(tenant, a["approval_id"], reviewer_id=reviewer, reviewer_role="reviewer",
                                   decision=decision, note="workbench")
                    st.rerun()
                except HarnessError as e:
                    st.error(f"{e.name}: {e}")

# --- Tab 2 -------------------------------------------------------------------------------------------------------
with t2:
    st.header("3-stage hybrid retrieval")
    q2 = st.text_input("Query", "How fast must a Critical deviation be escalated?", key="rag_q")
    c = st.columns(4)
    k1 = c[0].slider("BM25 k1", 0.5, 2.5, 1.25, 0.05)
    b = c[1].slider("BM25 b", 0.0, 1.0, 0.75, 0.05)
    rrf_k = c[2].slider("RRF k", 1, 200, 60)
    threshold = c[3].slider("Rerank threshold", 0.0, 1.0, 0.42, 0.01)
    if st.button("Run retrieval (loads the ONNX models on first use)"):
        from rank_bm25 import BM25Okapi

        from mcp_servers.rag_server import RERANK_POOL, RRF_POOL, bm25_tokens, rerank, rrf
        r = retriever()
        allowed = r._allowed(tenant, None)
        bm = BM25Okapi([ch["tokens"] for ch in r.chunks], k1=k1, b=b).get_scores(bm25_tokens(q2))
        sparse = sorted(((i, float(bm[i])) for i in range(len(r.chunks)) if allowed[i]), key=lambda t: -t[1])[:50]
        dense = r._dense(q2, 50, allowed)
        fused = rrf([[i for i, _ in sparse], [i for i, _ in dense]], rrf_k, RRF_POOL)
        ids = [i for i, _ in fused][:RERANK_POOL]
        scores = rerank(q2, [r.chunks[i]["text"] for i in ids]) if ids else []
        sp, de = dict(sparse), dict(dense)
        rows = [{"note": r.chunks[i]["note_id"], "BM25": round(sp.get(i, 0.0), 3), "cosine": round(de.get(i, 0.0), 3),
                 "RRF rank": rank, "rerank": round(float(s), 3), "kept": bool(s >= threshold),
                 "text": r.chunks[i]["text"][:120]}
                for rank, (i, s) in enumerate(zip(ids, scores), 1)]
        st.dataframe(sorted(rows, key=lambda x: -x["rerank"]), hide_index=True, width="stretch")
        if not any(x["kept"] for x in rows):
            st.warning("No chunk passes the threshold: the assistant abstains without calling the LLM.")

# --- Tab 3 -------------------------------------------------------------------------------------------------------
with t3:
    st.header("MCP servers and contracts")
    from contracts import models as M
    from mcp_servers.graphify import GRAPH
    from mcp_servers.rag_server import INDEX
    st.dataframe([
        {"server": "mcp-hybrid-rag", "artifact": str(INDEX / "chunks.jsonl"), "ready": (INDEX / "chunks.jsonl").exists()},
        {"server": "mcp-graphify-vault", "artifact": str(GRAPH), "ready": GRAPH.exists()},
        {"server": "mcp-state-store", "artifact": STATE_DB, "ready": Path(STATE_DB).exists()},
    ], hide_index=True, width="stretch")
    st.subheader("JSON-RPC payload inspector")
    model = st.selectbox("Contract", ["HybridRetrieveIn", "HybridRetrieveOut", "GraphSubgraphIn", "GraphSubgraphOut",
                                      "SubmitTransactionIn", "AppendAuditIn", "CallContext"])
    st.json(getattr(M, model).model_json_schema())
    st.subheader("Pre/post plugins")
    from triage.subagents import SUBAGENTS
    st.write("Pre-hooks: PII redactor, injection shield (always on, fail closed). Chunk gate: index-time review "
             "queue. Post: verifier, audit, answer cache. They are not toggleable: turning a guardrail off is a "
             "code change with review, not a switch.")
    st.dataframe([{"subagent": s.name, "servers": ", ".join(sorted(s.mcp_servers)), "tools": ", ".join(sorted(s.allowed_tools)),
                   "max tier": s.max_tier} for s in SUBAGENTS.values()],
                 hide_index=True, width="stretch")

# --- Tab 4 -------------------------------------------------------------------------------------------------------
with t4:
    st.header("Graphify subgraph and token savings")
    from mcp_servers.graphify import GRAPH, render
    if not GRAPH.exists():
        st.error("No graph. Run: uv run mcp_servers/graphify.py build")
    else:
        from contracts.models import GraphSubgraphIn
        v = vault()
        visible = sorted(n for n, x in v.nodes.items() if x["tenants"] is None or tenant in x["tenants"])
        anchors = st.multiselect("Anchor notes", visible, default=visible[:1], max_selections=4)
        depth = st.radio("Depth", [1, 2], index=1, horizontal=True)
        budget = st.slider("Token budget", 64, 800, 320, 16)
        if anchors:
            g = v.subgraph(GraphSubgraphIn.model_validate(
                {"ctx": ctx(tenant), "anchor_note_ids": anchors, "depth": depth, "max_tokens": budget}, strict=False))
            dot = "digraph G { rankdir=LR; node [shape=box, fontsize=10];\n" + "\n".join(
                [f'"{n["id"]}" [style=filled, fillcolor="{"#cfe8ff" if n["id"] in anchors else "#f2f2f2"}"];'
                 for n in g.nodes] + [f'"{s}" -> "{d}";' for s, _, d in g.edges]) + "\n}"
            st.graphviz_chart(dot)
            saved = 1 - g.token_count / g.raw_token_estimate if g.raw_token_estimate else 0.0
            m1, m2, m3 = st.columns(3)
            m1.metric("Subgraph tokens", g.token_count)
            m2.metric("Raw notes tokens", g.raw_token_estimate)
            m3.metric("Saved", f"{saved:.0%}")
            with st.expander("Prompt text"):
                st.code(render(g.nodes, g.edges))
        files = {m.group(1): p for p in (ROOT / "knowledge_vault").glob("*.md")
                 if (m := re.search(r"^id: (\S+)$", p.read_text(), re.M))}  # file names are not note ids
        note = st.selectbox("Note viewer", visible)
        st.markdown(files[note].read_text() if note in files else "not found")
    st.subheader(f"Session facts and audit ({tenant})")
    s = store()
    facts = s._q("SELECT session_id, key, value FROM session_facts WHERE tenant_id = ? ORDER BY id DESC LIMIT 50",
                 (tenant,))
    st.dataframe([{"session": a, "key": k, "value": val} for a, k, val in facts], hide_index=True, width="stretch")
    st.write(f"Audit chain intact: **{s.verify_chain(tenant)}** ({len(s.audit_events(tenant))} events)")

# --- Tab 5 -------------------------------------------------------------------------------------------------------
with t5:
    st.header("Red team")
    split = st.radio("Split", ["heldout", "dev"], horizontal=True)
    c = st.columns(4)
    if c[0].button("PII redaction"):
        from eval import redteam
        st.session_state.rt = ("PII recall", redteam.pii_run(split))
    if c[1].button("Injection shield"):
        from eval import redteam
        st.session_state.rt = ("Shield block rate", redteam.injection_run(split))
    if c[2].button("End to end (loads classifier)"):
        from eval import redteam
        st.session_state.rt = ("End-to-end block rate", redteam.e2e_run(split))
    if c[3].button("Cross-tenant cache probe"):
        from eval import kpis
        r = kpis.cache_isolation_kpi()
        st.session_state.rt = ("Cache isolation", {"value": r.value, "status": r.status, "note": r.note, "misses": []})
    if "rt" in st.session_state:
        label, res = st.session_state.rt
        val = res.get("recall", res.get("block_rate", res.get("value")))
        st.metric(label, f"{val:.4f}" if isinstance(val, float) else val, help=res.get("note"))
        if res.get("misses"):
            st.dataframe([{k: m.get(k) for k in ("id", "kind", "transform", "entity", "why", "text")} for m in res["misses"]],
                         hide_index=True, width="stretch")
    st.subheader("Circuit breakers (bulkheads)")
    if "bulkheads" not in st.session_state:
        from harness import default_bulkheads
        st.session_state.bulkheads = default_bulkheads()
    for name, bh in st.session_state.bulkheads.items():
        a, b_, c_ = st.columns([3, 1, 1])
        a.write(f"**{name}** pool size {bh.size}: {'OPEN (rejecting)' if bh.is_open else 'closed'}, "
                f"rejected {bh.rejected}")
        if b_.button("Trip", key=f"trip-{name}"):
            bh.trip()
            st.rerun()
        if c_.button("Reset", key=f"reset-{name}"):
            bh.reset()
            st.rerun()
    st.caption("These are this page's own bulkheads, for exercising the trip/reset logic. The API process has its own.")
    if st.button("Run the chaos suite (retry storms, timeouts, SKU repair loop)"):
        from eval import kpis
        r = kpis.chaos_kpi()
        st.write(f"{r.id} {r.status} {r.value} {r.note}")

# --- Tab 6 -------------------------------------------------------------------------------------------------------
with t6:
    st.header("36 enterprise KPIs")
    report = load("kpi_report.json")
    if report is None:
        st.error("No KPI report. Run: uv run eval/kpis.py")
    else:
        summary = report["summary"]
        cols = st.columns(5)
        for col, (label, n) in zip(cols, summary["counts"].items()):
            col.metric(label.replace("_", " ").title(), n)
        cols[4].metric("Release ready", "yes" if summary["release_ready"] else "no")
        if summary["release_blockers"]:
            st.warning("Safety KPIs blocking release: " + ", ".join(summary["release_blockers"]))
        st.dataframe(
            [{"KPI": k["id"], "Stage": k["stage"], "Name": k["name"], "Target": k["target"], "Value": k["value"],
              "Status": k["status"], "Safety": "yes" if k["blocking"] else "", "Source": k["source"], "Note": k["note"]}
             for k in report["kpis"]],
            hide_index=True, width="stretch")

    ragas = load("ragas_report.json")
    if ragas:
        st.subheader("RAGAS (Gate 3 run)")
        st.json({k: ragas[k] for k in ("generator", "judge", "answerable", "negative_abstain_rate", "pass") if k in ragas})

    k6 = load("k6_summary.json")
    if k6:
        import pandas as pd
        st.subheader(f"Load: {k6['vus_max']} VUs, {k6['rps']:.1f} RPS, 5xx {k6['server_5xx_rate']:.4f}")
        if k6["vus_max"] < 50_000:
            st.caption("Not a 50K-VU run: KPI-31..35 stay NOT_MEASURED (ADR-011).")
        lat = k6["latency_ms"]
        st.bar_chart(pd.DataFrame({"measured ms": [lat["med"], lat["p(95)"], lat["p(99)"]],
                                   "SLA ms": [240, 450, 590]}, index=["p50", "p95", "p99"]), stack=False)
    soak = load("soak.json")
    if soak and soak.get("samples"):
        import pandas as pd
        growth = f"{soak['rss_growth']:.1%}" if soak["rss_growth"] is not None else "n/a"
        st.subheader(f"Soak: {soak['minutes']} min at {soak['vus']} VUs, RSS growth {growth}")
        st.line_chart(pd.DataFrame(soak["samples"]).set_index("t_s")["rss_kb"] / 1024)
    st.caption("OTel span waterfall: export spans with OTEL_EXPORTER_OTLP_ENDPOINT to a collector (Jaeger/Tempo); "
               "the workbench does not embed one.")
