"""Gate 2: render docs/workflow_orchestration.png (300 DPI) from the locked design.md topology."""
from pathlib import Path

import graphviz
from PIL import Image
from contextlib import contextmanager

OUT = Path(__file__).resolve().parent.parent / "docs" / "workflow_orchestration"

g = graphviz.Digraph("workflow_orchestration", format="png")
g.attr(rankdir="TB", dpi="300", fontname="Helvetica", fontsize="20", compound="true",
       label="The Unreliable Assistant — Enterprise Workflow Orchestration (design.md v1.0)",
       labelloc="t", nodesep="0.35", ranksep="0.55")
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11", fillcolor="white")
g.attr("edge", fontname="Helvetica", fontsize="9")


@contextmanager
def cluster(parent, name, label, color):
    with parent.subgraph(name=f"cluster_{name}") as c:
        c.attr(label=label, style="rounded,filled", fillcolor=color, color="gray40", fontsize="14")
        yield c


with cluster(g, "ingress", "1. Ingress & Security Layer", "#e8f1fb") as c:
    c.node("ingress", "Ingress\nAPI Gateway (NGINX/Envoy)\nauthN · X-Tenant-ID check · traceparent")
    c.node("pii", "Pre-Exec Plugin\nPII Redactor\n(fail-closed, p10)")
    c.node("inj", "Pre-Exec Plugin\nInjection Classifier\n(fail-closed, p20)")
    c.node("cache", "Redis Semantic Cache Check\nSHA256(tenant:norm_query)", shape="diamond", style="filled")
    c.edges([("ingress", "pii"), ("pii", "inj"), ("inj", "cache")])

with cluster(g, "triage", "2. Triage & Subagent Routing", "#fdf3e1") as c:
    c.node("triage", "Triage Agent\ntemperature=0.0 · < 30 ms\n(intent, domain, risk_tier, complexity)")
    c.node("direct", "Direct Response")
    c.node("retrieval", "Retrieval Subagent\nread-only")
    c.node("execution", "Execution Subagent\nno RAG / web")
    c.node("audit", "Audit (Verification) Subagent\nNLI claim check · veto")
    c.node("reject", "REJECT_OOS\nsafe refusal", fillcolor="#f6d5d5")

with cluster(g, "mcp", "3. MCP & 3-Stage RAG Pipeline (JSON-RPC 2.0)", "#e6f5ea") as c:
    with cluster(c, "rag", "mcp-hybrid-rag (stdio)", "#d3eedb") as r:
        r.node("bm25", "Stage 1: BM25\nk1=1.25 b=0.75 top50 · <12 ms")
        r.node("bienc", "Stage 2: Bi-Encoder + FAISS\nbge-large ONNX top50 · <25 ms")
        r.node("rrf", "RRF Fusion\nk=60 · pool=40 · <3 ms")
        r.node("ce", "Stage 3: Cross-Encoder\nbge-reranker-v2-m3 FP16\ntop4 · thr 0.42 · <45 ms")
        r.edges([("bm25", "rrf"), ("bienc", "rrf"), ("rrf", "ce")])
    c.node("abstain", "Deterministic Abstention\n(score < 0.42, < 90 ms, no LLM)", fillcolor="#f6d5d5")
    c.node("graphify", "mcp-graphify-vault (stdio)\nObsidian [[wikilinks]] · depth ≤ 2\n~320 tok subgraph")
    c.node("state", "mcp-state-store (Streamable HTTP)\nRedis + Postgres\nsession_facts · approvals · audit chain")

with cluster(g, "control", "4. Control Plane & Egress", "#efe6f7") as c:
    c.node("react", "Bounded ReAct Loop\nmax_steps=5 · 450 ms budget\nINIT→…→VERIFY→DONE/HALTED")
    c.node("breaker", "Circuit Breaker\n(tool, args_hash) fails ≥ 2\n→ HALTED_BREAKER")
    c.node("hitl", "Tier 3 HITL Gate\nPENDING_APPROVAL → Reviewer", fillcolor="#fde2b8")
    c.node("llm", "LLM Generate\nClaude (claude-haiku-5-5) · TTFT ≤ 180 ms\nbuffered → NLI verify → release")
    c.node("post", "Post-Exec Plugins\nRAGAS Auditor (faithfulness ≥ 0.92)\n+ cache write (TTL 300±45 s)")
    c.node("otel", "OTel Export\n→ Prometheus · Tempo · Grafana")
    c.node("egress", "Egress\ntenant_id re-check · response")
    c.edges([("react", "breaker"), ("breaker", "hitl"), ("post", "otel"), ("otel", "egress")])

# cross-layer flow
g.edge("cache", "triage", label="miss")
g.edge("cache", "egress", label="hit", style="dashed")
g.edge("triage", "direct", label="DIRECT_ANSWER")
g.edge("triage", "retrieval", label="HYBRID_RAG")
g.edge("triage", "execution", label="TRANSACTION_EXEC")
g.edge("triage", "reject", label="REJECT_OOS")
g.edge("retrieval", "bm25", label="hybrid_retrieve")
g.edge("retrieval", "bienc")
g.edge("ce", "abstain", label="< 0.42", style="dashed")
g.edge("ce", "graphify", label="anchors")
g.edge("graphify", "llm", label="top4 + subgraph")
g.edge("execution", "react")
g.edge("react", "state", label="tool calls (harness)")
g.edge("hitl", "state", label="PENDING_APPROVAL")
g.edge("direct", "llm")
g.edge("llm", "audit")
g.edge("hitl", "audit")
g.edge("audit", "state", label="audit log only", style="dashed")
g.edge("audit", "post", label="grounded")
g.edge("abstain", "post")
g.edge("reject", "post")

if __name__ == "__main__":
    png = g.render(OUT, cleanup=True)
    Image.open(png).save(png, dpi=(300, 300))  # cairo leaves 72 DPI in the PNG header
    print(png)
