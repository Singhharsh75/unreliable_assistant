"""Gate 17: graphify vault (design.md §1.2.3) — note graph from frontmatter and [[wikilinks]], pruned per query.

  uv run mcp_servers/graphify.py build      # knowledge_vault/ -> indexes/graph.json

`graph_subgraph` walks outgoing links from the anchor notes (the notes the top chunks came from) up to `depth` hops,
skipping notes the caller's tenant may not read, and keeps nodes in hop order while the rendered subgraph fits
`max_tokens`. The rendered text is what goes into the prompt, so `token_count` is what the prompt pays;
`raw_token_estimate` is the full text of every reachable note, the dump the subgraph replaces (KPI-26).

`build` also writes a `code` layer (Python modules, their project imports, and the Pydantic schemas each defines)
for the workbench and for tracing which code a contract lives in. Prompts use the vault layer only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context.compress import estimate_tokens  # noqa: E402
from contracts.models import GraphSubgraphIn, GraphSubgraphOut  # noqa: E402

VAULT = Path(os.getenv("RAG_VAULT", ROOT / "knowledge_vault"))
GRAPH = Path(os.getenv("GRAPH_PATH", ROOT / "indexes" / "graph.json"))
_LINK = re.compile(r"\[\[([a-z0-9-]+)\]\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def build(vault: Path = VAULT, out: Path = GRAPH) -> dict:
    from mcp_servers.rag_server import load_notes  # lazy: pulls in the ONNX stack

    notes = load_notes(vault)
    ids = {str(m["id"]) for m, _ in notes}
    nodes, edges = {}, []
    for meta, body in notes:
        nid = str(meta["id"])
        text = re.sub(r"^#.*\n+", "", body).strip()  # drop the title line
        first = _SENT.split(text, maxsplit=1)[0] if text else ""
        nodes[nid] = {"id": nid, "domain": meta.get("domain"), "risk_tier": meta.get("risk_tier"),
                      "version": str(meta.get("version")), "tenants": meta.get("tenants"),
                      "summary": _LINK.sub(r"\1", first), "tokens": estimate_tokens(body)}
        edges += [(nid, "links_to", dst) for dst in dict.fromkeys(_LINK.findall(body)) if dst in ids and dst != nid]
    graph = {"nodes": nodes, "edges": edges, "code": code_graph()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=1) + "\n")
    return graph


CODE_DIRS = ("api", "contracts", "context", "guardrails", "harness", "mcp_servers", "observability", "triage", "eval")


def code_graph(root: Path = ROOT) -> dict:
    """Module nodes, `imports` edges between project modules, `defines` edges to BaseModel subclasses."""
    import ast

    mods = {".".join(p.relative_to(root).with_suffix("").parts): p for d in CODE_DIRS for p in (root / d).rglob("*.py")}
    mods = {m.removesuffix(".__init__"): p for m, p in mods.items()}
    nodes, edges = [], set()
    for mod, path in sorted(mods.items()):
        tree = ast.parse(path.read_text())
        nodes.append({"id": mod, "kind": "module", "tokens": estimate_tokens(path.read_text())})
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                targets = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            else:
                targets = []
            edges.update((mod, "imports", t) for t in targets if t in mods and t != mod)
            if isinstance(n, ast.ClassDef) and any(getattr(b, "id", getattr(b, "attr", "")) == "BaseModel"
                                                  or getattr(b, "id", "") in {"_In"} for b in n.bases):
                nodes.append({"id": f"{mod}:{n.name}", "kind": "schema"})
                edges.add((mod, "defines", f"{mod}:{n.name}"))
    return {"nodes": nodes, "edges": sorted(edges)}


def render(nodes: list[dict], edges: list[tuple[str, str, str]]) -> str:
    """Prompt text for a subgraph; its token estimate is the subgraph's cost."""
    lines = [f"- {n['id']} ({n['domain']}, tier {n['risk_tier']}): {n['summary']}" for n in nodes]
    return "\n".join(lines + [f"{s} -> {d}" for s, _, d in edges])


class GraphVault:
    def __init__(self, path: Path = GRAPH) -> None:
        g = json.loads(path.read_text())
        self.nodes: dict[str, dict] = g["nodes"]
        self.out: dict[str, list[str]] = {}
        for s, _, d in g["edges"]:
            self.out.setdefault(s, []).append(d)

    def subgraph(self, m: GraphSubgraphIn) -> GraphSubgraphOut:
        def visible(nid: str) -> bool:
            n = self.nodes.get(nid)
            return n is not None and (n["tenants"] is None or m.ctx.tenant_id in n["tenants"])

        frontier = list(dict.fromkeys(a for a in m.anchor_note_ids if visible(a)))
        reached, seen = list(frontier), set(frontier)
        for _ in range(m.depth):  # breadth-first, so nodes come out nearest first
            frontier = [d for s in frontier for d in sorted(self.out.get(s, [])) if visible(d) and d not in seen]
            frontier = list(dict.fromkeys(frontier))
            seen.update(frontier)
            reached += frontier
        kept: list[dict] = []
        for nid in reached:  # greedy in hop order: anchors first, then their neighbours
            n = {k: self.nodes[nid][k] for k in ("id", "domain", "risk_tier", "version", "summary")}
            ids = {x["id"] for x in kept} | {nid}
            edges = [(s, "links_to", d) for s in ids for d in self.out.get(s, []) if d in ids]
            if kept and estimate_tokens(render(kept + [n], edges)) > m.max_tokens:
                continue
            kept.append(n)
        ids = {x["id"] for x in kept}
        edges = sorted((s, "links_to", d) for s in ids for d in self.out.get(s, []) if d in ids)
        return GraphSubgraphOut(nodes=kept, edges=edges, token_count=estimate_tokens(render(kept, edges)),
                                raw_token_estimate=sum(self.nodes[n]["tokens"] for n in reached))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        g = build()
        print(f"graph: {len(g['nodes'])} notes, {len(g['edges'])} links; code: {len(g['code']['nodes'])} nodes, "
              f"{len(g['code']['edges'])} edges -> {GRAPH}")
    else:
        print(__doc__)
