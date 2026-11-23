"""Subagent topology (design.md Gate 7). Each subagent is bound to specific MCP servers and tools.

The binding is enforced in code: a ScopedHarness refuses any tool outside the subagent's allow-list,
so a prompt-injected Retrieval Subagent cannot reach a state-mutating tool (NG-2, NG-1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.errors import RBAC_DENIED, HarnessError
from harness.tool_harness import TIER_RANK, CallResult, Tier, ToolExecutionHarness
from triage.classifier import Route


@dataclass(frozen=True)
class SubagentSpec:
    name: str
    mcp_servers: frozenset[str]
    allowed_tools: frozenset[str]
    max_tier: Tier
    rule: str


SUBAGENTS: dict[str, SubagentSpec] = {
    "retrieval": SubagentSpec(
        "retrieval", frozenset({"mcp-hybrid-rag", "mcp-graphify-vault"}),
        frozenset({"hybrid_retrieve", "graph_subgraph"}), "TIER_1_READ",
        "Read-only; cannot invoke state-mutating tools.",
    ),
    "execution": SubagentSpec(
        "execution", frozenset({"mcp-state-store"}),
        frozenset({"get_session_facts", "put_session_fact", "lookup_lot", "submit_transaction", "get_approval"}),
        "TIER_2_COMPUTE",
        "No RAG or external web access; Tier 3 triggers the HITL freeze (submit_transaction only writes PENDING_APPROVAL).",
    ),
    "verification": SubagentSpec(
        "verification", frozenset({"mcp-state-store"}), frozenset({"append_audit"}), "TIER_1_READ",
        "Audit log only; independent model context; vetoes ungrounded drafts.",
    ),
}

_ROUTE_TO_SUBAGENT: dict[Route, str | None] = {
    "HYBRID_RAG": "retrieval",
    "TRANSACTION_EXEC": "execution",
    "DIRECT_ANSWER": None,
    "REJECT_OOS": None,
}

_CANNED: dict[str, str] = {
    "DIRECT_ANSWER": (
        "I answer questions about approved manufacturing, quality, logistics and governance documents, "
        "with citations, and I can help prepare records. Changes need human approval."
    ),
    "REJECT_OOS": (
        "I can't help with that. I only handle manufacturing, quality, logistics and governance questions "
        "from the approved knowledge base, and I don't give medical, legal or financial advice."
    ),
}


def subagent_for(route: Route) -> SubagentSpec | None:
    name = _ROUTE_TO_SUBAGENT[route]
    return SUBAGENTS[name] if name else None


def canned_response(route: Route) -> str | None:
    """Deterministic reply for routes that never reach an LLM."""
    return _CANNED.get(route)


class ScopedHarness:
    """Wraps ToolExecutionHarness so a subagent can only call its own tools, and never above its tier."""

    def __init__(self, harness: ToolExecutionHarness, spec: SubagentSpec) -> None:
        self._harness, self.spec = harness, spec

    async def call(self, tool: str, payload: dict[str, Any], *, caller_tier: Tier) -> CallResult:
        if tool not in self.spec.allowed_tools:
            raise HarnessError(RBAC_DENIED, f"{self.spec.name} subagent is not bound to {tool!r}. Do not retry.")
        capped: Tier = caller_tier if TIER_RANK[caller_tier] <= TIER_RANK[self.spec.max_tier] else self.spec.max_tier
        return await self._harness.call(tool, payload, caller_tier=capped)
