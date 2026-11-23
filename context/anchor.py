"""Immutable GoalAnchorBlock (design.md Gate 5 §1). Rendered at index 0 of every ReAct prompt."""
from __future__ import annotations

import re
from dataclasses import dataclass

from contracts.models import LOT_RE, SKU_RE

_ENTITY_RE = re.compile(r"\b(lot|sku)[\s-]*([0-9]{1,5})\b", re.IGNORECASE)


def normalize_entities(text: str) -> tuple[str, ...]:
    """Find LOT/SKU mentions in free text and return canonical IDs, e.g. 'lot 44' -> 'LOT-00044'."""
    found: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        prefix, digits = m.group(1).upper(), int(m.group(2))
        candidate = f"LOT-{digits:05d}" if prefix == "LOT" else f"SKU-{digits:03d}"
        pattern = LOT_RE if prefix == "LOT" else SKU_RE
        if re.match(pattern, candidate) and candidate not in found:
            found.append(candidate)
    return tuple(found)


@dataclass(frozen=True)
class GoalAnchorBlock:
    original_intent: str
    entity_ids: tuple[str, ...]
    policy_constraints: tuple[str, ...]

    def render(self) -> str:
        lines = [
            "GOAL ANCHOR (immutable; do not drift from it)",
            f"intent: {self.original_intent}",
            f"entities: {', '.join(self.entity_ids) or 'none'}",
            "policies: " + ("; ".join(self.policy_constraints) or "none"),
        ]
        return "\n".join(lines)


def build_anchor(query: str, policy_constraints: tuple[str, ...] = ()) -> GoalAnchorBlock:
    return GoalAnchorBlock(
        original_intent=query,
        entity_ids=normalize_entities(query),
        policy_constraints=tuple(policy_constraints),
    )


@dataclass(frozen=True)
class Turn:
    tool: str
    observation: str  # already compressed


@dataclass(frozen=True)
class ReActPrompt:
    anchor: GoalAnchorBlock
    turns: tuple[Turn, ...]  # sliding window, at most 3

    def messages(self) -> list[str]:
        """The anchor is always element 0. Nothing can be inserted before it."""
        out = [self.anchor.render()]
        out += [f"[{t.tool}] {t.observation}" for t in self.turns]
        return out
