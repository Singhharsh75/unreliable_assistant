"""Verification Subagent claim check (design.md §1.3 step 8, KPI-24, FMEA #13).

Each sentence of a draft is a claim. A claim is grounded when one source covers at least SUPPORT_THRESHOLD of its
content tokens (stemmed, stopwords removed) and contains every number and ID in the claim verbatim.
The draft is released only if every claim is grounded; otherwise the caller vetoes it.

shortcut: lexical entailment stands in for an NLI model, so a paraphrase can be vetoed and a negation flip
("must" -> "must not") can pass; upgrade to an ONNX NLI cross-encoder if KPI-24 audits show either.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from mcp_servers.rag_server import bm25_tokens

SUPPORT_THRESHOLD = 0.8
_STOP = frozenset(bm25_tokens(
    "a an the is are was were be been being of to in on for and or but with by at as from that this these those it its "
    "which who what when where how can may will shall do does did has have had than then so such any all each per via "
    "into there their they them you your we our i me my he she his her also only just currently"
))
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_LIST_MARK = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")  # "1. ", "- ": layout, not a claim
_EXACT = re.compile(r"\b[A-Z]{2,}-\d+\b|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ClaimCheck:
    sentence: str
    source: str | None  # id of the supporting source, None when unsupported
    coverage: float


@dataclass(frozen=True)
class Verdict:
    grounded: bool
    faithfulness: float  # supported claims / claims
    claims: tuple[ClaimCheck, ...]

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.source for c in self.claims if c.source))


def segments(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(_LIST_MARK.sub("", _CITE.sub("", text))) if s.strip()]


def _content(text: str) -> set[str]:
    return {t for t in bm25_tokens(text) if t not in _STOP}


def _has_exact(value: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(value)}(?!\.?\d)", text, re.IGNORECASE) is not None


def verify(draft: str, sources: list[tuple[str, str]]) -> Verdict:
    """`sources` is a list of (source_id, text). Sentences with no content words ("Yes.") are skipped; a draft with
    no checkable claim at all is not grounded (fail closed)."""
    prepared = [(sid, _content(text), text) for sid, text in sources]
    checks: list[ClaimCheck] = []
    for sentence in segments(draft):
        tokens = _content(sentence)
        if not tokens:
            continue
        exact = _EXACT.findall(sentence)
        best_id, best = None, 0.0
        for sid, ctoks, text in prepared:
            if not all(_has_exact(x, text) for x in exact):
                continue
            cov = len(tokens & ctoks) / len(tokens)
            if cov > best:
                best_id, best = sid, cov
        checks.append(ClaimCheck(sentence, best_id if best >= SUPPORT_THRESHOLD else None, best))
    supported = sum(c.source is not None for c in checks)
    faithfulness = supported / len(checks) if checks else 0.0
    return Verdict(bool(checks) and supported == len(checks), faithfulness, tuple(checks))


def extractive_quote(text: str, max_chars: int = 400) -> str:
    """Fallback after a veto: verbatim leading sentences of the top chunk, so it is grounded by construction."""
    out = ""
    for s in segments(text):
        if out and len(out) + len(s) + 1 > max_chars:
            break
        out = f"{out} {s}".strip()
    return out[:max_chars]
