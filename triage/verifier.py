"""Verification Subagent claim check (design.md §1.3 step 8, KPI-24, FMEA #13).

Each sentence of a draft is a claim. A claim is grounded when one source covers at least SUPPORT_THRESHOLD of its
content tokens (stemmed, stopwords removed) and contains every number and ID in the claim verbatim.
The draft is released only if every claim is grounded; otherwise the caller vetoes it.

With `entail` (triage.nli.entailment) a claim is grounded when an NLI model says a source window entails it, or
when the lexical check passes and no window contradicts it (so a negation flip is vetoed while a paraphrase that
spans several sentences, which NLI under-scores, is kept). Numbers and IDs must appear verbatim either way.
Without `entail` the lexical check alone runs; it vetoes paraphrases and can pass a negation flip.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from mcp_servers.rag_server import bm25_tokens

SUPPORT_THRESHOLD = 0.8
ENTAIL_THRESHOLD = 0.5
CONTRADICT_THRESHOLD = 0.5
NLI_CANDIDATES = 2  # premise windows per claim scored by NLI, best lexical overlap first
MIN_CLAIM_WORDS = 3  # shorter fragments without a number or ID make no checkable claim
WINDOW = 2  # sentences per premise window: NLI dilutes on long multi-fact premises
_STOP = frozenset(bm25_tokens(
    "a an the is are was were be been being of to in on for and or but with by at as from that this these those it its "
    "which who what when where how can may will shall do does did has have had than then so such any all each per via "
    "into there their they them you your we our i me my he she his her also only just currently"
))
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]|\(chunks?\s*\d+(?:\s*,\s*\d+)*\)")
_MARKUP = re.compile(r"\*\*|`")
_LIST_MARK = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")  # "1. ", "- ": layout, not a claim
# A statement about what the context lacks ("The chunks do not say ...") asserts no domain fact, so it is not checked.
_META = re.compile(r"\b(?:chunks?|context|knowledge base|documents?|sources?)\b[^.]*\b(?:do(?:es)? not|doesn't|don't|no)\b"
                   r"|\bnot (?:explicitly |directly )?(?:stated|specified|covered|provided)\b", re.IGNORECASE)
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
    return [s.strip() for s in _SENT.split(_LIST_MARK.sub("", _MARKUP.sub("", _CITE.sub("", text)))) if s.strip()]


def _content(text: str) -> set[str]:
    return {t for t in bm25_tokens(text) if t not in _STOP}


def _windows(text: str) -> list[str]:
    sents = segments(text)
    return [" ".join(sents[i:i + WINDOW]) for i in range(max(len(sents) - WINDOW + 1, 1))]


def _has_exact(value: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(value)}(?!\.?\d)", text, re.IGNORECASE) is not None


Entail = Callable[[list[tuple[str, str]]], list[tuple[float, float]]]  # (P(entail), P(contradict)) per pair


def verify(draft: str, sources: list[tuple[str, str]], entail: Entail | None = None) -> Verdict:
    """`sources` is a list of (source_id, text). Sentences with no content words ("Yes.") are skipped; a draft with
    no checkable claim at all is not grounded (fail closed)."""
    prepared = [(sid, _content(text), text) for sid, text in sources]
    checks: list[ClaimCheck] = []
    for sentence in segments(draft):
        tokens = _content(sentence)
        exact = _EXACT.findall(sentence)
        if _META.search(sentence):
            continue
        words = len(sentence.split())
        label = words <= 6 and sentence[-1] not in ".!?"  # list-item label such as "Who made the action"
        if not tokens or ((words < MIN_CLAIM_WORDS or label) and not exact):  # "No." and labels make no claim
            continue
        cands = [(len(tokens & ctoks) / len(tokens), sid, text) for sid, ctoks, text in prepared
                 if all(_has_exact(x, text) for x in exact)]
        cands.sort(key=lambda c: -c[0])
        lex, lex_id = (cands[0][0], cands[0][1]) if cands else (0.0, None)
        if entail is None:
            checks.append(ClaimCheck(sentence, lex_id if lex >= SUPPORT_THRESHOLD else None, lex))
            continue
        wins = [(len(tokens & _content(w)) / len(tokens), sid, w) for _, sid, text in cands for w in _windows(text)]
        wins.sort(key=lambda c: -c[0])
        top = [c for c in wins[:NLI_CANDIDATES] if c[0] > 0]
        probs = entail([(text, sentence) for _, _, text in top])
        entailed = max(((p[0], c[1]) for p, c in zip(probs, top)), default=(0.0, None))
        contradicted = any(p[1] >= CONTRADICT_THRESHOLD for p in probs)
        if entailed[0] >= ENTAIL_THRESHOLD:  # an entailing window outranks stray contradictions from other windows
            checks.append(ClaimCheck(sentence, entailed[1], entailed[0]))
        elif lex >= SUPPORT_THRESHOLD and not contradicted:
            checks.append(ClaimCheck(sentence, lex_id, lex))
        else:
            checks.append(ClaimCheck(sentence, None, max(entailed[0], lex)))
    supported = sum(c.source is not None for c in checks)
    faithfulness = supported / len(checks) if checks else 0.0
    return Verdict(bool(checks) and supported == len(checks), faithfulness, tuple(checks))


def with_question(question: str, sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Facts the user stated ("a lot with 5 months left") may be restated in an answer, and one claim may join a user
    fact with a chunk rule, so add the question and the joined text as sources."""
    return [*sources, ("question", question), ("joined", " ".join([question, *(t for _, t in sources)]))]


def extractive_quote(text: str, max_chars: int = 400) -> str:
    """Fallback after a veto: verbatim leading sentences of the top chunk, so it is grounded by construction."""
    out = ""
    for s in segments(text):
        if out and len(out) + len(s) + 1 > max_chars:
            break
        out = f"{out} {s}".strip()
    return out[:max_chars]
