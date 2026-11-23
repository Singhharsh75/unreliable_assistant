"""Deterministic triage (design.md Gate 6; temperature 0 by construction, no network call, so it fits the 25 ms budget).

Tags a query with (intent, domain, risk_tier, complexity) and picks one of four routes.
Safety rules (FMEA #9): confidence < 0.7 falls back to HYBRID_RAG, and TRANSACTION_EXEC is never a default.
A mutation route needs an imperative request, never a question about how a procedure works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from context.anchor import normalize_entities

Route = Literal["DIRECT_ANSWER", "HYBRID_RAG", "TRANSACTION_EXEC", "REJECT_OOS"]
CONFIDENCE_FLOOR = 0.7


@dataclass(frozen=True)
class TriageResult:
    route: Route
    intent: str
    domain: str
    risk_tier: str
    complexity: Literal["low", "medium", "high"]
    confidence: float
    reason: str


def _rx(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


_SOCIAL = re.compile(
    r"^\s*(?:(?:hi|hello|hey)(?:\s+there)?|thanks|thank you|good (?:morning|afternoon|evening)|bye|goodbye|ok(?:ay)?|cheers)\b[\s!.,]*$"
    r"|^\s*(?:what can you do|who are you|help|what are you|how do you work)\??\s*$",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"^\s*(?:how|what|why|when|where|which|who|explain|describe|summari[sz]e|list|tell me|is there|are there|does|do|can|should|must)\b"
    r"|\?\s*$",
    re.IGNORECASE,
)
# "can you approve ...", "could you submit ..." are requests, not questions about procedure.
_POLITE_REQUEST = re.compile(r"^\s*(?:please\s+)?(?:can|could|would|will) you\b|^\s*please\b|^\s*i (?:want|need|would like) to\b", re.IGNORECASE)
_HOW_PROCEDURE = re.compile(r"^\s*(?:how (?:do|does|should|can|to)|what is the (?:process|procedure|policy)|explain|describe)\b", re.IGNORECASE)
_MUTATE = _rx(
    "approve", "reject", "release", "quarantine", "submit", "start", "initiate", "open", "close", "escalate", "update",
    "change", "set", "mark", "transition", "move", "cancel", "create", "file", "raise", "put", "place", "hold",
)
_MUTATE_VERBS = (
    "approve|reject|release|quarantine|submit|start|initiate|open|close|escalate|update|change|set|mark|"
    "transition|move|cancel|create|file|raise|put|place|hold"
)
# The request must open with the verb, optionally after "please", "can you", or "I want to".
_MUTATE_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could|would|will) you\s+|i (?:want|need|would like) to\s+)?(?:please\s+)?(?:" + _MUTATE_VERBS + r")\b",
    re.IGNORECASE,
)
_WORKFLOW_NOUN = _rx(
    "deviation", "capa", "workflow", "transaction", "lot", "batch", "sku", "order", "shipment", "recall", "request",
    "ticket", "record", "status", "complaint", "change control", "form",
)
_STATUS_LOOKUP = re.compile(r"\b(?:status|state|disposition)\b.*\b(?:lot|batch)[\s-]*\d+|\b(?:lookup|look up|check)\b.*\b(?:lot|batch)[\s-]*\d+", re.IGNORECASE)

_OOS_TOPICS = _rx(
    "weather", "capital of", "recipe", "poem", "joke", "song", "movie", "football", "cricket", "stock price", "bitcoin",
    "crypto", "horoscope", "lottery", "president", "celebrity", "vacation", "holiday", "restaurant", "dating",
    "translate", "write me a story", "essay", "homework", "python code", "javascript", "world cup", "olympics",
    "election", "prime minister", "capital city", "movie", "tv show", "video game", "pizza", "stock market",
)
_ADVICE = _rx(
    "diagnose", "diagnosis", "symptom", "symptoms", "dosage", "prescribe", "medication", "my doctor", "lawsuit", "sue",
    "attorney", "legal advice", "tax", "invest", "investment", "should i buy", "mortgage", "divorce", "pain",
)
_INJECTION = _rx("ignore (?:all |your )?(?:previous|prior) instructions", "system prompt", "jailbreak", "you are now", "developer mode")

_DOMAINS: list[tuple[str, re.Pattern[str]]] = [
    ("logistics", _rx("shipping", "shipment", "cold chain", "cold-chain", "warehouse", "storage", "returns?", "carrier", "temperature excursion")),
    ("quality", _rx("deviation", "capa", "oos", "out of specification", "calibration", "batch record", "release", "qa", "inspection", "complaint", "recall", "stability", "cleaning", "root cause", "validation", "label", "sop", "protocol", "lot", "batch")),
    ("governance", _rx("access", "audit", "pii", "retention", "approval", "hitl", "training", "data integrity", "policy", "role", "permission")),
    ("catalog", _rx("sku", "product famil\\w+", "catalog")),
    ("tenant", _rx("acme", "globex", "sla", "escalation")),
]
_IN_DOMAIN = re.compile("|".join(p.pattern for _, p in _DOMAINS), re.IGNORECASE)


def _domain(q: str) -> str:
    for name, pat in _DOMAINS:
        if pat.search(q):
            return name
    return "general"


def _complexity(q: str, n_entities: int) -> Literal["low", "medium", "high"]:
    words = len(q.split())
    multi = bool(re.search(r"\b(?:and|compare|versus|vs|difference|both|then|across|relationship)\b", q, re.IGNORECASE))
    if words > 25 or (multi and (n_entities > 1 or words > 12)):
        return "high"
    if words > 10 or multi or n_entities:
        return "medium"
    return "low"


def classify(query: str) -> TriageResult:
    q = query.strip()
    entities = normalize_entities(q)
    complexity = _complexity(q, len(entities))
    domain = _domain(q)

    def out(route: Route, intent: str, tier: str, conf: float, why: str) -> TriageResult:
        if conf < CONFIDENCE_FLOOR and route != "HYBRID_RAG":
            return TriageResult("HYBRID_RAG", intent, domain, "TIER_1_READ", complexity, conf, f"low confidence ({why}); safe default")
        return TriageResult(route, intent, domain, tier, complexity, conf, why)

    if not q:
        return out("REJECT_OOS", "empty", "TIER_1_READ", 0.99, "empty query")
    if _INJECTION.search(q):
        return out("REJECT_OOS", "injection", "TIER_1_READ", 0.95, "instruction-override attempt")
    if _SOCIAL.match(q):
        return out("DIRECT_ANSWER", "smalltalk", "TIER_1_READ", 0.95, "greeting or capability question")

    in_domain = bool(_IN_DOMAIN.search(q)) or bool(entities)
    if _ADVICE.search(q) and not _rx("sop", "protocol", "policy", "training").search(q):
        return out("REJECT_OOS", "advice", "TIER_1_READ", 0.9, "medical/legal/financial advice")
    if _OOS_TOPICS.search(q) and not in_domain:
        return out("REJECT_OOS", "off_topic", "TIER_1_READ", 0.9, "off-topic")

    is_question = bool(_QUESTION.search(q)) and not _POLITE_REQUEST.search(q)
    procedural = bool(_HOW_PROCEDURE.search(q))
    imperative_mutation = (
        bool(_MUTATE_LEAD.match(q)) and bool(_WORKFLOW_NOUN.search(q)) and not procedural
        and (not is_question or bool(_POLITE_REQUEST.search(q)))
    )
    if imperative_mutation:
        return out("TRANSACTION_EXEC", "mutate", "TIER_3_MUTATE", 0.9 if entities or _rx("workflow", "deviation", "capa", "transaction").search(q) else 0.6, "imperative state change")
    if _STATUS_LOOKUP.search(q) and entities and not procedural:
        return out("TRANSACTION_EXEC", "lookup", "TIER_1_READ", 0.85, "record lookup")

    # No positive out-of-scope evidence: retrieval decides. A question the vault cannot answer is abstained
    # deterministically (rerank < 0.42), so refusing here on missing keywords would only add false refusals (KPI-06).
    return out("HYBRID_RAG", "question", "TIER_1_READ", 0.9 if domain != "general" else 0.75, "knowledge question")
