"""PII redactor (design.md §1.2.2 step 2, NG-4, KPI-02): regex detectors with checksum validation, fail closed.

Text is NFKC-normalized and stripped of zero-width characters first, so spaced or width-obfuscated values are
caught. Every hit becomes a numbered placeholder (`<EMAIL_1>`); the same value gets the same placeholder in one
request. The result is re-scanned, and any residue blocks the request (PII_BLOCKED).

shortcut: regex stands in for Presidio/NER, so free-text names and street addresses are not detected; add an NER
pass if the audit shows names leaking.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9"}
_DW = "|".join(_DIGIT_WORDS)
_TLD = "com|org|net|io|edu|gov|co|in|uk|de|fr|info|biz|us|ca|au"
_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_AT = r"(?:@|\s*[\[({<]\s*at\s*[\])}>]\s*)"
_DOT = r"(?:\.|\s*[\[({<]\s*dot\s*[\])}>]\s*)"

# (entity, compiled pattern), applied in order; earlier entities consume their span.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(rf"[a-z0-9._%+-]+{_AT}{_LABEL}(?:{_DOT}{_LABEL})*{_DOT}[a-z]{{2,}}", re.I)),
    ("EMAIL", re.compile(rf"\b[a-z0-9._%+-]+\s+at\s+{_LABEL}(?:\s+dot\s+{_LABEL})+\b", re.I)),
    ("EMAIL", re.compile(rf"\b[a-z0-9._%+-]+\s+at\s+{_LABEL}(?:\.{_LABEL})*\.(?:{_TLD})\b", re.I)),
    ("SSN", re.compile(r"(?<![\w-])\d{3}[-\s.]\d{2}[-\s.]\d{4}(?![\w-])")),
    ("SSN", re.compile(r"(?i)\b(?:ssn|social security(?: number)?|national id(?: number)?|nin|tax id|passport(?: number)?)"
                       r"\W{0,6}(?:is|:|#|no\.?|number)?\W{0,4}[A-Z0-9][A-Z0-9 -]{4,13}[A-Z0-9]\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")),
    ("CREDIT_CARD", re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,5})[\s.-]?\d{3,5}[\s.-]?\d{3,5}(?![\w-])")),
    ("PHONE", re.compile(rf"(?i)\b(?:(?:{_DW})(?:[\s,.-]+|$)){{7,}}")),
    ("ID_NUMBER", re.compile(r"(?<![\w.-])\d{9,19}(?![\w-])")),
]
_LUHN_ONLY = {"CREDIT_CARD"}


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _valid(entity: str, text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if entity == "CREDIT_CARD":
        return 13 <= len(digits) <= 19 and _luhn(digits)
    if entity == "PHONE":
        return 10 <= len(digits) <= 15 or bool(re.search(_DW, text, re.I))
    if entity == "IBAN":
        return any(c.isdigit() for c in text[:4]) and len(text.replace(" ", "")) >= 15
    return True


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)  # entity -> distinct values, never the values themselves

    @property
    def found(self) -> bool:
        return bool(self.counts)


def _scan(text: str, replace: bool) -> tuple[str, dict[str, int]]:
    seen: dict[tuple[str, str], str] = {}
    n: dict[str, int] = {}

    def sub(entity: str):
        def f(m: re.Match[str]) -> str:
            if not _valid(entity, m.group(0)):
                return m.group(0)
            key = (entity, re.sub(r"\W", "", m.group(0).lower()))
            if key not in seen:
                n[entity] = n.get(entity, 0) + 1
                seen[key] = f"<{entity}_{n[entity]}>"
            return seen[key]
        return f

    for entity, pat in _PATTERNS:
        text = pat.sub(sub(entity), text)
    return text, n


def redact(text: str) -> Redaction:
    out, counts = _scan(normalize(text), True)
    return Redaction(out, counts)


def contains_pii(text: str) -> bool:
    """Residue check on already-redacted text; placeholders themselves never match."""
    return bool(_scan(normalize(text), False)[1])
