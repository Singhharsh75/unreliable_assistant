"""Injection shield (design.md §1.2.2 step 2, KPI-03): scores text against weighted attack patterns after
canonicalization (NFKC, zero-width and homoglyph folding, leetspeak, letter-spacing, base64/hex/ROT13 decoding).

A strong pattern scores 1.0, a weak one 0.5; a score >= BLOCK_AT blocks. `scan` is used on user text (direct) and on
retrieved chunks or tool output (indirect, where imperative text aimed at the model is also strong).

shortcut: pattern scoring stands in for a trained classifier; it is blind to attacks with no listed phrasing. The
red-team benchmark (eval/redteam) measures that gap, and a classifier should replace this if KPI-03 slips.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
from dataclasses import dataclass

from guardrails.pii import normalize

BLOCK_AT = 1.0
_HOMOGLYPH = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ѕ": "s", "ј": "j", "һ": "h",
    "ԁ": "d", "ɡ": "g", "ո": "n", "ο": "o", "ν": "v", "ι": "i", "α": "a", "ρ": "p", "τ": "t", "κ": "k", "ɑ": "a",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_VERB = r"(?:ignore|disregard|forget|override|discard|bypass|skip|overrule|abandon|neglect|drop)"
_OBJ = r"(?:instructions?|rules?|prompts?|guidelines?|directions?|constraints?|guardrails?|polic(?:y|ies)|restrictions?|directives?|programming|training)"
_MODEL = r"(?:assistant|ai|llm|model|chatbot|agent|claude)"

STRONG = [re.compile(p, re.I | re.M) for p in [
    rf"\b{_VERB}\b[^.\n]{{0,40}}\b{_OBJ}\b",
    rf"\b(?:ignora|ignore[sz]?|ignoriere|ignorare|ignorem|olvida|oubliez|ignorer|vergiss)\b[^.\n]{{0,40}}\b(?:instrucciones|instructions|anweisungen|istruzioni|instru[cç][oõ]es|regeln|reglas|r[eè]gles|prompts?)\b",
    r"\byou (?:are|r) (?:now|no longer)\b",
    r"\b(?:developer|dev|god|admin|root|sudo|debug|unrestricted|jailbreak) mode\b",
    r"\bjail ?break(?:ing|en)?\b",
    r"\bdo anything now\b|\bdan mode\b|\bact as dan\b",
    r"\bno (?:restrictions|filters|limitations|guardrails|rules|censorship)\b",
    r"\bwithout (?:any )?(?:restrictions|filters|limitations|guardrails|censorship)\b",
    r"\b(?:reveal|print|show|repeat|display|output|leak|dump|recite|expose|echo|disclose|share)\b[^.\n]{0,30}\b(?:system|hidden|initial|original|secret|internal|full|entire|above|your)\b[^.\n]{0,12}\b(?:prompt|instructions?|message|rules|configuration|context)\b",
    r"\bwhat (?:is|are|were) your (?:system |initial |original |hidden )?(?:prompt|instructions|rules|directives)\b",
    r"</?(?:system|assistant|instructions?|prompt)>|\[/?inst\]|<\|(?:im_start|im_end|system|endoftext)\|>|^\s*#{2,}\s*(?:system|instructions?|new instructions?)|^\s*(?:system|assistant)\s*:|\bsystem\s*:\s*you\b",
    r"\b(?:approve|release|quarantine|delete|submit|execute|run|ship|sign)\b[^.\n]{0,50}\bwithout\b[^.\n]{0,15}\b(?:approval|review|verification|confirmation|human|hitl|sign-?off|authori[sz]ation|checks?)\b",
    r"\b(?:skip|bypass|disable|circumvent|turn off|switch off|evade|override|suppress)\b[^.\n]{0,20}\b(?:approval|verification|hitl|human review|guardrails?|safety|filters?|pii|redaction|checks?|four-eyes|abstention)\b",
    r"\b(?:show|list|dump|export|access|read|fetch|retrieve|give me|query)\b[^.;\n]{0,40}\b(?:other|another|all|every|different)\s+(?:tenants?|customers?\W+(?:data|records))\b",
    r"\btenant_id\b\s*(?:=|:|to|as)\s*(?!<)\S+[^.\n]{0,40}\b(?:instead|switch|change|use)\b|\b(?:switch|change|set|use)\b[^.\n]{0,20}\btenant[_ ]?id\b",
    rf"\b(?:send|post|forward|exfiltrate|email|upload|transmit)\b[^.\n]{{0,50}}\b(?:to|at)\b\s*(?:https?://|\S+@\S+)",
    r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*\)",
    rf"\b{_MODEL}\b[,:]?\s*(?:you )?(?:must|should|need to|are required to|will now|now)?\s*(?:ignore|disregard|forget|override|instead|stop|reveal|do not (?:tell|mention|follow))\b",
    rf"\b(?:important|attention|urgent|note|message|instruction)s?\b[^.\n]{{0,15}}\b(?:to|for)\b[^.\n]{{0,6}}\b{_MODEL}\b",
    r"\binstead (?:of (?:answering|the above)\W+)?(?:say|respond|reply|output|print|write|answer)\b",
    r"\b(?:new|updated|real|actual|revised) (?:system )?(?:instructions?|task|rules|directives)\b\s*[:\-]",
    r"\bfrom now on\b[^.\n]{0,40}\b(?:you|ignore|always|never|respond|answer)\b",
    r"\bpretend (?:that )?(?:you|to be|there)\b[^.\n]{0,40}\b(?:no|not|without|are|can|have)\b",
    r"\b(?:disable|turn off|remove|deactivate|unlock)\b[^.\n]{0,15}\b(?:your |the )?(?:safety|content|safeguards?|moderation|filter)\b",
    r"\bthis (?:text|document|message|note|chunk|passage|file) (?:is|contains) (?:an )?(?:instruction|command|directive)\b",
    r"\bwhen(?:ever)? (?:you )?(?:read|see|process|summari[sz]e|retrieve)s?\b[^.\n]{0,20}\bthis\b[^.\n]{0,30}\b(?:you|then|must|should|ignore)\b",
]]
WEAK = [re.compile(p, re.I | re.M) for p in [
    r"\b(?:act|behave|respond|speak) as (?:if|a|an|though)\b", r"\broleplay|role-play\b", r"\bhypothetically\b",
    r"\bfor (?:educational|research|testing) purposes\b", r"\bin a fictional\b|\bin a story where\b",
    r"\bsystem prompt\b", r"\byour (?:instructions|programming|rules|guidelines)\b", r"\bdecode (?:the|this) following\b",
    r"\bbase ?64\b|\brot ?13\b|\bhex(?:adecimal)? encoded\b", r"\bthe (?:above|previous) (?:text|message|instructions?)\b",
    r"\bmust (?:comply|obey|follow)\b", r"\bat all costs\b|\bno matter what\b", r"\bsecret (?:password|key|token)\b|\bapi[_ ]?key\b",
    r"\bnever (?:refuse|decline|say no)\b", r"\bnot (?:bound|restricted|limited) by\b", r"\boverride\b",
    r"\bconfidential\b[^.\n]{0,30}\b(?:reveal|share|tell|show)\b", r"\bdo not (?:mention|tell|inform|alert|notify)\b[^.\n]{0,20}\b(?:user|human|anyone|qa|reviewer)\b",
]]
# Indirect only (retrieved chunks, tool output). Reference text describes procedures; it does not speak as a requester,
# ask for approvals to be skipped, name another tenant as a target, or tell the assistant to hide things from reviewers.
# Written by concept (action + removed control, requester voice, cross-tenant target, concealment), not by sentence.
_ACT = r"(?:mark|set|release|execute|quarantine|approve|ship|sign|submit|delete|run|go ahead|proceed|push|promote|reject)"
_NOCTRL = r"(?:sign[- ]?off|four[- ]eyes|approvals?|re-?view(?:er|s)?|second (?:person|pair)|authori[sz]ation|co-?sign\w*|checks?|oversight)"
INDIRECT = [re.compile(p, re.I | re.M) for p in [
    rf"\b{_ACT}\b[^.\n]{{0,80}}\b(?:no need|not (?:needed|necessary|required)|unnecessary|no time|skip|without|don'?t need|do not need|there'?s no)\b[^.\n]{{0,40}}\b{_NOCTRL}\b",
    rf"\b(?:no need|no time|not (?:needed|necessary)|there'?s no)\b[^.\n]{{0,40}}\b{_NOCTRL}\b[^.\n]{{0,30}}\b{_ACT}\b",
    rf"\b{_NOCTRL}\b[^.\n]{{0,25}}\b(?:(?:are|is) )?(?:not (?:needed|necessary|required)|unnecessary)\b[^.\n]{{0,60}}|\b{_ACT}\b[^.\n]{{0,60}}\b{_NOCTRL}\b[^.\n]{{0,15}}\b(?:not (?:needed|necessary|required)|unnecessary)\b",
    r"\b(?:don'?t|do not|never) (?:let|tell|inform)\b[^.\n]{0,15}\b(?:qa|reviewers?|approvers?|anyone|them)\b[^.\n]{0,10}\b(?:know|about)\b",
    r"\b(?:ignore|forget|disregard)\b[^.\n]{0,15}\b(?:what|everything|anything) you (?:were|have been|had been) (?:told|given|taught)\b|\b(?:act|respond|answer|operate)\b[^.\n]{0,15}\bwithout (?:any )?(?:limits?|limitations|boundaries)\b|\bprint\b[^.\n]{0,10}\byour (?:setup|configuration|initial) (?:text|prompt|message)\b",
    r"\b(?:switch|move|go|change|jump|hop)\b[^.\n]{0,25}\b(?:other|another|different)\s+tenants?\b",
    r"\b(?:give|show|send|tell|list|dump|export|fetch)\s+(?:me|us)\b[^.\n]{0,50}\btenant[_ ]?\w*",
    r"\b(?:give|show|send|tell|list|dump|export)\s+(?:me|us)\b[^.\n]{0,20}\b(?:what|all|every|their|the)\b[^.\n]{0,40}\b(?:customers?|lots|records|data|secrets?|configuration|credentials)\b",
    r"\bdo not (?:tell|inform|notify|alert|mention|show|let)\b[^.\n]{0,25}\b(?:reviewers?|approvers?|qa|the user|humans?|anyone|auditors?)\b",
    r"\b(?:approve|release|ship|submit|execute|do)\b[^.\n]{0,30}\b(?:quietly|silently|secretly|discreetly|without (?:a )?(?:trace|log|record))\b",
    r"\bI['’]?m (?:the|your) (?:platform|system|site|tenant|root)?\s*(?:owner|admin\w*|operator|developer|boss)\b|\bI am (?:the|your) (?:platform |system |site |tenant )?(?:owner|admin\w*|operator|developer)\b",
    r"\bjust do what I say\b|\b(?:I|we) (?:need|want) you to\b|\b(?:set aside|stop following|put aside)\b[^.\n]{0,30}\b(?:guidance|polic(?:y|ies)|instructions?|rules)\b",
    r"\b(?:before|prior to) my (?:first|previous|earlier) (?:message|prompt|question)\b|\byou were (?:initiali[sz]ed|configured|given|told)\b[^.\n]{0,20}\b(?:with|by|to)\b",
    r"\b(?:stop being|no longer be|stop acting as|become|turn into)\b[^.\n]{0,40}\b(?:assistant|system|bot|model|ai)\b|\bexpos(?:e|es|ing)\b[^.\n]{0,15}\b(?:its|your|the) (?:configuration|config|prompt|instructions?|rules)\b",
    r"\b(?:never|who never) refuses?\b|\bhas no limits\b",
]]
_B64 = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_HEX = re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{2}[\s:]?){16,}(?![0-9a-f])", re.I)
_WIDE = re.compile(r"(?:\S(?:   | )){4,}")  # "I g n o r e   a l l": one space inside a word, three between words
_SPACED = re.compile(r"(?<![a-z0-9])(?:[a-z][ ._\-*]){3,}[a-z](?![a-z0-9])", re.I)


@dataclass(frozen=True)
class Scan:
    score: float
    hits: tuple[str, ...]
    blocked: bool


def _fold(text: str) -> str:
    t = normalize(text).translate(_HOMOGLYPH)
    t = _WIDE.sub(lambda m: m.group(0).replace("   ", "\0").replace(" ", "").replace("\0", " "), t)
    t = _SPACED.sub(lambda m: re.sub(r"[ ._\-*]", "", m.group(0)), t)

    def leet(m: re.Match[str]) -> str:  # only inside tokens that also hold letters, so "8 °C" and "LOT-00044" survive
        w = m.group(0)
        return w.translate(_LEET) if re.search(r"[a-z]", w, re.I) and re.search(r"[0-9@$!]", w) and not re.fullmatch(r"[A-Z]{2,}-\d+", w) and not re.search(r"\w@\w", w) else w
    return re.sub(r"[\w@$!]+", leet, t)


def _decoded(text: str) -> list[str]:
    out = [codecs.decode(text, "rot13")]
    for m in _B64.finditer(text):
        try:
            s = base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if s.isprintable() or "\n" in s:
            out.append(s)
    for m in _HEX.finditer(text):
        try:
            out.append(bytes.fromhex(re.sub(r"[\s:]", "", m.group(0))).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
    return out


def _score(text: str, weak: bool = True) -> tuple[float, list[str]]:
    hits = [f"S{i}" for i, p in enumerate(STRONG) if p.search(text)]
    score = float(len(hits) > 0)
    if weak:
        w = [f"W{i}" for i, p in enumerate(WEAK) if p.search(text)]
        score += 0.5 * len(w) + (0.5 * (len(hits) - 1) if len(hits) > 1 else 0)
        hits += w
    return score, hits


def scan(text: str, indirect: bool = False) -> Scan:
    """`indirect=True` for retrieved chunks and tool output: the INDIRECT patterns count as strong there."""
    folded = _fold(text)
    score, hits = _score(folded)
    if indirect:
        ih = [f"I{i}" for i, p in enumerate(INDIRECT) if p.search(folded)]
        if ih:
            score, hits = max(score, BLOCK_AT), list(hits) + ih
    for d in _decoded(text) + _decoded(folded):
        s, h = _score(_fold(d), weak=False)  # decoded payloads count only on strong patterns (ROT13 of prose is noise)
        if s >= BLOCK_AT:
            score, hits = max(score, BLOCK_AT + 0.5), hits + [f"decoded:{x}" for x in h]
            break
    return Scan(score, tuple(hits), score >= BLOCK_AT)
