"""Pre-execution hooks (design.md §1.2.2 step 2): PII redaction, then injection shield. Both fail closed."""
from __future__ import annotations

from dataclasses import dataclass

from guardrails import classifier, injection, pii
from harness.errors import INJECTION_BLOCKED, PII_BLOCKED, HarnessError
from observability.tracing import span

SAFE_REFUSAL = "I can't help with that request."


@dataclass(frozen=True)
class PreResult:
    text: str  # redacted; the only text allowed past this point
    pii: dict[str, int]
    injection_score: float


def run_prehooks(text: str) -> PreResult:
    """Raises HarnessError(PII_BLOCKED | INJECTION_BLOCKED). Any unexpected failure also blocks (FMEA #11)."""
    with span("hook.pre") as s:
        try:
            red = pii.redact(text)
            if pii.contains_pii(red.text):
                raise HarnessError(PII_BLOCKED, "PII remained after redaction")
        except HarnessError:
            s.set_attribute("blocked", "PII_BLOCKED")
            raise
        except Exception as e:
            s.set_attribute("blocked", "PII_BLOCKED")
            raise HarnessError(PII_BLOCKED, f"redactor failed: {type(e).__name__}") from e
        try:
            sc = injection.scan(red.text)
        except Exception as e:
            s.set_attribute("blocked", "INJECTION_BLOCKED")
            raise HarnessError(INJECTION_BLOCKED, f"shield failed: {type(e).__name__}") from e
        if sc.blocked:
            s.set_attribute("blocked", "INJECTION_BLOCKED")
            raise HarnessError(INJECTION_BLOCKED, "injection detected", {"score": sc.score, "rules": list(sc.hits)})
        s.set_attribute("pii_entities", sum(red.counts.values()))
        return PreResult(red.text, red.counts, sc.score)


def is_poisoned(chunk_text: str) -> bool:
    """Indirect injection: retrieved or tool-supplied text that tries to instruct the model. Regex shield, then the
    classifier on the canonicalized text. Fails closed."""
    try:
        return (injection.scan(chunk_text, indirect=True).blocked
                or classifier.injection_prob(injection._fold(chunk_text)) >= classifier.THRESHOLD)
    except Exception:
        return True
