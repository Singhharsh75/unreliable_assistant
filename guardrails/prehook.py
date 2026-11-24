"""Pre-execution hooks (design.md §1.2.2 step 2): PII redaction, then injection shield. Both fail closed.
They run as PRE_EXECUTION plugins in the order mcp_config.json gives (harness/plugins.py)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from guardrails import classifier, injection, pii
from harness import plugins
from harness.errors import CODE_NAMES, INJECTION_BLOCKED, PII_BLOCKED, HarnessError
from observability.tracing import span

SAFE_REFUSAL = "I can't help with that request."
log = logging.getLogger("guardrails.prehook")


@dataclass
class PreResult:
    text: str  # redacted; the only text allowed past the hooks
    pii: dict[str, int] = field(default_factory=dict)
    injection_score: float = 0.0


@plugins.hook("pii_redactor", PII_BLOCKED)
def _redact(r: PreResult) -> None:
    red = pii.redact(r.text)
    if pii.contains_pii(red.text):
        raise HarnessError(PII_BLOCKED, "PII remained after redaction")
    r.text, r.pii = red.text, red.counts


@plugins.hook("injection_shield", INJECTION_BLOCKED)
def _shield(r: PreResult) -> None:
    sc = injection.scan(r.text)
    if sc.blocked:
        raise HarnessError(INJECTION_BLOCKED, "injection detected", {"score": sc.score, "rules": list(sc.hits)})
    r.injection_score = sc.score


def run_prehooks(text: str) -> PreResult:
    """Raises HarnessError(PII_BLOCKED | INJECTION_BLOCKED). Any unexpected failure also blocks (FMEA #11)."""
    with span("hook.pre") as s:
        r = PreResult(text)
        for p in plugins.load().stage("PRE_EXECUTION"):
            hook, code = plugins.HOOKS.get(p.name, (None, INJECTION_BLOCKED))
            try:
                if hook is None:
                    raise LookupError(f"plugin {p.name} has no registered hook")
                hook(r)
            except HarnessError as e:
                s.set_attribute("blocked", e.name)
                raise
            except Exception as e:
                if not p.fail_closed:
                    log.exception("plugin %s failed; it is not fail_closed, so the request continues", p.name)
                    continue
                s.set_attribute("blocked", CODE_NAMES[code])
                raise HarnessError(code, f"{p.name} failed: {type(e).__name__}") from e
        s.set_attribute("pii_entities", sum(r.pii.values()))
        return r


def is_poisoned(chunk_text: str) -> bool:
    """Indirect injection: retrieved or tool-supplied text that tries to instruct the model. Regex shield, then the
    classifier on the canonicalized text. Fails closed."""
    try:
        return (injection.scan(chunk_text, indirect=True).blocked
                or classifier.injection_prob(injection._fold(chunk_text)) >= classifier.THRESHOLD)
    except Exception:
        return True
