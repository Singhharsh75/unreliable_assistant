"""Plugin registry: mcp_config.json orders and extends the pre-hooks, and cannot weaken the guardrails."""
import copy
import json

import pytest

from guardrails import run_prehooks
from harness import plugins
from harness.errors import INJECTION_BLOCKED, HarnessError

BASE = json.loads(plugins.CONFIG.read_text())


def config(edit):
    c = copy.deepcopy(BASE)
    edit({p["name"]: p for p in c["plugins"]}, c)
    return plugins.PluginConfig.model_validate(c)


def test_the_shipped_config_runs_the_redactor_then_the_shield():
    assert [p.name for p in plugins.load().stage("PRE_EXECUTION")] == ["pii_redactor", "injection_shield"]
    assert {p.name for p in plugins.load().stage("PRE_EXECUTION")} <= set(plugins.HOOKS)


@pytest.mark.parametrize("edit", [
    lambda p, c: c["plugins"].remove(p["injection_shield"]),
    lambda p, c: p["pii_redactor"].update(enabled=False),
    lambda p, c: p["injection_shield"].update(fail_closed=False),
    lambda p, c: p["injection_shield"].update(stage="POST_EXECUTION"),
    lambda p, c: p["injection_shield"].update(priority=1),
    lambda p, c: c["plugins"].insert(0, {"name": "early", "stage": "PRE_EXECUTION", "priority": 5}),
    lambda p, c: c["plugins"].append(dict(p["pii_redactor"])),
    lambda p, c: p["circuit_breaker"].update(max_identical_calls=50),
    lambda p, c: p["circuit_breaker"].update(enabled=False),
    lambda p, c: p["pii_redactor"].update(typo=1),
])
def test_a_config_that_weakens_a_guardrail_is_refused(edit):
    with pytest.raises(ValueError):
        config(edit)


def test_a_registered_plugin_runs_after_the_guardrails_and_fails_as_configured(monkeypatch):
    seen = []

    def extra(r):
        seen.append(r.text)
        raise RuntimeError("boom")

    monkeypatch.setattr(plugins, "HOOKS", {**plugins.HOOKS, "extra": (extra, INJECTION_BLOCKED)})
    for fail_closed in (False, True):
        cfg = config(lambda p, c: c["plugins"].append(
            {"name": "extra", "stage": "PRE_EXECUTION", "priority": 30, "fail_closed": fail_closed}))
        monkeypatch.setattr(plugins, "load", lambda: cfg)
        if fail_closed:
            with pytest.raises(HarnessError) as e:
                run_prehooks("Mail a@b.com about LOT-00044")
            assert e.value.code == INJECTION_BLOCKED
        else:
            assert "LOT-00044" in run_prehooks("Mail a@b.com about LOT-00044").text
    assert len(seen) == 2 and not any("a@b.com" in t for t in seen)
