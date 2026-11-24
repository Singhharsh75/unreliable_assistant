"""Gate 18 — the workbench renders all six tabs without errors, and a Tab 5 runner works."""
from streamlit.testing.v1 import AppTest


def test_workbench_renders_and_runs_the_cache_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DB", str(tmp_path / "state.db"))
    from mcp_servers.state_store import StateStore
    StateStore(str(tmp_path / "state.db"))._audit("tenant_demo", {
        "type": "agent.query", "intent": "lot_status", "status": "DONE", "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
        "steps": [{"step": 1, "thought": "look it up", "tool": "lookup_lot", "args": {"lot_id": "LOT-00044"}, "outcome": "HOLD"}]})
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert any("lookup_lot" in m.value for m in at.markdown) and any("thought: look it up" in x.value for x in at.text)
    next(b for b in at.button if b.label == "Cross-tenant cache probe").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Cache isolation" and m.value == "1.0000" for m in at.metric)
