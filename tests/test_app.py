"""Gate 18 — the workbench renders all six tabs without errors, and a Tab 5 runner works."""
from streamlit.testing.v1 import AppTest


def test_workbench_renders_and_runs_the_cache_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DB", str(tmp_path / "state.db"))
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    next(b for b in at.button if b.label == "Cross-tenant cache probe").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Cache isolation" and m.value == "1.0000" for m in at.metric)
