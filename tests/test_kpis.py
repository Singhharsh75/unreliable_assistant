"""Gate 12 — KPI matrix: all 36 KPIs reported; the in-code suites pass; gate policy (safety, waivers)."""
from eval import kpis


def test_matrix_has_all_36_kpis_in_order():
    assert [k.id for k in kpis.KPIS] == [f"KPI-{i:02d}" for i in range(1, 37)]


def test_in_code_suites_pass():
    results = [*kpis.triage_kpis(), kpis.repair_kpi(), kpis.chaos_kpi(), kpis.hitl_kpi(), kpis.serialization_kpi()]
    bad = [(r.id, r.value, r.note) for r in results if r.status != "PASS"]
    assert not bad, bad
    assert {r.id for r in results} == {"KPI-04", "KPI-05", "KPI-06", "KPI-15", "KPI-16", "KPI-19", "KPI-30"}


def test_safety_kpis_are_never_waived():
    r = kpis.judge("KPI-24", 0.2, waived_at=0.0)
    assert r.status == "FAIL" and r.blocking


def test_waiver_applies_only_between_waived_gate_and_target():
    assert kpis.judge("KPI-20", 0.80, waived_at=0.737).status == "WAIVED"
    assert kpis.judge("KPI-20", 0.70, waived_at=0.737).status == "FAIL"
    assert kpis.judge("KPI-20", 0.95, waived_at=0.737).status == "PASS"


def test_unmeasured_blocking_kpi_blocks_release():
    results = [kpis.judge(k.id, k.target) for k in kpis.KPIS]
    assert kpis.summarize(results)["release_ready"]
    results[1] = kpis.judge("KPI-02", None)
    s = kpis.summarize(results)
    assert not s["release_ready"] and s["release_blockers"] == ["KPI-02"] and s["failing"] == []


def test_ir_metrics():
    assert kpis._mrr(["a", "g"], {"g"}) == 0.5
    assert kpis._ndcg(["g", "x"], {"g"}) == 1.0
    assert kpis._ndcg(["x", "g"], {"g"}) < 1.0


def test_load_kpis_are_judged_only_at_scale(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(kpis, "RUNS", tmp_path)
    lat = {"med": 100, "p(95)": 200, "p(99)": 300}
    summary = {"vus_max": 10, "rps": 9000, "latency_ms": lat, "server_5xx_rate": 0.0}
    (tmp_path / "k6_summary.json").write_text(json.dumps(summary))
    (tmp_path / "api_metrics.json").write_text(json.dumps({"generator": "StubGenerator", "llm_ttft_ms": {"p95": 1}}))
    got = {r.id: r for r in kpis.load_kpis()}
    assert {got[k].status for k in ("KPI-25", "KPI-31", "KPI-35", "KPI-36")} == {"DEFERRED", "NOT_MEASURED"}
    assert "measured 10 VUs" in got["KPI-31"].note
    (tmp_path / "k6_summary.json").write_text(json.dumps({**summary, "vus_max": 50_000}))
    got = {r.id: r for r in kpis.load_kpis()}
    assert got["KPI-31"].status == "PASS" and got["KPI-35"].status == "PASS"


def test_cpu_waiver_is_a_ceiling_and_needs_cpu_only_hardware(monkeypatch):
    import onnxruntime

    lat = {"dense": [200.0] * 5, "rerank": [5000.0] * 5}
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert [r.status for r in kpis._model_latency(lat, "t")] == ["WAIVED", "FAIL"]  # rerank is past the ceiling
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    assert [r.status for r in kpis._model_latency(lat, "t")] == ["FAIL", "FAIL"]


def test_deferred_kpis_complete_the_environment_but_never_the_release():
    results = [kpis.judge(k.id, None if k.id in kpis.DEFERRED else k.target) for k in kpis.KPIS]
    s = kpis.summarize(results)
    assert s["counts"]["DEFERRED"] == 8 and s["environment_complete"] and not s["release_ready"]
    assert kpis.judge("KPI-31", 9999).status == "FAIL"  # a deferred KPI that is measured is judged


def test_goal_alignment_is_one_for_separable_tasks_and_half_for_identical_ones(monkeypatch):
    import numpy as np

    from eval import planner_eval
    from mcp_servers import rag_server

    vec = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    monkeypatch.setattr(rag_server, "embed", lambda ts: np.array([vec[t] for t in ts]))
    rows = [{"goal": "a", "steps": ["a"]}, {"goal": "b", "steps": ["b"]}, {"goal": "a", "steps": []}]
    assert planner_eval.alignment(rows)["kpi18_auc"] == 1.0
    assert planner_eval.alignment([{"goal": "a", "steps": ["a"]}, {"goal": "a", "steps": ["a"]}])["kpi18_auc"] == 0.5
