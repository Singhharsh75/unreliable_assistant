"""Developer workbench. Tab 1: workflow diagram (Gate 2). KPI Matrix: the 36 KPIs (Gate 12). Other tabs arrive with their gates."""
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
DIAGRAM = ROOT / "docs" / "workflow_orchestration.png"
KPI_REPORT = ROOT / "eval" / "runs" / "kpi_report.json"

st.set_page_config(page_title="Unreliable Assistant Workbench", layout="wide")
tab1, tab_kpi = st.tabs(["Architecture & Workflow", "KPI Matrix"])

with tab1:
    st.header("Workflow Orchestration")
    if DIAGRAM.exists():
        st.image(str(DIAGRAM), width="stretch")
    else:
        st.error("Diagram missing. Run: uv run --with graphviz --with pillow python scripts/render_workflow.py")
    with st.expander("design.md (Gate 1, locked)"):
        st.markdown((ROOT / "design.md").read_text())

with tab_kpi:
    st.header("36 Enterprise KPIs")
    if not KPI_REPORT.exists():
        st.error("No KPI report. Run: uv run eval/kpis.py --live")
    else:
        report = json.loads(KPI_REPORT.read_text())
        summary = report["summary"]
        cols = st.columns(5)
        for col, (label, n) in zip(cols, summary["counts"].items()):
            col.metric(label.replace("_", " ").title(), n)
        cols[4].metric("Release ready", "yes" if summary["release_ready"] else "no")
        if summary["release_blockers"]:
            st.warning("Safety KPIs blocking release: " + ", ".join(summary["release_blockers"]))
        st.dataframe(
            [{"KPI": k["id"], "Stage": k["stage"], "Name": k["name"], "Target": k["target"], "Value": k["value"],
              "Status": k["status"], "Safety": "yes" if k["blocking"] else "", "Source": k["source"], "Note": k["note"]}
             for k in report["kpis"]],
            hide_index=True, width="stretch")
