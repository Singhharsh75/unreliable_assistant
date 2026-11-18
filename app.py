"""Developer workbench. Gate 2 scope: Tab 1 shows the workflow diagram; other tabs arrive with their gates."""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
DIAGRAM = ROOT / "docs" / "workflow_orchestration.png"

st.set_page_config(page_title="Unreliable Assistant Workbench", layout="wide")
(tab1,) = st.tabs(["Architecture & Workflow"])

with tab1:
    st.header("Workflow Orchestration")
    if DIAGRAM.exists():
        st.image(str(DIAGRAM), width="stretch")
    else:
        st.error("Diagram missing. Run: uv run --with graphviz --with pillow python scripts/render_workflow.py")
    with st.expander("design.md (Gate 1, locked)"):
        st.markdown((ROOT / "design.md").read_text())
