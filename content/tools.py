"""Tools page: processing operations, and a Python console in the local app.

Operations are fixed pyOpenMS pipelines with parameter forms, so the hosted app
never runs user code. The Python console runs arbitrary code and is therefore
shown only when settings.json has ``online_deployment: false``. The viewer page
offers the same panels for the file it shows.
"""

from pathlib import Path

import streamlit as st
from openms_insight import StateManager

from src.common.common import page_setup
from src.components.tool_panel import console_panel, operation_panel

params = page_setup()

st.title("Tools")

workspace = Path(st.session_state.workspace)
mzml_dir = workspace / "mzML-files"
mzml_files = sorted(p for p in mzml_dir.glob("*.mzML"))

if not mzml_files:
    st.info("Upload an mzML file first.")
    st.stop()

default_name = st.session_state.get("viewer_current_file")
names = [p.name for p in mzml_files]
selected_name = st.selectbox(
    "mzML file",
    names,
    index=names.index(default_name) if default_name in names else 0,
    key="tools_mzml",
)
selected_file = mzml_dir / selected_name


is_local = st.session_state.location == "local"
tab_names = ["Operations"] + (["Python console"] if is_local else [])
tabs = st.tabs(tab_names)

# =============================================================================
# Operations
# =============================================================================

with tabs[0]:
    operation_panel(workspace, selected_file, key="tools")

# =============================================================================
# Python console (local app only)
# =============================================================================

if is_local:
    with tabs[1]:
        st.caption(
            "Runs Python with pyOpenMS on your machine, in a separate process. "
            "This tab exists only in the local app."
        )
        scan_id = None
        if st.session_state.get("viewer_current_file") == selected_name:
            scan_id = StateManager(session_key="viewer_state").get_selection("spectrum")
        written = console_panel(workspace, selected_file, key="tools_console", scan_id=scan_id)
        if written:
            st.caption("Wrote " + ", ".join(p.name for p in written) + ". Open them on the Viewer page.")
