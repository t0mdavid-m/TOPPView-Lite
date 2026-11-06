"""Layout page: arrange the components of the Viewer page."""

from pathlib import Path

import streamlit as st

from src.common.common import page_setup
from src.components.viewer_panels import DEFAULT_LAYOUT, editor_catalog, layout_path
from src.layout_manager import MAX_EXPERIMENTS, layout_editor

params = page_setup()

st.title("Layout")
st.caption(
    "Arrange the components shown on the **👀 Viewer** page: drag them into the grid "
    "(up to 3 per row) and press **Save**. Add files to compare several, each with its "
    "own file picker on the Viewer, two of them side by side. Components that a file cannot show, such as "
    "identifications without a matching idXML, are left out for that file."
)

layout_editor(
    layout_path(Path(st.session_state.workspace)),
    editor_catalog(),
    DEFAULT_LAYOUT,
    key="viewer_layout_editor",
    max_experiments=MAX_EXPERIMENTS,
    experiment_label="File",
    target="the Viewer",
    export_filename="toppview-lite-layout.json",
)
