"""Tools page: processing operations, and a Python console in the local app.

Operations are fixed pyOpenMS pipelines with parameter forms, so the hosted app
never runs user code. The Python console runs arbitrary code and is therefore
shown only when settings.json has ``online_deployment: false``.
"""

from pathlib import Path

import streamlit as st
from openms_insight import StateManager

from src.common.common import page_setup
from src.console import DEFAULT_CODE, SNIPPETS, run_script
from src.operations import OPERATIONS, output_path, run_operation_isolated
from src.preprocessing import preprocess_file
from src.preprocessing.identification import find_matching_idxml

FLOAT_LIMIT = 1e300  # pyOpenMS uses +-DBL_MAX for "no bound"
INT_LIMIT = 2_000_000_000

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


def _find_featurexml(mzml_path: Path):
    for path in sorted((workspace / "featureXML-files").glob("*")):
        if path.suffix.lower() == ".featurexml" and path.stem.lower() == mzml_path.stem.lower():
            return path
    return None


def _parameter_widget(entry: dict, widget_key: str):
    label = entry["key"]
    help_text = entry["description"] or None
    if entry["type"] == "bool":
        return st.checkbox(label, value=entry["value"], help=help_text, key=widget_key)
    if entry["type"] == "string":
        if entry["choices"]:
            return st.selectbox(
                label, entry["choices"], index=entry["choices"].index(entry["value"]),
                help=help_text, key=widget_key,
            )
        return st.text_input(label, value=entry["value"], help=help_text, key=widget_key)
    if entry["type"] == "int":
        lo = entry["min"] if entry["min"] > -INT_LIMIT else None
        hi = entry["max"] if entry["max"] < INT_LIMIT else None
        return st.number_input(
            label, value=int(entry["value"]), min_value=lo, max_value=hi, step=1,
            help=help_text, key=widget_key,
        )
    if entry["type"] == "float":
        lo = entry["min"] if entry["min"] > -FLOAT_LIMIT else None
        hi = entry["max"] if entry["max"] < FLOAT_LIMIT else None
        return st.number_input(
            label, value=float(entry["value"]), min_value=lo, max_value=hi, format="%g",
            help=help_text, key=widget_key,
        )
    return st.text_input(
        f"{label} (comma-separated)", value=", ".join(str(v) for v in entry["value"]),
        help=help_text, key=widget_key,
    )


def _changed(value, entry: dict) -> bool:
    if entry["type"] == "list":
        return value.strip() != ", ".join(str(v) for v in entry["value"])
    return value != entry["value"]


is_local = st.session_state.location == "local"
tab_names = ["Operations"] + (["Python console"] if is_local else [])
tabs = st.tabs(tab_names)

# =============================================================================
# Operations
# =============================================================================

with tabs[0]:
    op_names = {op.name: op for op in OPERATIONS.values()}
    operation = op_names[st.selectbox("Operation", list(op_names), key="tools_operation")]
    st.caption(operation.description)

    entries = operation.entries()
    show_advanced = st.toggle("Show advanced parameters", key="tools_advanced")
    overrides = {}
    with st.container(border=True):
        cols = st.columns(2)
        visible = [e for e in entries if show_advanced or not e["advanced"]]
        for i, entry in enumerate(visible):
            with cols[i % 2]:
                value = _parameter_widget(entry, f"tools_{operation.id}_{entry['key']}")
            if _changed(value, entry):
                overrides[entry["key"]] = value

    out_name = st.text_input(
        "Output file name",
        value=operation.default_output_name(selected_file),
        key=f"tools_out_{operation.id}_{selected_file.stem}",
    )
    try:
        target = output_path(workspace, operation, out_name)
        name_error = None
    except ValueError as err:
        target, name_error = None, str(err)

    overwrite = False
    if name_error:
        st.error(name_error)
    elif target.exists():
        overwrite = st.checkbox(f"Replace the existing {target.name}", key="tools_overwrite")
    elif operation.output_dir == "featureXML-files" and target.stem.lower() == selected_file.stem.lower():
        st.caption("The viewer shows this featureXML next to the mzML of the same name.")

    result_key = "tools_operation_result"
    can_run = target is not None and (overwrite or not target.exists())
    if st.button("Run", type="primary", disabled=not can_run, key="tools_run"):
        with st.spinner(f"Running {operation.name}..."):
            try:
                summary = run_operation_isolated(operation.id, selected_file, target, overrides)
                st.session_state[result_key] = ("ok", f"{summary} Written to {target.name}.")
                if target.suffix == ".mzML":
                    preprocess_file(target)
            except (RuntimeError, TimeoutError, ValueError) as err:
                st.session_state[result_key] = ("error", str(err))

    if result_key in st.session_state:
        status, message = st.session_state[result_key]
        (st.success if status == "ok" else st.error)(message)

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

        def _load_snippet():
            st.session_state["tools_console_code"] = SNIPPETS[st.session_state["tools_snippet"]]

        st.selectbox(
            "Snippet", list(SNIPPETS), index=None, placeholder="Load a snippet...",
            key="tools_snippet", on_change=_load_snippet,
        )
        if "tools_console_code" not in st.session_state:
            st.session_state["tools_console_code"] = DEFAULT_CODE
        code = st.text_area("Code", height=300, key="tools_console_code")
        timeout = st.number_input("Time limit (s)", 10, 7200, 300, step=10, key="tools_console_timeout")

        if st.button("Run code", type="primary", key="tools_console_run"):
            feature_path = _find_featurexml(selected_file)
            with st.spinner("Running..."):
                result = run_script(
                    code, workspace, selected_file,
                    featurexml_path=feature_path,
                    idxml_path=find_matching_idxml(selected_file, workspace),
                    scan_id=scan_id,
                    timeout=float(timeout),
                )
            st.session_state["tools_console_result"] = result

        result = st.session_state.get("tools_console_result")
        if result is not None:
            if result.returncode != 0:
                st.error("The code stopped with an error." if not result.timed_out else "Time limit reached.")
            st.code(result.output or "(no output)", language=None)
            st.caption("Files saved with save_mzml() appear on the Upload page and need processing there.")
