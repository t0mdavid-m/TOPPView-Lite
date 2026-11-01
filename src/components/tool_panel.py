"""Tool panels shared by the Tools page and the viewer.

``operation_panel`` renders an operation picker with a parameter form generated
from the pyOpenMS algorithm; ``console_panel`` renders the local-only Python
console. Both return the files they wrote during this run, so the viewer can
show new results right away.
"""

from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from src.console import DEFAULT_CODE, SNIPPETS, run_script
from src.operations import FEATURE_DIR_NAME, MZML_DIR_NAME, OPERATIONS, output_path, run_operation_isolated
from src.preprocessing import preprocess_file
from src.preprocessing.identification import find_matching_idxml

FLOAT_LIMIT = 1e300  # pyOpenMS uses +-DBL_MAX for "no bound"
INT_LIMIT = 2_000_000_000


def find_featurexml(workspace: Path, mzml_path: Path) -> Optional[Path]:
    """featureXML in the workspace with the same name as the mzML (case-insensitive)."""
    for path in sorted((Path(workspace) / FEATURE_DIR_NAME).glob("*")):
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


def operation_panel(workspace: Path, mzml_path: Path, key: str, columns: int = 2) -> List[Path]:
    """Operation picker, parameter form and Run button for ``mzml_path``.

    Returns the files written by a run in this script run (empty otherwise).
    A new mzML is preprocessed before returning, so the viewer can open it.
    """
    workspace = Path(workspace)
    op_names = {op.name: op for op in OPERATIONS.values()}
    operation = op_names[st.selectbox("Operation", list(op_names), key=f"{key}_operation")]
    st.caption(operation.description)

    show_advanced = st.toggle("Show advanced parameters", key=f"{key}_advanced")
    overrides: Dict[str, object] = {}
    with st.container(border=True):
        cols = st.columns(columns)
        visible = [e for e in operation.entries() if show_advanced or not e["advanced"]]
        for i, entry in enumerate(visible):
            with cols[i % columns]:
                value = _parameter_widget(entry, f"{key}_{operation.id}_{entry['key']}")
            if _changed(value, entry):
                overrides[entry["key"]] = value

    out_name = st.text_input(
        "Output file name",
        value=operation.default_output_name(mzml_path),
        key=f"{key}_out_{operation.id}_{mzml_path.stem}",
    )
    try:
        target, name_error = output_path(workspace, operation, out_name), None
    except ValueError as err:
        target, name_error = None, str(err)

    overwrite = False
    if name_error:
        st.error(name_error)
    elif target.exists():
        overwrite = st.checkbox(f"Replace the existing {target.name}", key=f"{key}_overwrite")
    elif operation.output_dir == FEATURE_DIR_NAME and target.stem.lower() == mzml_path.stem.lower():
        st.caption("The viewer shows this featureXML next to the mzML of the same name.")

    written: List[Path] = []
    result_key = f"{key}_result"
    can_run = target is not None and (overwrite or not target.exists())
    if st.button("Run", type="primary", disabled=not can_run, key=f"{key}_run"):
        with st.spinner(f"Running {operation.name}..."):
            try:
                summary = run_operation_isolated(operation.id, mzml_path, target, overrides)
                if target.suffix == ".mzML":
                    preprocess_file(target)
                st.session_state[result_key] = ("ok", f"{summary} Written to {target.name}.")
                written.append(target)
            except (RuntimeError, TimeoutError, ValueError) as err:
                st.session_state[result_key] = ("error", str(err))

    if result_key in st.session_state:
        status, message = st.session_state[result_key]
        (st.success if status == "ok" else st.error)(message)
    return written


def _snapshot(workspace: Path) -> Dict[Path, float]:
    files = {}
    for folder in (MZML_DIR_NAME, FEATURE_DIR_NAME):
        for path in (Path(workspace) / folder).glob("*"):
            if path.is_file():
                files[path] = path.stat().st_mtime
    return files


def console_panel(workspace: Path, mzml_path: Path, key: str, scan_id: Optional[int] = None) -> List[Path]:
    """Local Python console. Only call this when the app runs locally.

    Returns the mzML / featureXML files the code created or changed; new mzML
    files are preprocessed so the viewer can open them.
    """
    workspace = Path(workspace)
    code_key = f"{key}_code"

    def _load_snippet():
        st.session_state[code_key] = SNIPPETS[st.session_state[f"{key}_snippet"]]

    st.selectbox(
        "Snippet", list(SNIPPETS), index=None, placeholder="Load a snippet...",
        key=f"{key}_snippet", on_change=_load_snippet,
    )
    if code_key not in st.session_state:
        st.session_state[code_key] = DEFAULT_CODE
    code = st.text_area("Code", height=300, key=code_key)
    timeout = st.number_input("Time limit (s)", 10, 7200, 300, step=10, key=f"{key}_timeout")

    written: List[Path] = []
    if st.button("Run code", type="primary", key=f"{key}_run"):
        before = _snapshot(workspace)
        with st.spinner("Running..."):
            result = run_script(
                code, workspace, mzml_path,
                featurexml_path=find_featurexml(workspace, mzml_path),
                idxml_path=find_matching_idxml(mzml_path, workspace),
                scan_id=scan_id,
                timeout=float(timeout),
            )
        st.session_state[f"{key}_output"] = result
        after = _snapshot(workspace)
        written = sorted(p for p, mtime in after.items() if before.get(p) != mtime)
        for path in written:
            if path.suffix == ".mzML":
                preprocess_file(path)

    result = st.session_state.get(f"{key}_output")
    if result is not None:
        if result.returncode != 0:
            st.error("Time limit reached." if result.timed_out else "The code stopped with an error.")
        st.code(result.output or "(no output)", language=None)
    return written
