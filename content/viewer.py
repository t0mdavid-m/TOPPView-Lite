"""Interactive mzML Viewer Page.

This page displays preprocessed mzML files using openms_insight
with cross-component selection linking.

The components (src/components/viewer_panels.py) are arranged with Edit layout
(or on the Layout page); without a saved layout the viewer uses DEFAULT_LAYOUT
there. Each experiment of the layout shows the mzML file picked above it, with
its own StateManager, so several files can be compared side by side.
"""

from pathlib import Path

import streamlit as st
import polars as pl
import pyarrow.parquet as pq
from openms_insight import StateManager
from openms_insight.rendering.bridge import clear_component_cache

from src.common.common import page_setup
from src.preprocessing import (
    get_cache_paths,
    raw_cache_is_valid,
    component_cache_is_valid,
    load_im_info,
    ensure_im_inputs,
    preprocess_file,
)
from src.preprocessing.identification import (
    get_id_cache_paths,
    id_cache_is_valid,
    find_matching_idxml,
    load_search_params,
)
from src.preprocessing.component_inputs import create_spectra_table_with_ids
from src.preprocessing.features import find_matching_featurexml
from src.components.factory import (
    create_ms_components,
    create_id_components,
    reconstruct_ms_components,
    reconstruct_id_components,
)
from src.components.viewer_panels import (
    CATALOG,
    DEFAULT_LAYOUT,
    ViewerContext,
    available_catalog,
    editor_catalog,
    layout_path,
    renderers,
)
from src.layout_manager import MAX_EXPERIMENTS, layout_button, render_experiments, resolve_layout


# Page setup
params = page_setup()

st.title("mzML Viewer")

# Show return link if loaded from shared volume (cross-app integration)
if st.session_state.get("workspace_source") == "shared":
    source_info = st.session_state.get("source_info", {})
    source_app = source_info.get("app_name", "Source App")
    source_url = source_info.get("return_url")

    if source_url:
        st.markdown(f"[← Return to {source_app}]({source_url})")

# Get workspace mzML directory
mzML_dir = Path(st.session_state.workspace, "mzML-files")

# Auto-process files if coming from shared workspace (cross-app integration)
if st.session_state.get("workspace_source") == "shared":
    unprocessed_files = []
    for mzml_file in sorted(mzML_dir.glob("*.mzML")):
        paths = get_cache_paths(st.session_state.workspace, mzml_file)
        if not (raw_cache_is_valid(mzml_file, paths) and component_cache_is_valid(paths)):
            unprocessed_files.append(mzml_file)

    if unprocessed_files:
        source_app = st.session_state.get("source_info", {}).get("app_name", "external app")
        st.info(f"Processing {len(unprocessed_files)} file(s) from {source_app}...")
        for mzml_file in unprocessed_files:
            preprocess_file(mzml_file)
        # Rerun to ensure clean component state after processing
        st.rerun()

# Files processed by an earlier version are processed again (once per session,
# so a file that fails to process does not rerun the page forever)
reprocessed = st.session_state.setdefault("viewer_reprocessed", set())
outdated_files = []
for mzml_file in sorted(mzML_dir.glob("*.mzML")):
    paths = get_cache_paths(st.session_state.workspace, mzml_file)
    if (
        str(mzml_file) not in reprocessed
        and paths["metadata"].exists()
        and not (raw_cache_is_valid(mzml_file, paths) and component_cache_is_valid(paths))
    ):
        outdated_files.append(mzml_file)
if outdated_files:
    for mzml_file in outdated_files:
        reprocessed.add(str(mzml_file))
        preprocess_file(mzml_file)
    clear_component_cache()
    st.session_state.pop("viewer_current_file", None)
    st.rerun()

# Find preprocessed files
preprocessed_files = []
for mzml_file in sorted(mzML_dir.glob("*.mzML")):
    paths = get_cache_paths(st.session_state.workspace, mzml_file)
    if raw_cache_is_valid(mzml_file, paths) and component_cache_is_valid(paths):
        preprocessed_files.append(mzml_file)

# Handle no files case
if not preprocessed_files:
    st.warning("No preprocessed files available.")
    st.info("Please upload and process files on the Upload page first.")
    if st.button("Go to Upload Page", type="primary"):
        st.switch_page("content/upload.py")
    st.stop()

# =============================================================================
# Experiments: one grid per file, each below its own file picker
# =============================================================================

SELECTOR_KEY = "viewer_file"


def state_session_key(index: int) -> str:
    """Session key of experiment ``index``'s StateManager (the first keeps the old name)."""
    return "viewer_state" if index == 0 else f"viewer_state_{index}"


def load_context(selected_file: Path, index: int) -> ViewerContext:
    """Load the components of ``selected_file`` for experiment ``index``."""
    # A file switch clears this experiment's selections
    current_file_key = "viewer_current_file" if index == 0 else f"viewer_current_file_{index}"
    file_changed = st.session_state.get(current_file_key) != selected_file.name
    if file_changed:
        StateManager(session_key=state_session_key(index)).clear()
        clear_component_cache()
        st.session_state.pop(f"viewer_im_components_{selected_file.stem}", None)
        st.session_state[current_file_key] = selected_file.name

    paths = get_cache_paths(st.session_state.workspace, selected_file)
    im_info = load_im_info(paths)
    num_im_dimensions = im_info.get("num_dimensions", 0)
    file_id = selected_file.stem
    cache_path = str(paths["component_cache"])

    # Identification data of a matching idXML
    idxml_path = find_matching_idxml(selected_file, st.session_state.workspace)
    has_ids = False
    id_paths = None
    search_params = None
    if idxml_path:
        id_paths = get_id_cache_paths(st.session_state.workspace, idxml_path)
        if id_cache_is_valid(idxml_path, id_paths):
            has_ids = True
            search_params = load_search_params(id_paths)
            if not paths["spectra_table_with_ids"].exists():
                id_df = pl.read_parquet(id_paths["identifications"])
                create_spectra_table_with_ids(paths, id_df, im_info)

    # Components are kept per file, so experiments showing the same file share them
    component_cache_key = f"viewer_components_{file_id}"
    if component_cache_key not in st.session_state or file_changed:
        with st.spinner("Loading visualization components..."):
            try:
                ms_components = reconstruct_ms_components(cache_path, file_id, num_im_dimensions)
            except Exception:
                ms_components = create_ms_components(paths, im_info, file_id)
            id_components = None
            if has_ids:
                try:
                    id_components = reconstruct_id_components(cache_path, file_id)
                except Exception:
                    id_components = create_id_components(
                        paths, id_paths, im_info, file_id, search_params
                    )
        st.session_state[component_cache_key] = {
            "ms": ms_components,
            "id": id_components,
            "id_paths": id_paths,
        }
    cached = st.session_state[component_cache_key]

    return ViewerContext(
        workspace=Path(st.session_state.workspace),
        selected_file=selected_file,
        paths=paths,
        im_info=im_info,
        file_id=file_id,
        cache_path=cache_path,
        state_manager=StateManager(session_key=state_session_key(index)),
        ms_components=cached["ms"],
        id_components=cached["id"] if has_ids else None,
        id_paths=cached["id_paths"],
        search_params=search_params,
        featurexml_path=find_matching_featurexml(selected_file, st.session_state.workspace),
        im_inputs_mtime=ensure_im_inputs(paths, im_info),
        local=st.session_state.location == "local",
        index=index,
    )


files_by_name = {p.name: p for p in preprocessed_files}

# A tool run that wrote a new mzML asks for it to be opened in the first experiment
pending_file = st.session_state.pop("viewer_pending_file", None)
if pending_file in files_by_name:
    st.session_state[f"{SELECTOR_KEY}_experiment_0"] = pending_file

contexts = {}


def experiment_renderers(name: str, index: int):
    ctx = load_context(files_by_name[name], index)
    contexts[index] = ctx
    catalog = available_catalog(ctx)
    # Leave out, silently, what this file cannot show (no idXML, no ion mobility,
    # no featureXML)
    known = {c["name"] for c in catalog}
    return {n: (r if n in known else (lambda: None)) for n, r in renderers(ctx).items()}


def layout_buttons(index: int) -> None:
    if index == 0:
        layout_button(
            layout_path(st.session_state.workspace), editor_catalog(), DEFAULT_LAYOUT,
            key="viewer_layout_dialog", max_experiments=MAX_EXPERIMENTS,
            experiment_label="File", target="the Viewer",
            export_filename="toppview-lite-layout.json",
            help_text=(
                "Drag components into the grid (up to 3 per row). Add files to compare "
                "several side by side or one below the other, then press **Save**."
            ),
        )


# The layout keeps every component; a file that cannot show one leaves it out
layout, side_by_side = resolve_layout(
    layout_path(st.session_state.workspace), CATALOG, DEFAULT_LAYOUT
)
shown = render_experiments(
    layout,
    side_by_side,
    {name: name for name in files_by_name},
    experiment_renderers,
    key=SELECTOR_KEY,
    labels={c["name"]: c["label"] for c in CATALOG},
    experiment_label="mzML file",
    buttons=layout_buttons,
)

# Drop the components of files no experiment shows any more
shown_ids = {Path(name).stem for name in shown if name}
for prefix in ("viewer_components_", "viewer_im_components_"):
    for key in [k for k in st.session_state if k.startswith(prefix)]:
        if key[len(prefix):] not in shown_ids:
            del st.session_state[key]

# =============================================================================
# Sidebar Info (the first experiment)
# =============================================================================

ctx = contexts.get(0)
if ctx is not None:
    num_spectra = pq.read_metadata(ctx.paths["metadata"]).num_rows
    num_peaks = pq.read_metadata(ctx.paths["peaks"]).num_rows
    with st.sidebar:
        st.header("File Info")
        st.write(f"**File**: {ctx.selected_file.name}")
        st.write(f"**Spectra**: {num_spectra:,}")
        st.write(f"**Peaks**: {num_peaks:,}")

        if ctx.has_ids and ctx.id_paths:
            id_df = pl.read_parquet(ctx.id_paths["identifications"])
            num_matched = id_df.filter(pl.col("scan_id") > 0).height
            st.write(f"**Identifications**: {id_df.height:,} ({num_matched} linked)")

        if ctx.has_im:
            st.write(f"**Ion Mobility**: {ctx.im_info['type'].upper()}")
            st.write(f"**IM Dimensions**: {ctx.im_info.get('num_dimensions', 0)}")

        st.divider()
        st.header("Selection State")
        st.json(ctx.state_manager.get_state_for_vue())
