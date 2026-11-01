"""Interactive mzML Viewer Page.

This page displays preprocessed mzML files using openms_insight
with cross-component selection linking.

The components (src/components/viewer_panels.py) are arranged on the Layout
page; without a saved layout the viewer uses DEFAULT_LAYOUT there.
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
    DEFAULT_LAYOUT,
    ViewerContext,
    available_catalog,
    layout_path,
    renderers,
)
from src.layout_manager import render_layout, resolve_layout


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
# File Selection
# =============================================================================

# A tool run that wrote a new mzML asks for it to be opened
pending_file = st.session_state.pop("viewer_pending_file", None)
if pending_file is not None:
    match = [p for p in preprocessed_files if p.name == pending_file]
    if match:
        st.session_state["viewer_selected_file"] = match[0]

selected_file = st.selectbox(
    "Select mzML file",
    preprocessed_files,
    format_func=lambda p: p.name,
    key="viewer_selected_file",
)

# =============================================================================
# Component Loading
# =============================================================================

# Track current file to detect changes
current_file_key = f"viewer_current_file"
file_changed = st.session_state.get(current_file_key) != selected_file.name

if file_changed:
    # Clear state and caches when switching files
    state_manager = StateManager(session_key="viewer_state")
    state_manager.clear()
    clear_component_cache()
    st.session_state[current_file_key] = selected_file.name

# Get paths and info
paths = get_cache_paths(st.session_state.workspace, selected_file)
im_info = load_im_info(paths)
has_im = im_info.get("type") != "none"
num_im_dimensions = im_info.get("num_dimensions", 0)
file_id = selected_file.stem
cache_path = str(paths["component_cache"])

# Check for identification data
idxml_path = find_matching_idxml(selected_file, st.session_state.workspace)
has_ids = False
id_paths = None
search_params = None

if idxml_path:
    id_paths = get_id_cache_paths(st.session_state.workspace, idxml_path)
    if id_cache_is_valid(idxml_path, id_paths):
        has_ids = True
        search_params = load_search_params(id_paths)

        # Ensure spectra_table_with_ids exists
        if not paths["spectra_table_with_ids"].exists():
            id_df = pl.read_parquet(id_paths["identifications"])
            create_spectra_table_with_ids(paths, id_df, im_info)

# Create or reconstruct components
component_cache_key = f"viewer_components_{file_id}"

if component_cache_key not in st.session_state or file_changed:
    # Clear old component caches
    for key in list(st.session_state.keys()):
        if key.startswith("viewer_components_") and key != component_cache_key:
            del st.session_state[key]

    with st.spinner("Loading visualization components..."):
        # Try to reconstruct from cache first
        try:
            ms_components = reconstruct_ms_components(cache_path, file_id, num_im_dimensions)
        except Exception:
            # Fall back to creating components
            ms_components = create_ms_components(paths, im_info, file_id)

        id_components = None
        if has_ids:
            try:
                id_components = reconstruct_id_components(cache_path, file_id)
            except Exception:
                id_components = create_id_components(
                    paths, id_paths, im_info, file_id, search_params
                )

    # Cache components in session state
    st.session_state[component_cache_key] = {
        "ms": ms_components,
        "id": id_components,
        "has_ids": has_ids,
        "has_im": has_im,
        "id_paths": id_paths,
    }

# Get cached components
cached = st.session_state[component_cache_key]
ms_components = cached["ms"]
id_components = cached["id"]
has_ids = cached["has_ids"]
has_im = cached["has_im"]

# =============================================================================
# Sidebar Info
# =============================================================================

num_spectra = pq.read_metadata(paths["metadata"]).num_rows
num_peaks = pq.read_metadata(paths["peaks"]).num_rows

with st.sidebar:
    st.header("File Info")
    st.write(f"**File**: {selected_file.name}")
    st.write(f"**Spectra**: {num_spectra:,}")
    st.write(f"**Peaks**: {num_peaks:,}")

    if has_ids and cached["id_paths"]:
        id_df = pl.read_parquet(cached["id_paths"]["identifications"])
        num_ids = id_df.height
        num_matched = id_df.filter(pl.col("scan_id") > 0).height
        st.write(f"**Identifications**: {num_ids:,} ({num_matched} linked)")

    if has_im:
        st.write(f"**Ion Mobility**: {im_info['type'].upper()}")
        st.write(f"**IM Dimensions**: {im_info.get('num_dimensions', 0)}")

    st.divider()

    st.header("Selection State")
    state_manager = StateManager(session_key="viewer_state")
    st.json(state_manager.get_state_for_vue())

# =============================================================================
# Components, in the arrangement saved on the Layout page
# =============================================================================

ctx = ViewerContext(
    workspace=Path(st.session_state.workspace),
    selected_file=selected_file,
    paths=paths,
    im_info=im_info,
    file_id=file_id,
    cache_path=cache_path,
    state_manager=StateManager(session_key="viewer_state"),
    ms_components=ms_components,
    id_components=id_components if has_ids else None,
    id_paths=cached["id_paths"],
    search_params=search_params,
    featurexml_path=find_matching_featurexml(selected_file, st.session_state.workspace),
    local=st.session_state.location == "local",
)
catalog = available_catalog(ctx)
layout, _ = resolve_layout(layout_path(st.session_state.workspace), catalog, DEFAULT_LAYOUT)
render_layout(layout[0], renderers(ctx), {c["name"]: c["label"] for c in catalog})
