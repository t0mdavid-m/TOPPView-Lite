"""Interactive mzML Viewer Page.

This page displays preprocessed mzML files using streamlit_vue_components
with cross-component selection linking.

Layout:
- Ion Mobility table (if FAIMS data present)
- Heatmap (peak map)
- Spectra table | Peaks table (side by side)
- Spectrum plot
"""

from pathlib import Path

import streamlit as st
import pyarrow.parquet as pq
from streamlit_vue_components import StateManager

from src.common.common import page_setup
from src.preprocessing import (
    get_cache_paths,
    raw_cache_is_valid,
    component_cache_is_valid,
    load_im_info,
)
from src.components import create_components


# Page setup
params = page_setup()

st.title("mzML Viewer")

# Get workspace mzML directory
mzML_dir = Path(st.session_state.workspace, "mzML-files")

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

selected_file = st.selectbox(
    "Select mzML file",
    preprocessed_files,
    format_func=lambda p: p.name,
    key="viewer_selected_file",
)

# =============================================================================
# Load Components
# =============================================================================

# Use session state key based on file to cache components per file
file_cache_key = f"viewer_components_{selected_file.name}"

if file_cache_key not in st.session_state:
    # Clear components from other files
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("viewer_components_")]
    for k in keys_to_remove:
        del st.session_state[k]

    # Clear selection state when switching files
    state_manager = StateManager(session_key="viewer_state")
    state_manager.clear()

    # Load components for selected file
    paths = get_cache_paths(st.session_state.workspace, selected_file)
    im_info = load_im_info(paths)

    with st.spinner("Loading visualization components..."):
        im_table, spectra_table, peaks_table, spectrum_plot, heatmap = create_components(
            paths, im_info, file_id=selected_file.stem
        )

    # Get file stats
    num_spectra = pq.read_metadata(paths["metadata"]).num_rows
    num_peaks = pq.read_metadata(paths["peaks"]).num_rows

    st.session_state[file_cache_key] = {
        "im_table": im_table,
        "spectra_table": spectra_table,
        "peaks_table": peaks_table,
        "spectrum_plot": spectrum_plot,
        "heatmap": heatmap,
        "im_info": im_info,
        "num_spectra": num_spectra,
        "num_peaks": num_peaks,
    }

# Get cached components
cached = st.session_state[file_cache_key]
im_table = cached["im_table"]
spectra_table = cached["spectra_table"]
peaks_table = cached["peaks_table"]
spectrum_plot = cached["spectrum_plot"]
heatmap = cached["heatmap"]
im_info = cached["im_info"]
num_spectra = cached["num_spectra"]
num_peaks = cached["num_peaks"]

# =============================================================================
# Sidebar Info
# =============================================================================

with st.sidebar:
    st.header("File Info")
    st.write(f"**File**: {selected_file.name}")
    st.write(f"**Spectra**: {num_spectra:,}")
    st.write(f"**Peaks**: {num_peaks:,}")

    if im_info.get("type") != "none":
        st.write(f"**Ion Mobility**: {im_info['type'].upper()}")
        st.write(f"**IM Dimensions**: {im_info.get('num_dimensions', 0)}")

    st.divider()

    st.header("Selection State")
    state_manager = StateManager(session_key="viewer_state")
    st.json(state_manager.get_state_for_vue())

# =============================================================================
# Main Layout
# =============================================================================

# Create shared state manager
state_manager = StateManager(session_key="viewer_state")

# Ion Mobility table (if present)
if im_table is not None:
    im_table(key="im_table", state_manager=state_manager, height=180)

# Heatmap (full width)
heatmap(key="heatmap", state_manager=state_manager, height=350)

# Tables side by side
col1, col2 = st.columns([1, 1])

with col1:
    spectra_table(key="spectra_table", state_manager=state_manager, height=300)

with col2:
    peaks_table(key="peaks_table", state_manager=state_manager, height=300)

# Spectrum plot (full width)
spectrum_plot(key="spectrum_plot", state_manager=state_manager, height=300)
