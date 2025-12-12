"""Interactive mzML Viewer Page.

This page displays preprocessed mzML files using streamlit_vue_components
with cross-component selection linking.

Layout:
- Ion Mobility table (if FAIMS data present)
- Heatmap (peak map)
- Spectra table (with # IDs column if identifications present) | Peaks table (side by side)
- Spectrum plot
- Identifications table (if identifications present)
- SequenceView (if identification selected)
"""

import hashlib
from pathlib import Path

import streamlit as st
import polars as pl
import pyarrow.parquet as pq
from streamlit_vue_components import StateManager, Table, SequenceView

from src.common.common import page_setup
from src.preprocessing import (
    get_cache_paths,
    raw_cache_is_valid,
    component_cache_is_valid,
    load_im_info,
)
from src.preprocessing.identification import (
    get_id_cache_paths,
    id_cache_is_valid,
    find_matching_idxml,
    get_sequence_data_for_identification,
    load_search_params,
    precompute_spectrum_annotations,
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

    # Clear all render caches so new file data is sent to Vue
    from streamlit_vue_components.rendering.bridge import _cached_prepare_vue_data, _VUE_ECHOED_HASH_KEY
    from streamlit_vue_components.preprocessing.filtering import _cached_filter_and_collect
    _cached_prepare_vue_data.clear()
    _cached_filter_and_collect.clear()
    # Also clear the Vue hash tracking so Vue receives fresh data
    if _VUE_ECHOED_HASH_KEY in st.session_state:
        st.session_state[_VUE_ECHOED_HASH_KEY].clear()

    # Load components for selected file
    paths = get_cache_paths(st.session_state.workspace, selected_file)
    im_info = load_im_info(paths)

    # Check for identification data early to pass to create_components
    idxml_path = find_matching_idxml(selected_file, st.session_state.workspace)
    has_ids_early = False
    if idxml_path:
        id_paths_early = get_id_cache_paths(st.session_state.workspace, idxml_path)
        if id_cache_is_valid(idxml_path, id_paths_early):
            has_ids_early = True

    with st.spinner("Loading visualization components..."):
        im_table, spectra_table, peaks_table, spectrum_plot, heatmap = create_components(
            paths, im_info, file_id=selected_file.stem, has_ids=has_ids_early
        )

    # Get file stats
    num_spectra = pq.read_metadata(paths["metadata"]).num_rows
    num_peaks = pq.read_metadata(paths["peaks"]).num_rows

    # Check for identification data (reuse early check)
    id_df = None
    id_table = None
    fragment_masses_df = None
    peak_annotations_df = None
    search_params = None
    id_paths = None
    has_ids = False

    if idxml_path and has_ids_early:
        id_paths = get_id_cache_paths(st.session_state.workspace, idxml_path)
        if id_cache_is_valid(idxml_path, id_paths):
            id_df = pl.read_parquet(id_paths["identifications"])
            fragment_masses_df = pl.read_parquet(id_paths["fragment_masses"])
            # Load search params and peak annotations if available
            search_params = load_search_params(id_paths)
            if id_paths["peak_annotations"].exists():
                peak_annotations_df = pl.read_parquet(id_paths["peak_annotations"])
            has_ids = id_df.height > 0

            if has_ids:
                # Add identification count to spectra table
                # Count identifications per scan_id
                id_counts = (
                    id_df.filter(pl.col("scan_id") > 0)
                    .group_by("scan_id")
                    .agg(pl.len().alias("num_ids"))
                )

                # Read spectra table and join with id counts
                spectra_df = pl.read_parquet(paths["spectra_table"])
                spectra_df = spectra_df.join(id_counts, on="scan_id", how="left")
                spectra_df = spectra_df.with_columns(
                    pl.col("num_ids").fill_null(0).cast(pl.Int32)
                )

                # Recreate spectra_table with the new column
                spectra_filters = {"im_dimension": "im_id"} if im_info.get("type") != "none" else None
                spectra_table = Table(
                    cache_id=f"{selected_file.stem}_spectra_table_with_ids",
                    data=spectra_df.lazy(),
                    cache_path=str(paths["component_cache"]),
                    filters=spectra_filters,
                    interactivity={"spectrum": "scan_id"},
                    column_definitions=[
                        {"field": "scan_id", "title": "Scan ID", "sorter": "number", "width": 80},
                        {"field": "name", "title": "Name"},
                        {
                            "field": "retention_time",
                            "title": "RT (min)",
                            "sorter": "number",
                            "formatter": "money",
                            "formatterParams": {"precision": 2, "symbol": ""},
                        },
                        {"field": "ms_level", "title": "MS", "sorter": "number", "width": 50},
                        {
                            "field": "precursor_mz",
                            "title": "Precursor m/z",
                            "sorter": "number",
                            "formatter": "money",
                            "formatterParams": {"precision": 2, "symbol": ""},
                        },
                        {"field": "charge", "title": "z", "sorter": "number", "width": 50},
                        {"field": "num_peaks", "title": "# Peaks", "sorter": "number", "width": 80},
                        {"field": "num_ids", "title": "# IDs", "sorter": "number", "width": 60},
                    ],
                    title="Spectra",
                    index_field="scan_id",
                    go_to_fields=["scan_id", "name"],
                    default_row=0,
                )

                # Create identifications table component
                id_table = Table(
                    cache_id=f"{selected_file.stem}_ids_table",
                    data=id_df.lazy(),
                    cache_path=str(paths["component_cache"]),
                    filters={"spectrum": "scan_id"},
                    interactivity={"identification": "id_idx"},
                    column_definitions=[
                        {"field": "sequence_display", "title": "Sequence", "headerTooltip": True},
                        {"field": "score", "title": "Score", "sorter": "number", "hozAlign": "right",
                         "formatter": "money", "formatterParams": {"precision": 2}},
                        {"field": "charge", "title": "z", "sorter": "number", "hozAlign": "right"},
                        {"field": "protein_accession", "title": "Protein", "headerTooltip": True},
                    ],
                    index_field="id_idx",
                    title="Identifications",
                )

                # Precompute spectrum annotations (once, cached to parquet)
                # These are used for dynamic annotations on the single spectrum_plot
                annotated_path = paths["annotated_spectrum_plot"]
                if not annotated_path.exists():
                    peaks_df = pl.read_parquet(paths["peaks"])
                    precompute_spectrum_annotations(
                        peaks_df,
                        id_paths,
                        annotated_path,
                        status_callback=lambda msg: st.toast(msg)
                    )

    st.session_state[file_cache_key] = {
        "im_table": im_table,
        "spectra_table": spectra_table,
        "peaks_table": peaks_table,
        "spectrum_plot": spectrum_plot,
        "heatmap": heatmap,
        "im_info": im_info,
        "num_spectra": num_spectra,
        "num_peaks": num_peaks,
        "has_ids": has_ids,
        "id_table": id_table,
        "id_df": id_df,
        "fragment_masses_df": fragment_masses_df,
        "peak_annotations_df": peak_annotations_df,
        "search_params": search_params,
        "paths": paths,
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
has_ids = cached["has_ids"]
id_table = cached["id_table"]
id_df = cached["id_df"]
fragment_masses_df = cached["fragment_masses_df"]
peak_annotations_df = cached["peak_annotations_df"]
search_params = cached["search_params"]
paths = cached["paths"]

# =============================================================================
# Sidebar Info
# =============================================================================

with st.sidebar:
    st.header("File Info")
    st.write(f"**File**: {selected_file.name}")
    st.write(f"**Spectra**: {num_spectra:,}")
    st.write(f"**Peaks**: {num_peaks:,}")

    if has_ids and id_df is not None:
        num_ids = id_df.height
        num_matched = id_df.filter(pl.col("scan_id") > 0).height
        st.write(f"**Identifications**: {num_ids:,} ({num_matched} linked)")

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

# Use file-specific keys to force component refresh on file change
file_key = selected_file.stem

# Ion Mobility table (if present)
if im_table is not None:
    im_table(key=f"{file_key}_im_table", state_manager=state_manager, height=400)

# Heatmap (full width)
heatmap(key=f"{file_key}_heatmap", state_manager=state_manager, height=400)

# Tables side by side: Spectra | Peaks
col1, col2 = st.columns([1, 1])

with col1:
    spectra_table(key=f"{file_key}_spectra_table", state_manager=state_manager, height=400)

with col2:
    peaks_table(key=f"{file_key}_peaks_table", state_manager=state_manager, height=400)

# =============================================================================
# Spectrum Plot
# =============================================================================

# Render spectrum plot - filtering handles annotations automatically
# When identification is selected: filters by scan_id AND id_idx (shows annotated peaks)
# When no identification selected: filter_defaults maps identification to -1 (shows base peaks)
spectrum_plot(key=f"{file_key}_spectrum_plot", state_manager=state_manager, height=400)

# Identification table (below spectrum plot, only shown when IDs present)
if has_ids:
    id_table(key=f"{file_key}_id_table", state_manager=state_manager, height=300)

# =============================================================================
# SequenceView (if identification selected)
# =============================================================================

# Get current identification selection
current_state = state_manager.get_state_for_vue()
selected_id_idx = current_state.get("identification")

# Only show SequenceView if we have identifications and one is selected
if has_ids and id_df is not None and selected_id_idx is not None:
    # Look up the selected identification
    selected_rows = id_df.filter(pl.col("id_idx") == selected_id_idx)

    if selected_rows.height > 0:
        id_row = selected_rows.row(0, named=True)

        # Get sequence data for SequenceView
        peaks_df = pl.read_parquet(paths["peaks"])
        sequence_data, _, _ = get_sequence_data_for_identification(
            id_row, peaks_df, fragment_masses_df, peak_annotations_df, search_params
        )

        st.markdown("---")
        st.subheader(f"Peptide: {id_row['sequence_display']}")

        # Get observed masses and peak_ids for SequenceView
        scan_id = id_row.get("scan_id", -1)
        if scan_id > 0:
            spectrum_peaks = peaks_df.filter(pl.col("scan_id") == scan_id)
            if spectrum_peaks.height > 0:
                observed_masses = spectrum_peaks["mass"].to_list()
                # Get peak_ids for interactivity linking (same order as observed_masses)
                peak_ids = spectrum_peaks["peak_id"].to_list()
            else:
                observed_masses = []
                peak_ids = []
        else:
            observed_masses = []
            peak_ids = []

        precursor_mass = id_row.get("precursor_mz", 0.0)

        # Render SequenceView directly without per-ID caching
        # Fragment masses are already cached in fragment_masses.parquet
        # Use deconvolved=False since TOPPView-Lite works with raw m/z data
        sequence_view = SequenceView(
            cache_id=f"{selected_file.stem}_sequence_view",  # Single cache for all IDs
            sequence=id_row["sequence"],
            observed_masses=observed_masses,
            peak_ids=peak_ids,  # Pass peak_ids for interactivity linking
            precursor_mass=precursor_mass,
            cache_path=str(paths["component_cache"]),
            deconvolved=False,  # m/z data, not deconvolved neutral masses
            precursor_charge=id_row.get("charge", 2),  # Use precursor charge for max fragment charge
            interactivity={"peak": "peak_id"},  # Enable cross-component selection
            # Pass pre-computed sequence data to skip redundant calculation
            _precomputed_sequence_data=sequence_data,
        )
        # Use unique key per sequence to ensure Streamlit treats each selection as different
        seq_hash = hashlib.md5(id_row["sequence"].encode()).hexdigest()[:8]
        sequence_view(key=f"sequence_view_{seq_hash}", state_manager=state_manager, height=600)
