"""Interactive mzML Viewer Page.

This page displays preprocessed mzML files using openms_insight
with cross-component selection linking.

Layout:
- Ion Mobility table (if FAIMS data present)
- TIC / BPC chromatogram (click selects the MS1 spectrum)
- Heatmap (peak map)
- Spectra table (with # IDs column if identifications present) | Peaks table (side by side)
- Spectrum plot (annotated, or mirrored against the theoretical b/y ions, if identification selected)
- Identifications table (if identifications present)
- SequenceView (if identification selected)
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
    ensure_chromatogram_input,
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
from src.export import export_table_selection, filtered_output_path, peak_window
from src.preprocessing.features import (
    extract_featurexml_to_parquet,
    feature_cache_is_valid,
    find_matching_featurexml,
    get_feature_cache_paths,
)
from src.components.factory import (
    create_ms_components,
    create_feature_table,
    create_chromatogram_components,
    create_id_components,
    reconstruct_ms_components,
    reconstruct_id_components,
    create_mirror_plot,
    MIRROR_BOTTOM_IDENTIFIER,
)
from src.preprocessing.mirror import ensure_mirror_input


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
# Main Layout
# =============================================================================

# Create shared state manager
state_manager = StateManager(session_key="viewer_state")
file_key = selected_file.stem

# Ion Mobility table (if present)
if ms_components["im_table"] is not None:
    ms_components["im_table"](key=f"{file_key}_im_table", state_manager=state_manager, height=400)

# TIC / BPC (full width). Clicking a point selects that MS1 spectrum.
ensure_chromatogram_input(paths, im_info)
chromatograms = create_chromatogram_components(paths, im_info, file_id)
chromatogram_kind = st.segmented_control(
    "Chromatogram",
    options=["tic", "bpc"],
    format_func={"tic": "TIC", "bpc": "BPC"}.get,
    default="tic",
    key="viewer_chromatogram_kind",
    label_visibility="collapsed",
) or "tic"
chromatograms[chromatogram_kind](
    key=f"{file_key}_{chromatogram_kind}_plot", state_manager=state_manager, height=250
)

# Heatmap (full width)
ms_components["heatmap"](key=f"{file_key}_heatmap", state_manager=state_manager, height=400)

# Tables side by side: Spectra | Peaks
col1, col2 = st.columns([1, 1])

with col1:
    # Use spectra_table_with_ids if IDs are present, else basic spectra_table
    if has_ids and id_components:
        id_components["spectra_table_with_ids"](
            key=f"{file_key}_spectra_table",
            state_manager=state_manager,
            height=400
        )
    else:
        ms_components["spectra_table"](
            key=f"{file_key}_spectra_table",
            state_manager=state_manager,
            height=400
        )

with col2:
    ms_components["peaks_table"](key=f"{file_key}_peaks_table", state_manager=state_manager, height=400)

# =============================================================================
# Spectrum Plot & Identifications
# =============================================================================

if has_ids and id_components:
    spectrum_mode = st.segmented_control(
        "Spectrum view",
        ["Annotated", "Mirror vs. theoretical"],
        default="Annotated",
        key="viewer_spectrum_mode",
        label_visibility="collapsed",
    )

    if spectrum_mode == "Mirror vs. theoretical":
        mirror_key = f"viewer_mirror_{file_key}"
        mirror_paths, rebuilt = ensure_mirror_input(paths, cached["id_paths"], search_params)
        if rebuilt or mirror_key not in st.session_state:
            with st.spinner("Generating theoretical spectra..."):
                st.session_state[mirror_key] = create_mirror_plot(
                    mirror_paths["mirror"], cache_path, file_id, regenerate_cache=rebuilt
                )

        # The bottom half follows the selected identification.
        selected_id = state_manager.get_selection("identification")
        if state_manager.get_selection(MIRROR_BOTTOM_IDENTIFIER) != selected_id:
            if selected_id is None:
                state_manager.clear_selection(MIRROR_BOTTOM_IDENTIFIER)
            else:
                state_manager.set_selection(MIRROR_BOTTOM_IDENTIFIER, selected_id)

        if selected_id is None:
            st.caption("Select an identification below to compare its spectrum with the theoretical b/y ions.")
        else:
            match = (
                pl.scan_parquet(mirror_paths["summary"])
                .filter(pl.col("id_idx") == selected_id)
                .collect()
            )
            if match.height:
                row = match.row(0, named=True)
                st.caption(
                    f"{row['matched_ions']} of {row['total_ions']} theoretical b/y ions matched "
                    f"({row['coverage']:.0f} %). Matched peaks are red and labelled."
                )
            else:
                st.caption("This identification is not linked to a spectrum.")

        st.session_state[mirror_key](
            key=f"{file_key}_mirror_plot",
            state_manager=state_manager,
            height=450,
        )
    else:
        # Annotated spectrum plot (linked to SequenceView)
        id_components["annotated_spectrum_plot"](
            key=f"{file_key}_spectrum_plot",
            state_manager=state_manager,
            height=400,
            sequence_view_key=f"{file_key}_sequence_view"
        )

    # Render SequenceView first (computes annotations)
    sv_result = id_components["sequence_view"](
        key=f"{file_key}_sequence_view",
        state_manager=state_manager,
        height=800
    )

    # Identification table
    id_components["id_table"](key=f"{file_key}_id_table", state_manager=state_manager, height=300)
else:
    # Basic spectrum plot (no annotations)
    ms_components["spectrum_plot"](key=f"{file_key}_spectrum_plot", state_manager=state_manager, height=400)

# =============================================================================
# Features (featureXML with the same name as the mzML)
# =============================================================================

featurexml_path = find_matching_featurexml(selected_file, st.session_state.workspace)
if featurexml_path is not None:
    feature_paths = get_feature_cache_paths(st.session_state.workspace, featurexml_path)
    if not feature_cache_is_valid(featurexml_path, feature_paths):
        with st.spinner(f"Reading {featurexml_path.name}..."):
            extract_featurexml_to_parquet(featurexml_path, feature_paths, paths["metadata"])
    feature_table = create_feature_table(
        str(feature_paths["features"]), cache_path, file_id
    )
    st.caption(
        f"Features from {featurexml_path.name}. Click a feature to show the MS1 "
        "spectrum at its apex."
    )
    feature_table(key=f"{file_key}_feature_table", state_manager=state_manager, height=350)

    # A newly clicked feature shows the MS1 spectrum at its apex
    selected_feature = state_manager.get_selection("feature")
    applied_key = f"viewer_applied_feature_{file_key}"
    if selected_feature is not None and st.session_state.get(applied_key) != selected_feature:
        st.session_state[applied_key] = selected_feature
        apex = (
            pl.scan_parquet(feature_paths["features"])
            .filter(pl.col("feature_id") == selected_feature)
            .select("apex_scan_id")
            .collect()
        )
        if apex.height and apex["apex_scan_id"][0] > 0:
            state_manager.set_selection("spectrum", int(apex["apex_scan_id"][0]))
            state_manager.clear_selection("peak")
            state_manager.clear_selection("identification")
            st.rerun()

# =============================================================================
# Export filtered mzML
# =============================================================================


def _pagination_state(table, state):
    """A table's own sort and column filter state from the shared selection state."""
    identifier = getattr(table, "_pagination_identifier", None) or f"{table._cache_id}_page"
    return state.get(identifier) or {}


def _describe_window(window):
    parts = []
    for key, label in (("mz", "m/z"), ("int", "intensity")):
        bounds = window[key]
        if bounds is None:
            continue
        lo, hi = bounds
        if lo is not None and hi is not None:
            parts.append(f"{label} {lo:g} to {hi:g}")
        elif lo is not None:
            parts.append(f"{label} ≥ {lo:g}")
        else:
            parts.append(f"{label} ≤ {hi:g}")
    return ", ".join(parts)


@st.fragment
def export_section():
    """Write a copy of the mzML holding what the Spectra and Peak List tables show."""
    with st.expander("Export filtered mzML"):
        spectra_table = (
            id_components["spectra_table_with_ids"]
            if has_ids and id_components
            else ms_components["spectra_table"]
        )
        peaks_table = ms_components["peaks_table"]
        state = state_manager.get_state_for_vue()

        metadata = pl.read_parquet(paths["metadata"], columns=["scan_id", "retention_time", "ms_level"])
        scan_ids = spectra_table.export_data(state)["scan_id"].to_list()
        peak_filters = _pagination_state(peaks_table, state).get("column_filters", [])
        window = peak_window(peak_filters)

        summary = f"{len(scan_ids):,} of {metadata.height:,} spectra"
        if len(scan_ids) < metadata.height:
            summary += " (the rows of the Spectra table)"
        described = _describe_window(window)
        summary += f", peaks with {described} (the Peak List filters)." if described else ", all peaks."
        st.markdown(f"**Exports** {summary}")
        st.caption(
            "Filter the Spectra table (RT, MS level, precursor m/z, ...) and the m/z and "
            "intensity columns of the Peak List to change what is exported."
        )

        out_path = filtered_output_path(st.session_state.workspace, selected_file)
        result_key = f"export_result_{file_key}"
        if st.button("Create filtered mzML", disabled=not scan_ids):
            with st.spinner("Filtering..."):
                try:
                    backend = export_table_selection(
                        selected_file, out_path, metadata, scan_ids, peak_filters
                    )
                    st.session_state[result_key] = {"ok": True, "backend": backend}
                except (ValueError, RuntimeError) as err:
                    st.session_state[result_key] = {"ok": False, "error": str(err)}

        result = st.session_state.get(result_key)
        if result and not result["ok"]:
            st.error(result["error"])
        elif result and out_path.exists():
            size_mb = out_path.stat().st_size / 1e6
            st.caption(f"{out_path.name}: {size_mb:.1f} MB, written with {result['backend']}.")
            with open(out_path, "rb") as fh:
                st.download_button(
                    "Download filtered mzML", fh, file_name=out_path.name, mime="application/xml"
                )


export_section()
