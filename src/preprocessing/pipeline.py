"""Shared preprocessing pipeline functions."""

from pathlib import Path

import streamlit as st
import polars as pl

from src.preprocessing import (
    get_cache_paths,
    raw_cache_is_valid,
    extract_mzml_to_parquet,
    component_cache_is_valid,
    create_component_inputs,
    create_spectra_table_with_ids,
    load_im_info,
)
from src.preprocessing.identification import (
    get_id_cache_paths,
    id_cache_is_valid,
    extract_idxml_to_parquet,
    link_identifications_to_spectra,
    find_matching_idxml,
    load_search_params,
)
from src.components.factory import create_ms_components, create_id_components


def preprocess_file(mzml_path: Path, workspace: Path = None) -> bool:
    """Run preprocessing pipeline for a file with progress indicator.

    Args:
        mzml_path: Path to the mzML file
        workspace: Workspace path (defaults to st.session_state.workspace)

    Returns:
        True if processing succeeded, False otherwise
    """
    if workspace is None:
        workspace = st.session_state.workspace

    paths = get_cache_paths(workspace, mzml_path)
    try:
        with st.status(f"Processing {mzml_path.name}...", expanded=True) as status:
            if not raw_cache_is_valid(mzml_path, paths):
                st.write("Extracting mzML data...")
                extract_mzml_to_parquet(mzml_path, paths, lambda msg: st.write(msg))
            if not component_cache_is_valid(paths):
                st.write("Creating component inputs...")
                create_component_inputs(paths, lambda msg: st.write(msg))

            im_info = load_im_info(paths)
            has_ids = False
            id_paths = None
            search_params = None

            # Check for matching idXML and process if found
            idxml_path = find_matching_idxml(mzml_path, workspace)
            if idxml_path:
                st.write(f"Found matching idXML: {idxml_path.name}")
                id_paths = get_id_cache_paths(workspace, idxml_path)
                if not id_cache_is_valid(idxml_path, id_paths):
                    st.write("Extracting identification data...")
                    extract_idxml_to_parquet(idxml_path, id_paths, lambda msg: st.write(msg))

                    # Link identifications to spectra
                    st.write("Linking identifications to spectra...")
                    id_df = pl.read_parquet(id_paths["identifications"])
                    metadata_df = pl.read_parquet(paths["metadata"])
                    linked_df = link_identifications_to_spectra(
                        id_df, metadata_df, status_callback=lambda msg: st.write(msg)
                    )
                    linked_df.write_parquet(id_paths["identifications"])

                # Create spectra_table_with_ids for ID count column
                st.write("Creating spectra table with ID counts...")
                id_df = pl.read_parquet(id_paths["identifications"])
                create_spectra_table_with_ids(paths, id_df, im_info, lambda msg: st.write(msg))

                has_ids = True
                search_params = load_search_params(id_paths)

            st.write("Pre-computing visualization caches...")
            # Create MS components
            create_ms_components(paths, im_info, file_id=mzml_path.stem)

            # Create ID components if identifications exist
            if has_ids and id_paths:
                st.write("Creating identification components...")
                create_id_components(paths, id_paths, im_info, file_id=mzml_path.stem, search_params=search_params)

            status.update(label=f"Processed {mzml_path.name}", state="complete")
        return True
    except Exception as e:
        st.error(f"Error processing {mzml_path.name}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False
