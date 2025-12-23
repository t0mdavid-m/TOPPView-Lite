from pathlib import Path

import streamlit as st
import pandas as pd

from src.common.common import (
    page_setup,
    save_params,
    v_space,
    show_table,
    TK_AVAILABLE,
    tk_directory_dialog,
)
from src import fileupload
from src.preprocessing import (
    get_cache_paths, raw_cache_is_valid, extract_mzml_to_parquet,
    component_cache_is_valid, create_component_inputs, create_spectra_table_with_ids, load_im_info,
)
from src.preprocessing.identification import (
    get_id_cache_paths, id_cache_is_valid, extract_idxml_to_parquet,
    link_identifications_to_spectra, find_matching_idxml, load_search_params,
)
from src.components.factory import create_ms_components, create_id_components
import polars as pl

params = page_setup()


def process_idxml_file(idxml_path: Path) -> bool:
    """Process an idXML file and link to corresponding mzML.

    Args:
        idxml_path: Path to the idXML file

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Find matching mzML
        mzml_dir = Path(st.session_state.workspace, "mzML-files")
        mzml_path = mzml_dir / f"{idxml_path.stem}.mzML"

        if not mzml_path.exists():
            st.warning(f"No matching mzML found for {idxml_path.name}")
            return False

        # Check if mzML is preprocessed
        paths = get_cache_paths(st.session_state.workspace, mzml_path)
        if not raw_cache_is_valid(mzml_path, paths):
            st.warning(f"mzML file {mzml_path.name} needs to be preprocessed first")
            return False

        im_info = load_im_info(paths)

        # Process idXML
        id_paths = get_id_cache_paths(st.session_state.workspace, idxml_path)
        if not id_cache_is_valid(idxml_path, id_paths):
            st.write(f"Extracting {idxml_path.name}...")
            extract_idxml_to_parquet(idxml_path, id_paths, lambda msg: st.write(msg))

            # Link identifications to spectra
            st.write("Linking identifications to spectra...")
            id_df = pl.read_parquet(id_paths["identifications"])
            metadata_df = pl.read_parquet(paths["metadata"])
            linked_df = link_identifications_to_spectra(
                id_df, metadata_df, status_callback=lambda msg: st.write(msg)
            )
            linked_df.write_parquet(id_paths["identifications"])

            # Create spectra_table_with_ids
            st.write("Creating spectra table with ID counts...")
            create_spectra_table_with_ids(paths, linked_df, im_info, lambda msg: st.write(msg))

            # Create ID components
            st.write("Creating identification components...")
            search_params = load_search_params(id_paths)
            create_id_components(paths, id_paths, im_info, file_id=mzml_path.stem, search_params=search_params)

            st.success(f"Processed {idxml_path.name}")
        else:
            st.info(f"{idxml_path.name} already processed")

        return True
    except Exception as e:
        st.error(f"Error processing {idxml_path.name}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

st.title("File Upload")

# Check if there are any files in the workspace
mzML_dir = Path(st.session_state.workspace, "mzML-files")
# TOPPView-Lite: disabled auto-loading example files
# if not any(Path(mzML_dir).iterdir()):
#     # No files present, load example data
#     fileupload.load_example_mzML_files()


def get_file_status(mzml_path: Path) -> str:
    """Get preprocessing status for a file. (TOPPView-Lite addition)"""
    paths = get_cache_paths(st.session_state.workspace, mzml_path)
    if raw_cache_is_valid(mzml_path, paths) and component_cache_is_valid(paths):
        return "Ready"
    return "Not processed"


def get_matching_idxml_status(mzml_path: Path) -> str:
    """Check if there's a matching idXML file for the mzML."""
    idxml_path = find_matching_idxml(mzml_path, st.session_state.workspace)
    if idxml_path:
        return f"✓ {idxml_path.name}"
    return "—"


def preprocess_file(mzml_path: Path) -> bool:
    """Run preprocessing pipeline for a file. (TOPPView-Lite addition)"""
    paths = get_cache_paths(st.session_state.workspace, mzml_path)
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
            idxml_path = find_matching_idxml(mzml_path, st.session_state.workspace)
            if idxml_path:
                st.write(f"Found matching idXML: {idxml_path.name}")
                id_paths = get_id_cache_paths(st.session_state.workspace, idxml_path)
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

tabs = ["File Upload"]
if st.session_state.location == "local":
    tabs.append("Files from local folder")

tabs = st.tabs(tabs)

with tabs[0]:
    with st.form("mzML-upload", clear_on_submit=True):
        files = st.file_uploader(
            "mzML files", accept_multiple_files=(st.session_state.location == "local")
        )
        cols = st.columns(3)
        if cols[1].form_submit_button("Add files to workspace", type="primary"):
            if files:
                fileupload.save_uploaded_mzML(files)
            else:
                st.warning("Select files first.")

# Local file upload option: via directory path
if st.session_state.location == "local":
    with tabs[1]:
        st_cols = st.columns([0.05, 0.95], gap="small")
        with st_cols[0]:
            st.write("\n")
            st.write("\n")
            dialog_button = st.button(
                "📁",
                key="local_browse",
                help="Browse for your local directory with MS data.",
                disabled=not TK_AVAILABLE,
            )
            if dialog_button:
                st.session_state["local_dir"] = tk_directory_dialog(
                    "Select directory with your MS data",
                    st.session_state["previous_dir"],
                )
                st.session_state["previous_dir"] = st.session_state["local_dir"]
        with st_cols[1]:
            # with st.form("local-file-upload"):
            local_mzML_dir = st.text_input(
                "path to folder with mzML files", value=st.session_state["local_dir"]
            )
        # raw string for file paths
        local_mzML_dir = rf"{local_mzML_dir}"
        cols = st.columns([0.65, 0.3, 0.4, 0.25], gap="small")
        copy_button = cols[1].button(
            "Copy files to workspace", type="primary", disabled=(local_mzML_dir == "")
        )
        use_copy = cols[2].checkbox(
            "Make a copy of files",
            key="local_browse-copy_files",
            value=True,
            help="Create a copy of files in workspace.",
        )
        if not use_copy:
            st.warning(
                "**Warning**: You have deselected the `Make a copy of files` option. "
                "This **_assumes you know what you are doing_**. "
                "This means that the original files will be used instead. "
            )
        if copy_button:
            fileupload.copy_local_mzML_files_from_directory(local_mzML_dir, use_copy)

if any(Path(mzML_dir).iterdir()):
    v_space(2)
    # Display all mzML files currently in workspace
    # TOPPView-Lite: collect file paths for status check
    mzml_files = [f for f in Path(mzML_dir).iterdir() if "external_files.txt" not in f.name]
    df = pd.DataFrame(
        {
            "file name": [f.name for f in mzml_files],
            "status": [get_file_status(f) for f in mzml_files],
            "idXML": [get_matching_idxml_status(f) for f in mzml_files],
        }
    )

    # Check if local files are available
    external_files = Path(mzML_dir, "external_files.txt")
    if external_files.exists():
        with open(external_files, "r") as f_handle:
            ext_paths = [Path(line.strip()) for line in f_handle.readlines()]
            ext_df = pd.DataFrame({
                "file name": [p.name for p in ext_paths],
                "status": [get_file_status(p) for p in ext_paths],
                "idXML": [get_matching_idxml_status(p) for p in ext_paths],
            })
            df = pd.concat([df, ext_df], ignore_index=True)

    st.markdown("##### mzML files in current workspace:")
    show_table(df)

    # TOPPView-Lite: Add processing and navigation buttons
    ready_count = len(df[df["status"] == "Ready"])
    unprocessed_count = len(df) - ready_count

    # Process button for unprocessed files
    if unprocessed_count > 0:
        cols = st.columns(3)
        if cols[1].button(f"Process {unprocessed_count} file(s)", type="primary"):
            for f in mzml_files:
                if get_file_status(f) != "Ready":
                    preprocess_file(f)
            if external_files.exists():
                with open(external_files, "r") as fh:
                    for line in fh:
                        f = Path(line.strip())
                        if f.exists() and get_file_status(f) != "Ready":
                            preprocess_file(f)
            st.rerun()

    # Navigation to viewer
    if ready_count > 0:
        if ready_count == len(df):
            st.success(f"All {ready_count} file(s) ready for viewing!")
        cols = st.columns(3)
        if cols[1].button("Go to Viewer", type="primary" if unprocessed_count == 0 else "secondary"):
            st.switch_page("content/viewer.py")

    v_space(1)
    # Remove files
    with st.expander("🗑️ Remove mzML files"):
        to_remove = st.multiselect(
            "select mzML files", options=[f.stem for f in sorted(mzML_dir.iterdir())]
        )
        c1, c2 = st.columns(2)
        if c2.button(
            "Remove **selected**", type="primary", disabled=not any(to_remove)
        ):
            params = fileupload.remove_selected_mzML_files(to_remove, params)
            save_params(params)
            st.rerun()

        if c1.button("⚠️ Remove **all**", disabled=not any(mzML_dir.iterdir())):
            params = fileupload.remove_all_mzML_files(params)
            save_params(params)
            st.rerun()

# ======================= idXML Upload Section =======================
v_space(2)
st.markdown("---")
st.markdown("### Identification Data (Optional)")
st.markdown("""
Upload idXML files to enable peptide sequence visualization. Files are matched to mzML files by filename
(e.g., `sample.mzML` ↔ `sample.idXML`).
""")

idxml_dir = Path(st.session_state.workspace, "idXML-files")
idxml_dir.mkdir(parents=True, exist_ok=True)

idxml_tabs = ["idXML Upload"]
if st.session_state.location == "local":
    idxml_tabs.append("idXML from local folder")

idxml_tabs = st.tabs(idxml_tabs)

with idxml_tabs[0]:
    with st.form("idXML-upload", clear_on_submit=True):
        idxml_files = st.file_uploader(
            "idXML files",
            accept_multiple_files=(st.session_state.location == "local"),
            type=["idXML"],
        )
        cols = st.columns(3)
        if cols[1].form_submit_button("Add idXML files", type="primary"):
            if idxml_files:
                saved_paths = fileupload.save_uploaded_idXML(idxml_files)
                # Process each saved idXML file
                if saved_paths:
                    for idxml_path in saved_paths:
                        process_idxml_file(idxml_path)
            else:
                st.warning("Select files first.")

if st.session_state.location == "local" and len(idxml_tabs) > 1:
    with idxml_tabs[1]:
        st_cols = st.columns([0.05, 0.95], gap="small")
        with st_cols[0]:
            st.write("\n")
            st.write("\n")
            idxml_dialog_button = st.button(
                "📁",
                key="idxml_local_browse",
                help="Browse for your local directory with idXML data.",
                disabled=not TK_AVAILABLE,
            )
            if idxml_dialog_button:
                st.session_state["idxml_local_dir"] = tk_directory_dialog(
                    "Select directory with your idXML data",
                    st.session_state.get("previous_dir", ""),
                )
        with st_cols[1]:
            local_idxml_dir = st.text_input(
                "path to folder with idXML files",
                value=st.session_state.get("idxml_local_dir", ""),
            )
        local_idxml_dir = rf"{local_idxml_dir}"
        cols = st.columns([0.65, 0.35], gap="small")
        idxml_copy_button = cols[1].button(
            "Copy idXML files", type="primary", disabled=(local_idxml_dir == "")
        )
        if idxml_copy_button:
            fileupload.copy_local_idXML_files_from_directory(local_idxml_dir, make_copy=True)

# Show current idXML files
idxml_files_list = fileupload.get_idxml_files()
if idxml_files_list:
    # Check processing status for each file
    mzml_dir = Path(st.session_state.workspace, "mzML-files")
    status_list = []
    unprocessed_files = []
    for idxml_file in idxml_files_list:
        mzml_exists = (mzml_dir / f"{idxml_file.stem}.mzML").exists()
        id_paths = get_id_cache_paths(st.session_state.workspace, idxml_file)
        if not mzml_exists:
            status = "No matching mzML"
        elif not id_cache_is_valid(idxml_file, id_paths):
            status = "Needs processing"
            unprocessed_files.append(idxml_file)
        else:
            status = "Ready"
        status_list.append(status)

    idxml_df = pd.DataFrame({
        "idXML file": [f.name for f in idxml_files_list],
        "Status": status_list,
    })
    st.markdown("##### idXML files in workspace:")
    show_table(idxml_df)

    # Show process button if there are unprocessed files
    if unprocessed_files:
        if st.button(f"Process {len(unprocessed_files)} idXML file(s)", type="primary"):
            for idxml_file in unprocessed_files:
                process_idxml_file(idxml_file)
            st.rerun()

    with st.expander("🗑️ Remove idXML files"):
        to_remove_idxml = st.multiselect(
            "select idXML files",
            options=[f.stem for f in idxml_files_list],
        )
        c1, c2 = st.columns(2)
        if c2.button("Remove **selected** idXML", type="primary", disabled=not any(to_remove_idxml)):
            fileupload.remove_selected_idXML_files(to_remove_idxml)
            st.rerun()
        if c1.button("⚠️ Remove **all** idXML", disabled=not idxml_files_list):
            fileupload.remove_all_idXML_files()
            st.rerun()

save_params(params)
