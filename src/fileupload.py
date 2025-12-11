import shutil
from pathlib import Path

import streamlit as st

from src.common.common import reset_directory


@st.cache_data
def save_uploaded_mzML(uploaded_files: list[bytes]) -> None:
    """
    Saves uploaded mzML files to the mzML directory.

    Args:
        uploaded_files (List[bytes]): List of uploaded mzML files.

    Returns:
        None
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # A list of files is required, since online allows only single upload, create a list
    if st.session_state.location == "online":
        uploaded_files = [uploaded_files]
    # If no files are uploaded, exit early
    if not uploaded_files:
        st.warning("Upload some files first.")
        return
    # Write files from buffer to workspace mzML directory, add to selected files
    for f in uploaded_files:
        if f.name not in [f.name for f in mzML_dir.iterdir()] and f.name.endswith(
            "mzML"
        ):
            with open(Path(mzML_dir, f.name), "wb") as fh:
                fh.write(f.getbuffer())
    st.success("Successfully added uploaded files!")


def copy_local_mzML_files_from_directory(local_mzML_directory: str, make_copy: bool=True) -> None:
    """
    Copies local mzML files from a specified directory to the mzML directory.

    Args:
        local_mzML_directory (str): Path to the directory containing the mzML files.
        make_copy (bool): Whether to make a copy of the files in the workspace. Default is True. If False, local file paths will be written to an external_files.txt file.

    Returns:
        None
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # Check if local directory contains mzML files, if not exit early
    if not any(Path(local_mzML_directory).glob("*.mzML")):
        st.warning("No mzML files found in specified folder.")
        return
    # Copy all mzML files to workspace mzML directory, add to selected files
    files = Path(local_mzML_directory).glob("*.mzML")
    for f in files:
        if make_copy:
            shutil.copy(f, Path(mzML_dir, f.name))
        else:
            # Create a temporary file to store the path to the local directories
            external_files = Path(mzML_dir, "external_files.txt")
            # Check if the file exists, if not create it
            if not external_files.exists():
                external_files.touch()
            # Write the path to the local directories to the file
            with open(external_files, "a") as f_handle:
                f_handle.write(f"{f}\n")
                
    st.success("Successfully added local files!")


def load_example_mzML_files() -> None:
    """
    Copies example mzML files to the mzML directory.

    Args:
        None

    Returns:
        None
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # Copy files from example-data/mzML to workspace mzML directory, add to selected files
    for f in Path("example-data", "mzML").glob("*.mzML"):
        shutil.copy(f, mzML_dir)
    st.success("Example mzML files loaded!")


def remove_selected_mzML_files(to_remove: list[str], params: dict) -> dict:
    """
    Removes selected mzML files from the mzML directory.

    Args:
        to_remove (List[str]): List of mzML files to remove.
        params (dict): Parameters.


    Returns:
        dict: parameters with updated mzML files
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # remove all given files from mzML workspace directory and selected files
    for f in to_remove:
        Path(mzML_dir, f + ".mzML").unlink()
    for k, v in params.items():
        if isinstance(v, list):
            if f in v:
                params[k].remove(f)
    st.success("Selected mzML files removed!")
    return params


def remove_all_mzML_files(params: dict) -> dict:
    """
    Removes all mzML files from the mzML directory.

    Args:
        params (dict): Parameters.

    Returns:
        dict: parameters with updated mzML files
    """
    mzML_dir = Path(st.session_state.workspace, "mzML-files")
    # reset (delete and re-create) mzML directory in workspace
    reset_directory(mzML_dir)
    # reset all parameter items which have mzML in key and are list
    for k, v in params.items():
        if "mzML" in k and isinstance(v, list):
            params[k] = []
    st.success("All mzML files removed!")
    return params


# ======================= idXML file functions =======================

def save_uploaded_idXML(uploaded_files: list[bytes]) -> list[Path]:
    """
    Saves uploaded idXML files to the idXML directory.

    Args:
        uploaded_files: List of uploaded idXML files.

    Returns:
        List of paths to saved idXML files.
    """
    idxml_dir = Path(st.session_state.workspace, "idXML-files")
    idxml_dir.mkdir(parents=True, exist_ok=True)

    # A list of files is required, since online allows only single upload
    if st.session_state.location == "online":
        uploaded_files = [uploaded_files]

    if not uploaded_files:
        st.warning("Upload some files first.")
        return []

    saved_paths = []
    for f in uploaded_files:
        if f.name not in [f.name for f in idxml_dir.iterdir()] and f.name.endswith("idXML"):
            file_path = Path(idxml_dir, f.name)
            with open(file_path, "wb") as fh:
                fh.write(f.getbuffer())
            saved_paths.append(file_path)
    st.success("Successfully added uploaded idXML files!")
    return saved_paths


def copy_local_idXML_files_from_directory(local_idXML_directory: str, make_copy: bool = True) -> None:
    """
    Copies local idXML files from a specified directory to the idXML directory.

    Args:
        local_idXML_directory: Path to the directory containing the idXML files.
        make_copy: Whether to make a copy of the files in the workspace.
    """
    idxml_dir = Path(st.session_state.workspace, "idXML-files")
    idxml_dir.mkdir(parents=True, exist_ok=True)

    if not any(Path(local_idXML_directory).glob("*.idXML")):
        st.warning("No idXML files found in specified folder.")
        return

    files = Path(local_idXML_directory).glob("*.idXML")
    for f in files:
        if make_copy:
            shutil.copy(f, Path(idxml_dir, f.name))
        else:
            external_files = Path(idxml_dir, "external_files.txt")
            if not external_files.exists():
                external_files.touch()
            with open(external_files, "a") as f_handle:
                f_handle.write(f"{f}\n")

    st.success("Successfully added local idXML files!")


def get_idxml_files() -> list[Path]:
    """Get list of idXML files in the workspace."""
    idxml_dir = Path(st.session_state.workspace, "idXML-files")
    if not idxml_dir.exists():
        return []

    files = [f for f in idxml_dir.iterdir() if f.suffix.lower() == ".idxml"]

    # Check for external files
    external_files = idxml_dir / "external_files.txt"
    if external_files.exists():
        with open(external_files, "r") as f_handle:
            for line in f_handle:
                path = Path(line.strip())
                if path.exists():
                    files.append(path)

    return files


def remove_selected_idXML_files(to_remove: list[str]) -> None:
    """
    Removes selected idXML files from the idXML directory.

    Args:
        to_remove: List of idXML file stems to remove.
    """
    idxml_dir = Path(st.session_state.workspace, "idXML-files")
    for f in to_remove:
        idxml_path = Path(idxml_dir, f + ".idXML")
        if idxml_path.exists():
            idxml_path.unlink()
    st.success("Selected idXML files removed!")


def remove_all_idXML_files() -> None:
    """Removes all idXML files from the idXML directory."""
    idxml_dir = Path(st.session_state.workspace, "idXML-files")
    if idxml_dir.exists():
        reset_directory(idxml_dir)
    st.success("All idXML files removed!")
