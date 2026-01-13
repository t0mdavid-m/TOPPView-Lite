"""Export helper for 'View in TOPPView-Lite' button.

This module is designed to be copied into other OpenMS Streamlit template apps
to enable the "View in TOPPView-Lite" integration.

Usage:
    from src.integration.toppview_export import render_toppview_button

    # Single file:
    render_toppview_button(
        mzml_paths=Path("results/sample.mzML"),
        idxml_paths=Path("results/sample.idXML"),
        app_name="MHCquant",
    )

    # Multiple files:
    render_toppview_button(
        mzml_paths=[Path("results/sample1.mzML"), Path("results/sample2.mzML")],
        idxml_paths=[Path("results/sample1.idXML"), Path("results/sample2.idXML")],
        app_name="MHCquant",
    )

Configuration (add to your app's settings.json):
    {
        "toppview_shared_exports": "/app/shared-exports",
        "toppview_lite_url": "http://toppview-lite:8501"
    }
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List

import streamlit as st


# Default configuration values
_DEFAULT_SHARED_EXPORTS = "/app/shared-exports"
_DEFAULT_TOPPVIEW_URL = "http://localhost:28513"


def _get_shared_exports_path() -> Path:
    """Get shared exports path from settings.json or default."""
    if "settings" in st.session_state:
        path = st.session_state.settings.get("toppview_shared_exports", "")
        if path:
            return Path(path)
    return Path(_DEFAULT_SHARED_EXPORTS)


def _get_toppview_url() -> str:
    """Get TOPPView-Lite URL from settings.json or default."""
    if "settings" in st.session_state:
        url = st.session_state.settings.get("toppview_lite_url", "")
        if url:
            return url
    return _DEFAULT_TOPPVIEW_URL


def _to_list(paths: Union[Path, List[Path], None]) -> List[Path]:
    """Convert single path or list to list."""
    if paths is None:
        return []
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def is_toppview_available() -> bool:
    """Check if TOPPView-Lite integration is available.

    Returns:
        True if shared exports volume is mounted and accessible.
    """
    return _get_shared_exports_path().exists()


def export_to_toppview(
    mzml_paths: Union[Path, List[Path]],
    idxml_paths: Optional[Union[Path, List[Path]]] = None,
    app_name: str = "unknown",
    return_url: Optional[str] = None,
    workspace_hash: Optional[str] = None,
) -> str:
    """Export files to the shared TOPPView-Lite volume.

    Args:
        mzml_paths: Path or list of paths to mzML files.
        idxml_paths: Optional path or list of paths to idXML files.
        app_name: Name of the source application.
        return_url: URL to return to the source app. If None, constructs from workspace.
        workspace_hash: Optional workspace hash. If None, uses current workspace ID.

    Returns:
        URL to open in TOPPView-Lite.

    Raises:
        RuntimeError: If shared exports volume is not available.
    """
    shared_exports_path = _get_shared_exports_path()

    if not shared_exports_path.exists():
        raise RuntimeError(
            "TOPPView-Lite integration not available. "
            f"Shared exports volume not found at {shared_exports_path}"
        )

    # Convert to lists
    mzml_list = _to_list(mzml_paths)
    idxml_list = _to_list(idxml_paths)

    if not mzml_list:
        raise ValueError("At least one mzML file must be provided")

    # Use existing workspace hash or generate one
    if workspace_hash is None:
        workspace_hash = st.query_params.get("workspace", "default")

    # Create export directory structure
    export_dir = shared_exports_path / workspace_hash
    mzml_dir = export_dir / "mzML-files"
    idxml_dir = export_dir / "idXML-files"

    mzml_dir.mkdir(parents=True, exist_ok=True)
    idxml_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    # Copy mzML files
    for mzml_path in mzml_list:
        if mzml_path.exists():
            dst_mzml = mzml_dir / mzml_path.name
            if not dst_mzml.exists():
                shutil.copy(mzml_path, dst_mzml)
            exported_files.append(mzml_path.name)

    # Copy idXML files
    for idxml_path in idxml_list:
        if idxml_path.exists():
            dst_idxml = idxml_dir / idxml_path.name
            if not dst_idxml.exists():
                shutil.copy(idxml_path, dst_idxml)
            exported_files.append(idxml_path.name)

    # Construct return URL if not provided
    if return_url is None:
        return_url = f"/?workspace={workspace_hash}"

    # Write source info for TOPPView-Lite to display return link
    source_info = {
        "app_name": app_name,
        "return_url": return_url,
        "exported_at": datetime.now().isoformat(),
        "files": exported_files,
    }
    source_info_path = export_dir / "source_info.json"
    source_info_path.write_text(json.dumps(source_info, indent=2))

    # Generate TOPPView-Lite URL
    toppview_url = _get_toppview_url()
    return f"{toppview_url}/?workspace={workspace_hash}"


def render_toppview_button(
    mzml_paths: Union[Path, List[Path]],
    idxml_paths: Optional[Union[Path, List[Path]]] = None,
    app_name: str = "unknown",
    return_url: Optional[str] = None,
    button_text: str = "View in TOPPView-Lite",
    button_type: str = "secondary",
) -> bool:
    """Render a button that exports data and provides a link to TOPPView-Lite.

    Args:
        mzml_paths: Path or list of paths to mzML files.
        idxml_paths: Optional path or list of paths to idXML files.
        app_name: Name of the source application.
        return_url: URL to return to the source app.
        button_text: Text to display on button.
        button_type: Streamlit button type ("primary" or "secondary").

    Returns:
        True if button was clicked and export succeeded.
    """
    # Check if integration is available
    if not is_toppview_available():
        st.caption(
            "TOPPView-Lite integration not available (shared volume not mounted)"
        )
        return False

    # Convert to list for counting
    mzml_list = _to_list(mzml_paths)
    file_count = len(mzml_list)

    # Create unique button key
    if file_count == 1:
        button_key = f"toppview_export_{mzml_list[0].stem}"
    else:
        button_key = f"toppview_export_{file_count}_files"

    if st.button(f"{button_text}", type=button_type, key=button_key):
        with st.spinner(f"Preparing export ({file_count} file{'s' if file_count > 1 else ''})..."):
            try:
                url = export_to_toppview(
                    mzml_paths=mzml_paths,
                    idxml_paths=idxml_paths,
                    app_name=app_name,
                    return_url=return_url,
                )
                st.success(f"Export ready! ({file_count} file{'s' if file_count > 1 else ''})")
                st.markdown(f"[Open in TOPPView-Lite]({url})")
                return True
            except Exception as e:
                st.error(f"Export failed: {e}")
                return False

    return False
