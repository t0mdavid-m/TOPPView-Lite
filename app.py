import streamlit as st
from pathlib import Path
import json
# For some reason the windows version only works if this is imported here
import pyopenms

if "settings" not in st.session_state:
        with open("settings.json", "r") as f:
            st.session_state.settings = json.load(f)


def check_cli_startup() -> bool:
    """Check for CLI startup configuration and set up session state.

    Returns:
        True if CLI startup config was found and applied
    """
    cli_config_path = Path(__file__).parent / ".cli_startup.json"
    if not cli_config_path.exists():
        return False

    try:
        with open(cli_config_path, "r") as f:
            config = json.load(f)

        # Store CLI config in session state
        if "cli_startup" not in st.session_state:
            st.session_state.cli_startup = config
            st.session_state.cli_auto_process = config.get("auto_process", True)

            # Set query param to use CLI workspace
            if "workspace" not in st.query_params:
                st.query_params.workspace = "cli-workspace"

        return True
    except (json.JSONDecodeError, IOError):
        return False


if __name__ == '__main__':
    # Check for CLI startup configuration
    has_cli_config = check_cli_startup()

    # Determine default page based on CLI startup
    default_page = "welcome"
    if has_cli_config and st.session_state.get("cli_startup"):
        # If CLI provided files, go directly to upload page for processing
        default_page = "upload"

    pages = {
        str(st.session_state.settings["app-name"]): [
            st.Page(Path("content", "welcome.py"), title="Welcome", icon="👋", default=(default_page == "welcome")),
            st.Page(Path("content", "upload.py"), title="Upload", icon="📂", default=(default_page == "upload")),
            st.Page(Path("content", "viewer.py"), title="Viewer", icon="👀"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()
