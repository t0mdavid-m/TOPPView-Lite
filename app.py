import streamlit as st
from pathlib import Path
import json
# For some reason the windows version only works if this is imported here
import pyopenms

if "settings" not in st.session_state:
        with open("settings.json", "r") as f:
            st.session_state.settings = json.load(f)

if __name__ == '__main__':
    # Check if a file is being loaded via CLI (load_file query parameter)
    # If so, redirect directly to the viewer page
    load_file = st.query_params.get("load_file")
    if load_file:
        default_page = Path("content", "viewer.py")
    else:
        default_page = Path("content", "welcome.py")

    pages = {
        str(st.session_state.settings["app-name"]): [
            st.Page(Path("content", "welcome.py"), title="Welcome", icon="👋", default=(not load_file)),
            st.Page(Path("content", "upload.py"), title="Upload", icon="📂"),
            st.Page(Path("content", "viewer.py"), title="Viewer", icon="👀", default=bool(load_file)),
        ],
    }

    pg = st.navigation(pages)
    pg.run()
