import streamlit as st
from pathlib import Path
import json
# For some reason the windows version only works if this is imported here
import pyopenms

if "settings" not in st.session_state:
        with open("settings.json", "r") as f:
            st.session_state.settings = json.load(f)

if __name__ == '__main__':
    pages = {
        str(st.session_state.settings["app-name"]): [
            st.Page(Path("content", "welcome.py"), title="Welcome", icon="👋", default=True),
            st.Page(Path("content", "upload.py"), title="Upload", icon="📂"),
            st.Page(Path("content", "viewer.py"), title="Viewer", icon="👀"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()
