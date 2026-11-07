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
            st.Page(Path("content", "layout.py"), title="Layout", icon="🧩"),
            st.Page(Path("content", "tools.py"), title="Tools", icon="🧰"),
        ],
    }

    # Workspace sync with an online workspace (off unless enabled in settings.json).
    # It is opened from its sidebar panel, so its menu entry is hidden: this
    # Streamlit cannot register a page without listing it.
    sync = st.session_state.settings.get("workspace_sync", {}).get("enabled")
    if sync:
        list(pages.values())[-1].append(
            st.Page(Path("content", "workspace_sync.py"), title="Workspace Sync", icon="🔄")
        )

    pg = st.navigation(pages)
    if sync:
        st.html('<style>[data-testid="stSidebarNav"] li:has(a[href$="/workspace_sync"])'
                '{display: none;}</style>')
    pg.run()
