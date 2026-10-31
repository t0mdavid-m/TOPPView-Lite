"""Smoke tests: every TOPPView-Lite page renders without an exception.

The template's ``test_gui.py`` launches the template's demo pages, which this
app does not ship, so this module covers the app's own pages instead.
"""
import json

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    "page", ["content/welcome.py", "content/upload.py", "content/viewer.py"]
)
def test_page_renders(page, tmp_path):
    at = AppTest.from_file(page)
    with open("settings.json", "r") as f:
        at.session_state.settings = json.load(f)
    at.session_state.settings["test"] = True
    at.session_state.settings["workspaces_dir"] = str(tmp_path)
    at.secrets["workspace"] = "test"
    at.run(timeout=60)
    assert not at.exception, at.exception
