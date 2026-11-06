"""Smoke tests: every TOPPView-Lite page renders without an exception.

The template's ``test_gui.py`` launches the template's demo pages, which this
app does not ship, so this module covers the app's own pages instead.
"""
import json

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    "page", ["content/welcome.py", "content/upload.py", "content/viewer.py", "content/layout.py", "content/tools.py"]
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


@pytest.mark.parametrize("online, has_console", [(False, True), (True, False)])
def test_tools_console_only_in_local_app(online, has_console, tmp_path, monkeypatch):
    """The Python console runs arbitrary code, so the hosted app must not show it."""
    # Online workspaces live under $WORKSPACES_DIR, not settings["workspaces_dir"]
    monkeypatch.setenv("WORKSPACES_DIR", str(tmp_path / "online"))
    import shutil
    from pathlib import Path

    at = AppTest.from_file("content/tools.py")
    with open("settings.json", "r") as f:
        at.session_state.settings = json.load(f)
    at.session_state.settings["test"] = True
    at.session_state.settings["online_deployment"] = online
    at.session_state["controllo"] = True  # captcha solved
    at.session_state.settings["workspaces_dir"] = str(tmp_path)
    at.secrets["workspace"] = "test"
    at.run(timeout=60)
    workspace = Path(at.session_state.workspace)
    (workspace / "mzML-files").mkdir(parents=True, exist_ok=True)
    shutil.copy(_tiny_mzml(tmp_path), workspace / "mzML-files" / "tiny.mzML")
    at.run(timeout=60)
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    assert labels[0] == "Operations"
    assert ("Python console" in labels) is has_console


def _tiny_mzml(tmp_path):
    import numpy as np
    import pyopenms as oms

    exp = oms.MSExperiment()
    spec = oms.MSSpectrum()
    spec.setRT(1.0)
    spec.setMSLevel(1)
    spec.set_peaks((np.array([100.0, 200.0]), np.array([1.0, 2.0])))
    exp.addSpectrum(spec)
    path = tmp_path / "tiny.mzML"
    oms.MzMLFile().store(str(path), exp)
    return path

