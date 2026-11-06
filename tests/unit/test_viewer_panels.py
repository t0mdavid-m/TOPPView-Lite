"""The viewer's component catalog and its layout."""

from src.components.viewer_panels import (
    CATALOG,
    DEFAULT_LAYOUT,
    PANELS,
    editor_catalog,
)
from src.layout_manager import validate_layout


def test_every_catalog_entry_has_a_panel():
    assert {c["name"] for c in CATALOG} == set(PANELS)


def test_default_layout_is_valid_for_the_editor():
    assert validate_layout([DEFAULT_LAYOUT], editor_catalog()) == ""
    placed = {name for row in DEFAULT_LAYOUT for name in row}
    assert placed == set(PANELS)


def test_editor_catalog_says_when_a_component_shows():
    labels = {c["name"]: c["label"] for c in editor_catalog()}
    assert labels["peak_map"] == "Peak map"
    assert "idXML" in labels["id_table"]
    assert "featureXML" in labels["features"]


def test_each_experiment_has_its_own_keys():
    from pathlib import Path
    from src.components.viewer_panels import ViewerContext

    def ctx(index):
        return ViewerContext(
            workspace=Path("."), selected_file=Path("run.mzML"), paths={}, im_info={},
            file_id="run", cache_path="", state_manager=None, ms_components={}, index=index,
        )

    # The first experiment keeps the keys of a viewer with one file
    assert ctx(0).file_key == "run" and ctx(0).key("viewer_tools") == "viewer_tools"
    assert ctx(1).file_key == "run_e1" and ctx(1).key("viewer_tools") == "viewer_tools_e1"
    # Two experiments of a five-file layout validate
    assert validate_layout([DEFAULT_LAYOUT, DEFAULT_LAYOUT], editor_catalog()) == ""
