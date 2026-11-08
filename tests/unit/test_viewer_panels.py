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

class _FakeHeatmap:
    def __init__(self):
        self.boxes = "unset"
        self.rendered = False

    def set_boxes(self, boxes, **kwargs):
        self.boxes = boxes
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        self.rendered = True


def _ctx(tmp_path, heatmap, featurexml_path=None):
    from unittest.mock import MagicMock

    from src.components.viewer_panels import ViewerContext

    return ViewerContext(
        workspace=tmp_path,
        selected_file=tmp_path / "run.mzML",
        paths={},
        im_info={"type": "none"},
        file_id="run",
        cache_path=str(tmp_path),
        state_manager=MagicMock(),
        ms_components={"heatmap": heatmap},
        featurexml_path=featurexml_path,
    )


def test_peak_map_without_features_clears_the_boxes(tmp_path):
    from src.components import viewer_panels

    heatmap = _FakeHeatmap()
    viewer_panels.peak_map(_ctx(tmp_path, heatmap))
    assert heatmap.boxes is None and heatmap.rendered


def test_peak_map_draws_features_most_intense_first(tmp_path, monkeypatch):
    import polars as pl

    from src.components import viewer_panels

    parquet = tmp_path / "features.parquet"
    pl.DataFrame(
        {
            "feature_id": [0, 1],
            "intensity": [1.0, 5.0],
            "rt_start": [1.0, 2.0],
            "rt_end": [1.5, 2.5],
            "mz_start": [400.0, 500.0],
            "mz_end": [402.0, 502.0],
        }
    ).write_parquet(parquet)
    monkeypatch.setattr(viewer_panels.st, "toggle", lambda *a, **k: True)
    monkeypatch.setattr(
        viewer_panels, "_feature_paths", lambda ctx, consume=False: ({"features": parquet}, False)
    )
    monkeypatch.setattr(viewer_panels, "_show_selected_feature", lambda ctx, paths: None)

    heatmap = _FakeHeatmap()
    viewer_panels.peak_map(_ctx(tmp_path, heatmap, tmp_path / "run.featureXML"))

    assert heatmap.boxes["feature_id"].to_list() == [1, 0]
    assert heatmap.kwargs["selection"] == "feature"
    assert heatmap.kwargs["x_range"] == ("rt_start", "rt_end")


def test_peak_map_works_with_an_insight_without_boxes(tmp_path):
    from src.components import viewer_panels

    class OldHeatmap:
        rendered = False

        def __call__(self, **kwargs):
            OldHeatmap.rendered = True

    viewer_panels.peak_map(_ctx(tmp_path, OldHeatmap(), tmp_path / "run.featureXML"))
    assert OldHeatmap.rendered
