"""The peak map's axes: RT unit, swapped axes, and the zoom kept across them."""

from unittest.mock import patch

import polars as pl
import pytest

from src.components.factory import create_peak_map
from src.components.peak_map_axes import axes_suffix, convert_zoom, zoom_identifier


def test_default_axes_keep_the_existing_cache_and_zoom():
    assert axes_suffix("min", False) == ""
    assert zoom_identifier("min", False) == "heatmap_zoom"
    assert zoom_identifier("s", True) == "heatmap_zoom_rt_s_swapped"
    assert len({zoom_identifier(u, s) for u in ("min", "s") for s in (False, True)}) == 4


def test_zoom_to_seconds_and_swapped():
    # The heatmap reports xRange/yRange
    zoom = {"xRange": [10.0, 20.0], "yRange": [400.0, 600.0]}
    converted = convert_zoom(zoom, ("min", False), ("s", True))
    assert converted["x"] == [400.0, 600.0]
    assert converted["y"] == pytest.approx([600.0, 1200.0])
    assert converted["xRange"] == converted["x"]
    assert converted["yRange"] == converted["y"]


def test_zoom_round_trip():
    zoom = {"x": [450.0, 500.0], "y": [300.0, 900.0]}
    there = convert_zoom(zoom, ("s", True), ("min", False))
    assert there["x"] == pytest.approx([5.0, 15.0])
    assert there["y"] == [450.0, 500.0]
    back = convert_zoom(there, ("min", False), ("s", True))
    assert back["x"] == pytest.approx(zoom["x"])
    assert back["y"] == pytest.approx(zoom["y"])


def test_zoom_on_one_axis_and_full_view():
    # Only RT zoomed: m/z stays in full on the other side
    converted = convert_zoom({"xRange": [1.0, 2.0], "yRange": [-1, -1]}, ("min", False), ("min", True))
    assert converted == {"xRange": [-1, -1], "yRange": [1.0, 2.0], "y": [1.0, 2.0]}
    assert convert_zoom(None, ("min", False), ("s", False)) is None
    assert convert_zoom({"xRange": [-1, -1], "yRange": [-1, -1]}, ("min", False), ("s", False)) is None


@pytest.fixture
def paths(tmp_path):
    path = tmp_path / "heatmap_input.parquet"
    pl.DataFrame({
        "peak_id": [0, 1],
        "scan_id": [1, 2],
        "retention_time": [1.5, 2.0],
        "mass": [400.0, 500.0],
        "intensity": [10.0, 20.0],
    }).write_parquet(path)
    return {"heatmap_input": path, "component_cache": tmp_path}


def test_peak_map_in_seconds_swapped(paths):
    with patch("src.components.factory.Heatmap") as heatmap:
        create_peak_map(paths, {"type": "none"}, "run", rt_unit="s", swapped=True)
    kwargs = heatmap.call_args.kwargs
    assert kwargs["cache_id"] == "run_peaks_heatmap_rt_s_swapped"
    assert kwargs["zoom_identifier"] == "heatmap_zoom_rt_s_swapped"
    assert (kwargs["x_column"], kwargs["y_column"]) == ("mass", "retention_time_s")
    assert kwargs["y_label"] == "Retention Time (s)"
    assert kwargs["data"].collect()["retention_time_s"].to_list() == pytest.approx([90.0, 120.0])
    assert "data_path" not in kwargs


def test_peak_map_in_minutes_swapped(paths):
    with patch("src.components.factory.Heatmap") as heatmap:
        create_peak_map(paths, {"type": "none"}, "run", rt_unit="min", swapped=True)
    kwargs = heatmap.call_args.kwargs
    assert kwargs["cache_id"] == "run_peaks_heatmap_rt_min_swapped"
    assert (kwargs["x_column"], kwargs["y_column"]) == ("mass", "retention_time")
    assert kwargs["data_path"] == str(paths["heatmap_input"])
    assert kwargs["filters"] is None
