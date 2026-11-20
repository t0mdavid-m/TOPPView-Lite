"""The precursor marker of fragment spectra on the spectrum plot."""

import polars as pl
import pytest

from src.components.precursor_marker import (
    PRECURSOR_COLOR,
    precursor_markers,
    with_precursor,
)


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {
            "scan_id": [1, 2, 3, 4],
            "ms_level": pl.Series([1, 2, 2, 2], dtype=pl.Int8),
            "precursor_mz": pl.Series([0.0, 512.27, 640.5, 0.0], dtype=pl.Float32),
            "charge": [0, 2, 0, 3],
        }
    ).write_parquet(path)
    return path


def test_fragment_spectrum_marks_its_precursor(metadata):
    markers = precursor_markers(metadata, 2)
    assert markers == [{"x": pytest.approx(512.27, abs=1e-3), "label": "Precursor 2+", "color": PRECURSOR_COLOR}]


def test_unknown_charge_is_left_out_of_the_label(metadata):
    assert precursor_markers(metadata, "3")[0]["label"] == "Precursor"


@pytest.mark.parametrize("scan_id", [1, 4, 99, None, "x"])
def test_no_marker_without_a_precursor(metadata, scan_id):
    assert precursor_markers(metadata, scan_id) is None


def test_no_marker_without_metadata(tmp_path):
    assert precursor_markers(tmp_path / "missing.parquet", 2) is None


def test_plot_unchanged_without_markers():
    plot = object()
    assert with_precursor(plot, "id", "/nowhere", None) is plot


def test_plot_from_cache_draws_the_marker(tmp_path):
    from openms_insight import LinePlot

    data = pl.DataFrame(
        {"scan_id": [2, 2], "peak_id": [0, 1], "mass": [300.1, 512.27], "intensity": [5.0, 9.0]}
    )
    plot = LinePlot(
        cache_id="spectrum_plot",
        data=data.lazy(),
        cache_path=str(tmp_path),
        filters={"spectrum": "scan_id"},
        x_column="mass",
        y_column="intensity",
    )
    markers = [{"x": 512.27, "label": "Precursor 2+", "color": PRECURSOR_COLOR}]
    marked = with_precursor(plot, "spectrum_plot", str(tmp_path), markers)
    args = marked._get_component_args()
    assert args["xMarkers"][0]["label"] == "Precursor 2+"
    assert args["xMarkers"][0]["x"] == pytest.approx(512.27)
    assert args["xMarkerTolerance"] == pytest.approx(0.02)
    assert "xMarkers" not in plot._get_component_args()
