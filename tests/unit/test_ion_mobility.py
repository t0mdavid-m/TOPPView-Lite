"""Per-peak ion mobility: extraction, map and mobilogram inputs, components."""

import json

import numpy as np
import polars as pl
import pyopenms as oms
import pytest

from src.preprocessing import (
    component_cache_is_valid,
    create_component_inputs,
    ensure_im_inputs,
    extract_mzml_to_parquet,
    get_cache_paths,
    raw_cache_is_valid,
)
from src.preprocessing import ion_mobility
from src.preprocessing.ion_mobility import create_im_inputs


def _spectrum(rt, mz, intensity, mobility=None, array_name="mean inverse reduced ion mobility array"):
    spec = oms.MSSpectrum()
    spec.setRT(rt)
    spec.setMSLevel(1)
    spec.set_peaks((np.asarray(mz, float), np.asarray(intensity, float)))
    if mobility is not None:
        fda = oms.FloatDataArray()
        fda.setName(array_name)
        fda.set_data(np.asarray(mobility, dtype=np.float32))
        spec.setFloatDataArrays([fda])
    return spec


@pytest.fixture
def im_mzml(tmp_path):
    exp = oms.MSExperiment()
    exp.addSpectrum(_spectrum(60.0, [400.0, 500.0, 600.0], [10.0, 20.0, 30.0], [0.8, 1.0, 1.2]))
    exp.addSpectrum(_spectrum(61.0, [450.0, 550.0], [5.0, 7.0], [0.9, 0.9]))
    exp.addSpectrum(_spectrum(62.0, [480.0], [3.0]))  # no ion mobility array
    path = tmp_path / "mzML-files" / "im.mzML"
    path.parent.mkdir()
    oms.MzMLFile().store(str(path), exp)
    return path


def _extract(path, workspace):
    paths = get_cache_paths(workspace, path)
    extract_mzml_to_parquet(path, paths)
    create_component_inputs(paths)
    return paths


def test_peaks_keep_their_ion_mobility(im_mzml, tmp_path):
    paths = _extract(im_mzml, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert info["type"] == "tims" and info["unit"] == "Vs/cm2"

    peaks = pl.read_parquet(paths["peaks"]).sort("peak_id")
    mobility = peaks["ion_mobility"].to_list()
    assert mobility[:5] == pytest.approx([0.8, 1.0, 1.2, 0.9, 0.9])
    assert np.isnan(mobility[5])  # third spectrum has no array
    assert raw_cache_is_valid(im_mzml, paths)
    # No FAIMS summary table for per-peak ion mobility, and the cache is complete without it
    assert not paths["im_table"].exists()
    assert component_cache_is_valid(paths)


def test_old_cache_without_ion_mobility_is_extracted_again(im_mzml, tmp_path):
    paths = _extract(im_mzml, tmp_path)
    pl.read_parquet(paths["peaks"]).drop("ion_mobility").write_parquet(paths["peaks"])
    assert not raw_cache_is_valid(im_mzml, paths)


def test_im_info_without_version_is_extracted_again(im_mzml, tmp_path):
    paths = _extract(im_mzml, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    del info["version"]
    paths["im_info"].write_text(json.dumps(info))
    assert not raw_cache_is_valid(im_mzml, paths)


def test_im_inputs(im_mzml, tmp_path):
    paths = _extract(im_mzml, tmp_path)
    mtime = ensure_im_inputs(paths, json.loads(paths["im_info"].read_text()))
    assert mtime is not None

    im_peaks = pl.read_parquet(paths["im_peaks"])
    assert im_peaks.height == 5
    assert im_peaks.columns == ["peak_id", "scan_id", "mass", "ion_mobility", "intensity"]

    mobilogram = pl.read_parquet(paths["mobilogram"])
    # Scan 2 has both peaks at 0.9: one bin with their summed intensity
    scan2 = mobilogram.filter(pl.col("scan_id") == 2)
    assert scan2.height == 1
    assert scan2["intensity"][0] == pytest.approx(12.0)
    assert scan2["ion_mobility"][0] == pytest.approx(0.9, abs=(1.2 - 0.8) / ion_mobility.MOBILOGRAM_BINS)
    assert mobilogram["intensity"].sum() == pytest.approx(72.0)


def test_binned_level_keeps_the_most_intense_peak_per_bin(tmp_path, monkeypatch):
    paths = {
        "peaks": tmp_path / "peaks.parquet",
        "im_peaks": tmp_path / "im_peaks.parquet",
        "im_peaks_binned": tmp_path / "im_peaks_binned.parquet",
        "mobilogram": tmp_path / "mobilogram.parquet",
    }
    pl.DataFrame({
        "peak_id": [0, 1, 2, 3],
        "scan_id": [1, 1, 1, 2],
        "retention_time": [1.0] * 4,
        "mass": [100.0, 100.001, 200.0, 100.0],
        "intensity": [5.0, 9.0, 1.0, 2.0],
        "im_id": [-1] * 4,
        "ion_mobility": [1.0, 1.0, 1.5, 1.0],
    }).write_parquet(paths["peaks"])
    monkeypatch.setattr(ion_mobility, "MZ_BINS", 10)
    monkeypatch.setattr(ion_mobility, "IM_BINS", 10)
    assert create_im_inputs(paths)
    binned = pl.read_parquet(paths["im_peaks_binned"]).sort("peak_id")
    # Peaks 0 and 1 share a bin in scan 1: the more intense one stays
    assert binned["peak_id"].to_list() == [1, 2, 3]


def test_files_without_peak_mobility_get_no_inputs(tmp_path):
    exp = oms.MSExperiment()
    exp.addSpectrum(_spectrum(60.0, [400.0], [1.0]))
    path = tmp_path / "mzML-files" / "plain.mzML"
    path.parent.mkdir()
    oms.MzMLFile().store(str(path), exp)
    paths = _extract(path, tmp_path)
    assert ensure_im_inputs(paths, json.loads(paths["im_info"].read_text())) is None
    assert "ion_mobility" not in pl.read_parquet(paths["peaks"]).columns


def test_im_components(im_mzml, tmp_path):
    """Components are mocked: Insight preprocessing in a subprocess is covered by the browser check."""
    from unittest.mock import patch

    from src.components import factory

    paths = _extract(im_mzml, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    mtime = ensure_im_inputs(paths, info)
    with patch.object(factory, "Heatmap") as heatmap, patch.object(factory, "SeriesPlot") as series:
        components = factory.create_im_components(paths, info, mtime, "im")
    assert set(components) == {"im_map", "mobilogram"}

    map_kwargs = heatmap.call_args.kwargs
    assert map_kwargs["cache_id"] == "im_im_map"
    assert (map_kwargs["x_column"], map_kwargs["y_column"]) == ("mass", "ion_mobility")
    assert map_kwargs["filters"] == {"spectrum": "scan_id"}
    assert map_kwargs["interactivity"] == {"peak": "peak_id"}
    assert map_kwargs["zoom_identifier"] != "heatmap_zoom"  # the peak map's
    assert len(map_kwargs["levels"]) == 1
    assert map_kwargs["y_label"] == "Ion mobility 1/K₀ (V·s/cm²)"
    assert map_kwargs["regenerate_cache"] is True  # no cache yet

    series_kwargs = series.call_args.kwargs
    assert series_kwargs["data_path"] == str(paths["mobilogram"])
    assert series_kwargs["filters"] == {"spectrum": "scan_id"}
    assert series_kwargs["x_column"] == "ion_mobility"


def test_stale_cache(tmp_path):
    from src.components.factory import _stale_cache

    assert _stale_cache(str(tmp_path), "c", 0.0)
    (tmp_path / "c").mkdir()
    manifest = tmp_path / "c" / "manifest.json"
    manifest.write_text("{}")
    mtime = manifest.stat().st_mtime
    assert not _stale_cache(str(tmp_path), "c", mtime - 1)
    assert _stale_cache(str(tmp_path), "c", mtime + 1)


def test_faims_needs_the_compensation_voltage_unit():
    from src.preprocessing import detect_ion_mobility_type

    plain = oms.MSExperiment()
    plain.addSpectrum(_spectrum(60.0, [400.0], [1.0]))  # drift time -1, unit NONE
    assert detect_ion_mobility_type(plain)[0] == "none"

    faims = oms.MSExperiment()
    for cv in (-40.0, -60.0):
        spec = _spectrum(60.0, [400.0], [1.0])
        spec.setDriftTime(cv)
        spec.setDriftTimeUnit(oms.DriftTimeUnit.FAIMS_COMPENSATION_VOLTAGE)
        faims.addSpectrum(spec)
    im_type, unit, cvs = detect_ion_mobility_type(faims)
    assert (im_type, unit, cvs) == ("faims", "V", [-40.0, -60.0])


def test_extraction_drops_insight_caches_of_the_previous_extraction(im_mzml, tmp_path):
    paths = _extract(im_mzml, tmp_path)
    stale = paths["component_cache"] / "im_peaks_heatmap" / "manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    extract_mzml_to_parquet(im_mzml, paths)
    assert not paths["component_cache"].exists()
