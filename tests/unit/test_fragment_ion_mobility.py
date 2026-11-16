"""Ion mobility of fragment spectra: per-peak values and precursor ion mobility."""

import json

import numpy as np
import polars as pl
import pyopenms as oms
import pytest

from src.preprocessing import (
    create_component_inputs,
    ensure_im_inputs,
    extract_mzml_to_parquet,
    get_cache_paths,
    raw_cache_is_valid,
)
from src.preprocessing.ion_mobility import precursor_im_column


def _ms1(rt, mz, intensity):
    spec = oms.MSSpectrum()
    spec.setRT(rt)
    spec.setMSLevel(1)
    spec.set_peaks((np.asarray(mz, float), np.asarray(intensity, float)))
    return spec


def _ms2(rt, precursor_mz, precursor_im, mz, intensity, mobility=None, unit=oms.DriftTimeUnit.VSSC):
    spec = _ms1(rt, mz, intensity)
    spec.setMSLevel(2)
    precursor = oms.Precursor()
    precursor.setMZ(precursor_mz)
    precursor.setCharge(2)
    if precursor_im is not None:
        precursor.setDriftTime(precursor_im)
        spec.setDriftTime(precursor_im)
        spec.setDriftTimeUnit(unit)
    spec.setPrecursors([precursor])
    if mobility is not None:
        fda = oms.FloatDataArray()
        fda.setName("inverse reduced ion mobility array")
        fda.set_data(np.asarray(mobility, dtype=np.float32))
        spec.setFloatDataArrays([fda])
    return spec


def _store(tmp_path, spectra, name="pasef.mzML"):
    exp = oms.MSExperiment()
    for spec in spectra:
        exp.addSpectrum(spec)
    path = tmp_path / "mzML-files" / name
    path.parent.mkdir(exist_ok=True)
    oms.MzMLFile().store(str(path), exp)
    return path


def _extract(path, workspace):
    paths = get_cache_paths(workspace, path)
    extract_mzml_to_parquet(path, paths)
    create_component_inputs(paths)
    return paths


@pytest.fixture
def pasef_mzml(tmp_path):
    """MS1 without ion mobility; fragment spectra with per-peak 1/K0."""
    return _store(tmp_path, [
        _ms1(60.0, [400.0, 500.0], [10.0, 20.0]),
        _ms2(61.0, 500.2, 0.95, [150.0, 250.0, 350.0], [3.0, 4.0, 5.0], [0.94, 0.95, 0.96]),
        _ms2(62.0, 410.7, 1.12, [175.0, 275.0], [6.0, 7.0], [1.11, 1.13]),
        _ms2(63.0, 455.5, None, [190.0], [2.0]),
    ])


def test_ion_mobility_found_in_fragment_spectra_only(pasef_mzml, tmp_path):
    paths = _extract(pasef_mzml, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert info["type"] == "tims" and info["unit"] == "Vs/cm2"

    peaks = pl.read_parquet(paths["peaks"]).sort("peak_id")
    mobility = peaks["ion_mobility"].to_list()
    assert np.isnan(mobility[0]) and np.isnan(mobility[1])  # MS1 has none
    assert mobility[2:7] == pytest.approx([0.94, 0.95, 0.96, 1.11, 1.13])
    assert raw_cache_is_valid(pasef_mzml, paths)


def test_ion_mobility_map_follows_a_fragment_spectrum(pasef_mzml, tmp_path):
    paths = _extract(pasef_mzml, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert ensure_im_inputs(paths, info) is not None
    im_peaks = pl.read_parquet(paths["im_peaks"])
    assert sorted(im_peaks["scan_id"].unique().to_list()) == [2, 3]
    mobilogram = pl.read_parquet(paths["mobilogram"])
    assert mobilogram.filter(pl.col("scan_id") == 2)["intensity"].sum() == pytest.approx(12.0)


def test_precursor_ion_mobility_in_spectra_table(pasef_mzml, tmp_path):
    paths = _extract(pasef_mzml, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert info["precursor_im_unit"] == "Vs/cm2"

    table = pl.read_parquet(paths["spectra_table"]).sort("scan_id")
    values = table["precursor_im"].to_list()
    # null (not NaN, which the table's JSON cannot hold) where there is none
    assert values[0] is None  # MS1
    assert values[1:3] == pytest.approx([0.95, 1.12])
    assert values[3] is None  # fragment spectrum without one
    assert table["precursor_im"].is_nan().sum() == 0

    column = precursor_im_column(info)
    assert column["field"] == "precursor_im" and column["title"] == "1/K₀"


def _as_written_by_msconvert(path):
    """Name the precursors' ion mobility 1/K0, as converters other than OpenMS do."""
    text = path.read_text().replace(
        'accession="MS:1002476" name="ion mobility drift time"',
        'accession="MS:1002815" name="inverse reduced ion mobility"',
    ).replace(
        'unitAccession="UO:0000028" unitName="millisecond" unitCvRef="UO"',
        'unitAccession="MS:1002814" unitName="volt-second per square centimeter" unitCvRef="MS"',
    )
    path.write_text(text)
    return path


def test_precursor_ion_mobility_without_per_peak_values(tmp_path):
    """A converted timsTOF file: summed MS1, PASEF spectra with only the precursor's 1/K0."""
    path = _as_written_by_msconvert(_store(tmp_path, [
        _ms1(60.0, [400.0], [10.0]),
        _ms2(61.0, 400.2, 0.88, [150.0], [3.0]),
    ]))
    paths = _extract(path, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert info["type"] == "none"
    assert info["precursor_im_unit"] == "Vs/cm2"
    table = pl.read_parquet(paths["spectra_table"]).sort("scan_id")
    assert table["precursor_im"].to_list()[1] == pytest.approx(0.88)
    assert precursor_im_column(info)["title"] == "1/K₀"


@pytest.mark.parametrize("unit", [oms.DriftTimeUnit.MILLISECOND, oms.DriftTimeUnit.NONE, oms.DriftTimeUnit.VSSC])
def test_precursor_drift_time_written_by_openms(tmp_path, unit):
    # OpenMS writes every precursor drift time as milliseconds, so without
    # per-peak values the unit is not known
    path = _store(tmp_path, [_ms1(60.0, [400.0], [1.0]), _ms2(61.0, 400.2, 0.9, [150.0], [3.0], unit=unit)])
    paths = _extract(path, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert info["precursor_im_unit"] == "unknown"
    assert precursor_im_column(info)["title"] == "IM"
    assert pl.read_parquet(paths["spectra_table"])["precursor_im"].to_list()[1] == pytest.approx(0.9)


def test_precursor_ccs(tmp_path):
    path = _store(tmp_path, [_ms1(60.0, [400.0], [1.0]), _ms2(61.0, 400.2, 410.0, [150.0], [3.0])])
    text = path.read_text().replace(
        'accession="MS:1002476" name="ion mobility drift time" value="410"',
        'accession="MS:1002954" name="collisional cross sectional area" value="410"',
    ).replace('unitAccession="UO:0000028" unitName="millisecond"', 'unitAccession="UO:0000324" unitName="square angstrom"')
    path.write_text(text)
    paths = _extract(path, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert precursor_im_column(info)["title"] == "CCS (Å²)"


def test_no_precursor_ion_mobility_no_column(tmp_path):
    path = _store(tmp_path, [_ms1(60.0, [400.0], [1.0]), _ms2(61.0, 400.2, None, [150.0], [3.0])])
    paths = _extract(path, tmp_path)
    info = json.loads(paths["im_info"].read_text())
    assert info["precursor_im_unit"] == ""
    assert precursor_im_column(info) is None
    assert "precursor_im" not in pl.read_parquet(paths["spectra_table"]).columns
