"""Mass spectrometry imaging: imzML to mzML, pixel positions, ion images."""

import numpy as np
import polars as pl
import pyopenms as oms
import pytest

from src.preprocessing import create_component_inputs, extract_mzml_to_parquet, get_cache_paths
from src.preprocessing.imaging import (
    convert_imzml_to_mzml,
    convert_pairs,
    create_ion_image,
    find_pairs,
    has_pixels,
    imzml_dir,
    ion_image_name,
    peak_mz,
)

WIDTH, HEIGHT = 4, 3


def _write_imzml(path):
    """4 x 3 pixels; m/z 500.25 only in the left column, 700.1 everywhere."""
    exp = oms.MSExperiment()
    for y in range(1, HEIGHT + 1):
        for x in range(1, WIDTH + 1):
            mz = [300.0, 500.25, 700.1]
            intensity = [0.0, 100.0 if x == 1 else 0.0, 10.0 * x]
            spectrum = oms.MSSpectrum()
            spectrum.set_peaks((np.asarray(mz), np.asarray(intensity)))
            spectrum.setMSLevel(1)
            spectrum.setMetaValue("imzml:x", x)
            spectrum.setMetaValue("imzml:y", y)
            spectrum.setMetaValue("imzml:z", 1)
            exp.addSpectrum(spectrum)
    imaging = oms.MSImagingExperiment()
    imaging.setMSExperiment(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    oms.ImzMLFile().store(str(path), imaging)
    return path


@pytest.fixture
def imaging_file(tmp_path):
    imzml = _write_imzml(imzml_dir(tmp_path) / "tissue.imzML")
    mzml = tmp_path / "mzML-files" / "tissue.mzML"
    assert convert_imzml_to_mzml(imzml, mzml) == WIDTH * HEIGHT
    paths = get_cache_paths(tmp_path, mzml)
    extract_mzml_to_parquet(mzml, paths)
    create_component_inputs(paths)
    return mzml, paths


def test_pixels_become_spectra(imaging_file):
    mzml, paths = imaging_file
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml), exp)
    assert exp.size() == WIDTH * HEIGHT
    assert [s.getRT() for s in exp] == [float(i) for i in range(WIDTH * HEIGHT)]
    assert all(s.getMSLevel() == 1 for s in exp)
    # zero-intensity peaks are dropped
    assert exp[0].size() == 2 and exp[1].size() == 1

    metadata = pl.read_parquet(paths["metadata"]).sort("scan_id")
    assert metadata["pixel_x"].to_list()[:5] == [1, 2, 3, 4, 1]
    assert metadata["pixel_y"].to_list()[:5] == [1, 1, 1, 1, 2]
    assert has_pixels(paths)


def test_tic_and_ion_images(imaging_file):
    _, paths = imaging_file
    tic = pl.read_parquet(create_ion_image(paths, None)).sort("scan_id")
    assert tic.height == WIDTH * HEIGHT
    assert tic["intensity"].to_list()[:4] == pytest.approx([110.0, 20.0, 30.0, 40.0])

    ion = pl.read_parquet(create_ion_image(paths, 500.25, 10.0)).sort("scan_id")
    # every pixel listed, those without the ion with 0
    assert ion["intensity"].to_list()[:4] == pytest.approx([100.0, 0.0, 0.0, 0.0])
    assert ion.filter(pl.col("pixel_x") == 1)["intensity"].to_list() == pytest.approx([100.0] * HEIGHT)

    # outside the tolerance
    far = pl.read_parquet(create_ion_image(paths, 500.30, 10.0))
    assert far["intensity"].sum() == 0


def test_image_names_differ_per_window():
    assert ion_image_name(None, 10) == "tic"
    assert ion_image_name(500.25, 10) != ion_image_name(500.25, 20)
    assert "." not in ion_image_name(500.25, 2.5)


def test_peak_mz(imaging_file):
    _, paths = imaging_file
    peaks = pl.read_parquet(paths["peaks"])
    row = peaks.filter(pl.col("mass") > 600).row(0, named=True)
    assert peak_mz(paths, row["peak_id"]) == pytest.approx(700.1)
    assert peak_mz(paths, None) is None
    assert peak_mz(paths, 10**9) is None


def test_ordinary_files_have_no_pixels(tmp_path):
    exp = oms.MSExperiment()
    spectrum = oms.MSSpectrum()
    spectrum.set_peaks((np.asarray([400.0]), np.asarray([1.0])))
    spectrum.setMSLevel(1)
    exp.addSpectrum(spectrum)
    mzml = tmp_path / "mzML-files" / "plain.mzML"
    mzml.parent.mkdir()
    oms.MzMLFile().store(str(mzml), exp)
    paths = get_cache_paths(tmp_path, mzml)
    extract_mzml_to_parquet(mzml, paths)
    assert "pixel_x" not in pl.read_parquet(paths["metadata"]).columns
    assert not has_pixels(paths)


def test_convert_pairs(tmp_path):
    folder = imzml_dir(tmp_path)
    _write_imzml(folder / "a.imzML")
    _write_imzml(folder / "b.imzML")
    (folder / "b.ibd").unlink()
    mzml_dir = tmp_path / "mzML-files"

    pairs = find_pairs(folder)
    assert pairs["a"]["ibd"] is not None and pairs["b"]["ibd"] is None

    messages = convert_pairs(tmp_path, mzml_dir)
    assert any(m.startswith("a: 12 pixels") for m in messages)
    assert any(m.startswith("b: waiting for its .ibd") for m in messages)
    assert (mzml_dir / "a.mzML").exists() and not (mzml_dir / "b.mzML").exists()

    # converted already: nothing to do for a
    assert not any(m.startswith("a:") for m in convert_pairs(tmp_path, mzml_dir))
