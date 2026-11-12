"""Chromatograms stored in the mzML: extraction and the viewer's catalog."""

import os
from pathlib import Path

import numpy as np
import polars as pl
import pyopenms as oms
import pytest

from src.components.viewer_panels import ViewerContext, available_catalog
from src.preprocessing.extraction import get_cache_paths
from src.preprocessing.stored_chromatograms import (
    ensure_stored_chromatograms,
    stored_chromatogram_paths,
)


def _chromatogram(native_id, precursor_mz, product_mz, peptide, rt_seconds, intensity, kind):
    chrom = oms.MSChromatogram()
    chrom.setNativeID(native_id)
    chrom.setChromatogramType(kind)
    precursor = oms.Precursor()
    precursor.setMZ(precursor_mz)
    if peptide:
        precursor.setMetaValue("peptide_sequence", peptide)
    chrom.setPrecursor(precursor)
    product = oms.Product()
    product.setMZ(product_mz)
    chrom.setProduct(product)
    chrom.set_peaks((np.asarray(rt_seconds, dtype=float), np.asarray(intensity, dtype=float)))
    return chrom


SRM = oms.ChromatogramSettings.ChromatogramType.SELECTED_REACTION_MONITORING_CHROMATOGRAM
TIC = oms.ChromatogramSettings.ChromatogramType.TOTAL_ION_CURRENT_CHROMATOGRAM


@pytest.fixture
def srm_file(tmp_path):
    exp = oms.MSExperiment()
    exp.setChromatograms([
        _chromatogram("tr1", 500.25, 600.3, "PEPTIDEK", [120, 60, 180], [5, 10, 2], SRM),
        _chromatogram("tr2", 500.25, 700.4, "PEPTIDEK", [60, 120], [3, 4], SRM),
        _chromatogram("tr3", 610.8, 800.1, "ELVISLIVESK", [60, 120], [7, 8], SRM),
        _chromatogram("", 0.0, 0.0, "", [], [], TIC),
    ])
    path = tmp_path / "mzML-files" / "srm.mzML"
    path.parent.mkdir()
    oms.MzMLFile().store(str(path), exp)
    return path


def _paths(srm_file):
    return get_cache_paths(srm_file.parent.parent, srm_file)


def test_one_row_per_chromatogram_and_its_points(srm_file):
    paths = _paths(srm_file)
    assert ensure_stored_chromatograms(srm_file, paths) == (4, True)
    files = stored_chromatogram_paths(paths)
    index = pl.read_parquet(files["index"])
    assert index["native_id"].to_list() == ["tr1", "tr2", "tr3", "chromatogram 4"]
    assert index["type"].to_list() == ["SRM", "SRM", "SRM", "TIC"]
    assert index["peptide"].to_list() == ["PEPTIDEK", "PEPTIDEK", "ELVISLIVESK", ""]
    assert index["num_points"].to_list() == [3, 2, 2, 0]
    # The transitions of one precursor share a group
    assert index["group_id"].to_list() == [0, 0, 1, 2]
    first = index.row(0, named=True)
    assert first["product_mz"] == pytest.approx(600.3)
    assert (first["rt_start"], first["rt_end"]) == pytest.approx((1.0, 3.0))
    assert first["max_intensity"] == pytest.approx(10)

    points = pl.read_parquet(files["points"])
    tr1 = points.filter(pl.col("chrom_index") == 0)
    # RT in minutes, in RT order
    assert tr1["retention_time"].to_list() == pytest.approx([1.0, 2.0, 3.0])
    assert tr1["intensity"].to_list() == pytest.approx([10, 5, 2])
    assert set(points["native_id"]) == {"tr1", "tr2", "tr3"}


def test_files_are_written_again_only_for_a_newer_mzml(srm_file):
    paths = _paths(srm_file)
    ensure_stored_chromatograms(srm_file, paths)
    stale = Path(paths["component_cache"], "srm_stored_chromatogram_table")
    stale.mkdir(parents=True)
    assert ensure_stored_chromatograms(srm_file, paths) == (4, False)
    assert stale.exists()

    future = srm_file.stat().st_mtime + 10
    os.utime(srm_file, (future, future))
    assert ensure_stored_chromatograms(srm_file, paths) == (4, True)
    # Insight caches built from the old files are gone
    assert not stale.exists()


def test_a_file_without_chromatograms(tmp_path):
    exp = oms.MSExperiment()
    spectrum = oms.MSSpectrum()
    spectrum.set_peaks(([100.0], [1.0]))
    exp.addSpectrum(spectrum)
    path = tmp_path / "plain.mzML"
    oms.MzMLFile().store(str(path), exp)
    paths = get_cache_paths(tmp_path, path)
    assert ensure_stored_chromatograms(path, paths) == (0, True)
    assert pl.read_parquet(stored_chromatogram_paths(paths)["points"]).height == 0


def _ctx(num_spectra, num_chromatograms):
    return ViewerContext(
        workspace=Path("."), selected_file=Path("run.mzML"), paths={}, im_info={},
        file_id="run", cache_path="", state_manager=None, ms_components={},
        num_spectra=num_spectra, num_chromatograms=num_chromatograms,
    )


def test_the_catalog_follows_what_the_file_holds():
    spectra_only = {c["name"] for c in available_catalog(_ctx(10, 0))}
    assert "peak_map" in spectra_only and "stored_chromatograms" not in spectra_only

    both = {c["name"] for c in available_catalog(_ctx(10, 2))}
    assert {"peak_map", "stored_chromatograms"} <= both

    # An SRM file has chromatograms only: no spectrum panels
    srm = {c["name"] for c in available_catalog(_ctx(0, 3))}
    assert srm == {"stored_chromatograms", "tools"}


def test_the_plots_follow_the_table():
    from unittest.mock import patch

    from src.components.factory import create_stored_chromatogram_components

    with patch("src.components.factory.Table") as table, patch(
        "src.components.factory.SeriesPlot"
    ) as series:
        components = create_stored_chromatogram_components("index.pq", "points.pq", "cache", "srm")
    assert set(components) == {"table", "selected", "group"}
    assert table.call_args.kwargs["interactivity"] == {
        "stored_chromatogram": "chrom_index",
        "chromatogram_group": "group_id",
    }
    plots = {call.kwargs["cache_id"]: call.kwargs for call in series.call_args_list}
    assert plots["srm_stored_chromatogram_selected"]["filters"] == {
        "stored_chromatogram": "chrom_index"
    }
    group = plots["srm_stored_chromatogram_group"]
    assert group["filters"] == {"chromatogram_group": "group_id"}
    # One line per chromatogram of the group
    assert group["series_column"] == "native_id"
