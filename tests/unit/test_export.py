"""Tests for the filtered mzML export."""

import shutil
from types import SimpleNamespace

import numpy as np
import pyopenms as oms
import pytest

from src import export
from src.export import (
    export_filtered_mzml,
    filefilter_command,
    filtered_output_path,
)


@pytest.fixture
def small_mzml(tmp_path):
    """Three MS1 and three MS2 spectra at RT 10..60 s, peaks at m/z 100..500."""
    exp = oms.MSExperiment()
    for i in range(6):
        spec = oms.MSSpectrum()
        spec.setRT(10.0 * (i + 1))
        spec.setMSLevel(1 if i % 2 == 0 else 2)
        spec.set_peaks((np.array([100.0, 200.0, 300.0, 400.0, 500.0]), np.full(5, 10.0)))
        exp.addSpectrum(spec)
    path = tmp_path / "small.mzML"
    oms.MzMLFile().store(str(path), exp)
    return path


def _load(path):
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(path), exp)
    return exp.getSpectra()


def test_output_path(tmp_path):
    assert filtered_output_path(tmp_path, tmp_path / "x" / "run.mzML") == (
        tmp_path / "exports" / "run_filtered.mzML"
    )


def test_filefilter_command():
    cmd = filefilter_command("in.mzML", "out.mzML", (60.0, 120.5), (200.0, 900.0), [2, 1])
    assert cmd == [
        "FileFilter", "-in", "in.mzML", "-out", "out.mzML",
        "-rt", "60.0:120.5", "-mz", "200.0:900.0",
        "-peak_options:level", "1", "2",
    ]


def test_pyopenms_backend_filters_rt_mz_and_level(small_mzml, tmp_path):
    out = tmp_path / "out" / "f.mzML"
    backend = export_filtered_mzml(small_mzml, out, (15.0, 55.0), (150.0, 450.0), [2], use_topp=False)
    assert backend == "pyOpenMS"
    spectra = _load(out)
    assert [s.getRT() for s in spectra] == [20.0, 40.0]
    assert all(s.getMSLevel() == 2 for s in spectra)
    assert all(s.get_peaks()[0].tolist() == [200.0, 300.0, 400.0] for s in spectra)


def test_validation(small_mzml, tmp_path):
    with pytest.raises(ValueError):
        export_filtered_mzml(small_mzml, tmp_path / "o.mzML", (0, 100), (0, 1000), [], use_topp=False)
    with pytest.raises(ValueError):
        export_filtered_mzml(small_mzml, tmp_path / "o.mzML", (100, 0), (0, 1000), [1], use_topp=False)


def test_topp_backend_reports_failure(monkeypatch, small_mzml, tmp_path):
    monkeypatch.setattr(export.shutil, "which", lambda name: "/opt/FileFilter")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="line 1\nError: boom")

    monkeypatch.setattr(export.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        export_filtered_mzml(small_mzml, tmp_path / "o.mzML", (0, 100), (0, 1000), [1])
    assert calls[0][0] == "/opt/FileFilter"


@pytest.mark.skipif(shutil.which("FileFilter") is None, reason="FileFilter not on PATH")
def test_filefilter_matches_pyopenms(small_mzml, tmp_path):
    args = ((15.0, 55.0), (150.0, 450.0), [1, 2])
    assert export_filtered_mzml(small_mzml, tmp_path / "t.mzML", *args, use_topp=True) == "FileFilter"
    export_filtered_mzml(small_mzml, tmp_path / "p.mzML", *args, use_topp=False)
    topp, py = _load(tmp_path / "t.mzML"), _load(tmp_path / "p.mzML")
    assert [s.getRT() for s in topp] == [s.getRT() for s in py]
    assert [s.get_peaks()[0].tolist() for s in topp] == [s.get_peaks()[0].tolist() for s in py]


# --- Coupling to the viewer's tables ------------------------------------------

import polars as pl

from src.export import export_table_selection, filefilter_equivalent, peak_window


def _metadata():
    """Matches small_mzml: scans 1..6, RT 10..60 s, MS1 on odd scans."""
    return pl.DataFrame({
        "scan_id": [1, 2, 3, 4, 5, 6],
        "retention_time": [r / 60.0 for r in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)],
        "ms_level": [1, 2, 1, 2, 1, 2],
    })


def test_peak_window_reads_mass_and_intensity_filters():
    window = peak_window([
        {"field": "mass", "type": ">=", "value": 150},
        {"field": "mass", "type": "<=", "value": 450},
        {"field": "intensity", "type": ">=", "value": 5},
        {"field": "peak_id", "type": ">=", "value": 3},
    ])
    assert window == {"mz": (150.0, 450.0), "int": (5.0, None)}
    assert peak_window([]) == {"mz": None, "int": None}


def test_filefilter_command_open_bounds():
    cmd = filefilter_command("a", "b", mz_range=(None, 400.0), int_range=(5.0, None))
    assert cmd == ["FileFilter", "-in", "a", "-out", "b", "-mz", ":400.0", "-int", "5.0:"]


def test_equivalent_for_level_and_rt_selection():
    meta = _metadata()
    assert filefilter_equivalent(meta, [2, 4, 6]) == (None, [2])
    rt, levels = filefilter_equivalent(meta, [2, 3, 4])
    assert levels == [1, 2]
    assert rt[0] < 20.0 < 40.0 < rt[1] and rt[0] > 10.0 and rt[1] < 50.0
    rt, levels = filefilter_equivalent(meta, [4, 6])
    assert levels == [2] and rt[0] < 40.0 and rt[1] > 60.0


def test_not_equivalent_when_selection_has_gaps():
    # Scans 2 and 6 but not 4: no RT range over MS2 picks exactly these.
    assert filefilter_equivalent(_metadata(), [2, 6]) is None


def test_table_selection_with_pyopenms(small_mzml, tmp_path):
    out = tmp_path / "sel.mzML"
    backend = export_table_selection(
        small_mzml, out, _metadata(), [2, 6],
        [{"field": "mass", "type": "<=", "value": 250}], use_topp=False,
    )
    assert backend == "pyOpenMS"
    spectra = _load(out)
    assert [s.getRT() for s in spectra] == [20.0, 60.0]
    assert all(s.get_peaks()[0].tolist() == [100.0, 200.0] for s in spectra)


def test_table_selection_uses_filefilter_when_expressible(monkeypatch, small_mzml, tmp_path):
    monkeypatch.setattr(export.shutil, "which", lambda name: "/opt/FileFilter")
    calls = []
    monkeypatch.setattr(
        export.subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    backend = export_table_selection(
        small_mzml, tmp_path / "o.mzML", _metadata(), [2, 4, 6],
        [{"field": "intensity", "type": ">=", "value": 5}],
    )
    assert backend == "FileFilter"
    cmd = calls[0]
    assert cmd[cmd.index("-peak_options:level") + 1:] == ["2"]
    assert cmd[cmd.index("-int") + 1] == "5.0:"
    assert "-rt" not in cmd and "-mz" not in cmd

    # A selection FileFilter cannot express falls back to pyOpenMS.
    calls.clear()
    backend = export_table_selection(small_mzml, tmp_path / "o2.mzML", _metadata(), [2, 6])
    assert backend == "pyOpenMS" and not calls


def test_table_selection_rejects_empty(small_mzml, tmp_path):
    with pytest.raises(ValueError, match="no spectra"):
        export_table_selection(small_mzml, tmp_path / "o.mzML", _metadata(), [], use_topp=False)


@pytest.mark.skipif(shutil.which("FileFilter") is None, reason="FileFilter not on PATH")
def test_filefilter_selection_matches_pyopenms(small_mzml, tmp_path):
    meta, peaks = _metadata(), [{"field": "mass", "type": ">=", "value": 250}]
    assert export_table_selection(small_mzml, tmp_path / "t.mzML", meta, [2, 3, 4], peaks, use_topp=True) == "FileFilter"
    export_filtered_mzml(small_mzml, tmp_path / "p.mzML", mz_range=(250.0, None), scan_ids=[2, 3, 4], use_topp=False)
    topp, py = _load(tmp_path / "t.mzML"), _load(tmp_path / "p.mzML")
    assert [s.getRT() for s in topp] == [s.getRT() for s in py] == [20.0, 30.0, 40.0]
    assert [s.get_peaks()[0].tolist() for s in topp] == [s.get_peaks()[0].tolist() for s in py]
