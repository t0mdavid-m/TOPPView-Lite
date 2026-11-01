"""Tests for the Tools page operations and the local console launcher."""

import numpy as np
import pyopenms as oms
import pytest

from src.console import SNIPPETS, run_script
from src.operations import (
    OPERATIONS,
    apply_overrides,
    output_path,
    param_entries,
    run_operation,
    run_operation_isolated,
)


@pytest.fixture
def lcms_mzml(tmp_path):
    """Centroided MS1 run with two co-eluting isotope traces (z=1) and noise."""
    rng = np.random.default_rng(0)
    exp = oms.MSExperiment()
    for i in range(60):
        rt = 100.0 + i * 1.0
        elution = 1e6 * np.exp(-((rt - 130.0) ** 2) / (2 * 4.0**2))
        mz = np.array([300.0, 500.0, 501.00336, 700.0 + rng.random()])
        inten = np.array([1e3, elution, elution * 0.3, 5e2])
        spec = oms.MSSpectrum()
        spec.setRT(rt)
        spec.setMSLevel(1)
        spec.set_peaks((mz, inten))
        exp.addSpectrum(spec)
    path = tmp_path / "mzML-files" / "run.mzML"
    path.parent.mkdir()
    oms.MzMLFile().store(str(path), exp)
    return path


def test_param_entries_types_and_tags():
    entries = {e["key"]: e for e in param_entries(oms.PeakPickerHiRes().getParameters())}
    assert entries["signal_to_noise"]["type"] == "float"
    assert entries["signal_to_noise"]["advanced"] is False
    assert entries["spacing_difference"]["advanced"] is True
    assert entries["report_FWHM"]["type"] == "bool"
    assert entries["report_FWHM_unit"]["choices"] == ["relative", "absolute"]
    assert entries["ms_levels"]["type"] == "list"


def test_apply_overrides_coerces_and_rejects_unknown():
    base = oms.PeakPickerHiRes().getParameters()
    p = apply_overrides(base, {"signal_to_noise": 2, "report_FWHM": True, "ms_levels": "1, 2"})
    assert p.getValue("signal_to_noise") == 2.0
    assert p.getValue("report_FWHM") == "true"
    assert list(p.getValue("ms_levels")) == [1, 2]
    with pytest.raises(ValueError):
        apply_overrides(base, {"no_such_param": 1})
    with pytest.raises(ValueError):
        apply_overrides(base, {"report_FWHM_unit": "furlongs"})


def test_operation_params_route_by_prefix():
    op = OPERATIONS["feature_finder_metabo"]
    params = op.params({"mtd:mass_error_ppm": 5.0, "ffm:charge_upper_bound": 2})
    assert params["mtd:"].getValue("mass_error_ppm") == 5.0
    assert params["ffm:"].getValue("charge_upper_bound") == 2
    with pytest.raises(ValueError, match="Unknown"):
        op.params({"mass_error_ppm": 5.0})  # missing section prefix


def test_output_path_validation(tmp_path):
    op = OPERATIONS["feature_finder_metabo"]
    assert output_path(tmp_path, op, "run") == tmp_path / "featureXML-files" / "run.featureXML"
    for bad in ["../evil", "a/b", ".hidden", "", "x;rm"]:
        with pytest.raises(ValueError):
            output_path(tmp_path, op, bad)


def test_feature_finder_metabo_finds_the_isotope_pattern(lcms_mzml, tmp_path):
    out = tmp_path / "featureXML-files" / "run.featureXML"
    summary = run_operation("feature_finder_metabo", lcms_mzml, out, {"epd:width_filtering": "off"})
    assert "features" in summary
    fmap = oms.FeatureMap()
    oms.FeatureXMLFile().load(str(out), fmap)
    mzs = [f.getMZ() for f in fmap]
    assert any(abs(mz - 500.0) < 0.01 for mz in mzs)


def test_mass_traces_and_peak_picking(lcms_mzml, tmp_path):
    out = tmp_path / "traces.featureXML"
    assert "mass traces" in run_operation("mass_trace_detection", lcms_mzml, out, {})
    fmap = oms.FeatureMap()
    oms.FeatureXMLFile().load(str(out), fmap)
    assert fmap.size() >= 2

    picked = tmp_path / "picked.mzML"
    assert "Centroided 60" in run_operation("peak_picking", lcms_mzml, picked, {})
    assert picked.exists()


def test_isolated_run_reports_errors(lcms_mzml, tmp_path):
    with pytest.raises(RuntimeError, match="Unknown"):
        run_operation_isolated("peak_picking", lcms_mzml, tmp_path / "x.mzML", {"bogus": 1}, timeout=120)


def test_console_runs_in_separate_process(lcms_mzml):
    workspace = lcms_mzml.parent.parent
    code = "import os\nprint(exp.getNrSpectra(), spectrum_index, os.getpid())\nsave_mzml(exp, '../escape')"
    result = run_script(code, workspace, lcms_mzml, scan_id=3, timeout=120)
    assert result.returncode == 0, result.output
    first = result.output.splitlines()[0].split()
    assert first[:2] == ["60", "2"]
    assert int(first[2]) != __import__("os").getpid()
    # save helpers keep files inside the workspace folder
    assert (workspace / "mzML-files" / "escape.mzML").exists()
    assert not (workspace / "escape.mzML").exists()


def test_console_reports_errors_and_timeouts(lcms_mzml):
    workspace = lcms_mzml.parent.parent
    result = run_script("raise ValueError('boom')", workspace, lcms_mzml, timeout=120)
    assert result.returncode == 1 and "boom" in result.output
    result = run_script("while True: pass", workspace, lcms_mzml, timeout=5)
    assert result.timed_out


@pytest.mark.parametrize("name", list(SNIPPETS))
def test_snippets_run(name, lcms_mzml):
    workspace = lcms_mzml.parent.parent
    result = run_script(SNIPPETS[name], workspace, lcms_mzml, scan_id=1, timeout=120)
    assert result.returncode == 0, result.output


def test_console_loads_features_and_ids(lcms_mzml, tmp_path):
    workspace = lcms_mzml.parent.parent
    fmap = oms.FeatureMap()
    f = oms.Feature()
    f.setMZ(500.0)
    fmap.push_back(f)
    fmap.ensureUniqueId()
    fxml = tmp_path / "run.featureXML"
    oms.FeatureXMLFile().store(str(fxml), fmap)

    pep = oms.PeptideIdentification()
    pep.setRT(130.0)
    pep.setMZ(500.0)
    hit = oms.PeptideHit()
    hit.setSequence(oms.AASequence.fromString("PEPTIDEK"))
    pep.setHits([hit])
    peps = oms.PeptideIdentificationList() if hasattr(oms, "PeptideIdentificationList") else []
    peps.push_back(pep) if hasattr(peps, "push_back") else peps.append(pep)
    idxml = tmp_path / "run.idXML"
    oms.IdXMLFile().store(str(idxml), [oms.ProteinIdentification()], peps)

    code = "print(feature_map.size(), len(peptide_ids), peptide_ids[0].getHits()[0].getSequence().toString())"
    result = run_script(code, workspace, lcms_mzml, featurexml_path=fxml, idxml_path=idxml, timeout=120)
    assert result.returncode == 0, result.output
    assert result.output.splitlines()[-1] == "1 1 PEPTIDEK"


def test_console_accepts_relative_paths(lcms_mzml, monkeypatch):
    workspace = lcms_mzml.parent.parent
    monkeypatch.chdir(workspace.parent)
    rel_ws = workspace.relative_to(workspace.parent)
    result = run_script("print(exp.getNrSpectra())", rel_ws, rel_ws / "mzML-files" / "run.mzML", timeout=120)
    assert result.returncode == 0, result.output
    assert result.output.splitlines()[-1] == "60"
