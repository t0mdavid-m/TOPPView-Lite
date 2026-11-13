"""The table of all identifications, and the TIC/BPC plots' own identifier."""

from unittest.mock import patch

import polars as pl
import pytest

from src.components.factory import (
    ALL_IDENTIFICATIONS_IDENTIFIER,
    CHROMATOGRAM_SPECTRUM_IDENTIFIER,
    create_all_identifications_table,
    create_chromatogram_components,
)


def test_tic_and_bpc_do_not_hold_the_spectrum_selection(tmp_path):
    # They hold MS1 spectra only; with ``spectrum`` Insight would clear every
    # MS2 spectrum selected elsewhere
    paths = {"chromatogram": tmp_path / "chromatogram.parquet", "component_cache": tmp_path}
    with patch("src.components.factory.SeriesPlot") as series:
        plots = create_chromatogram_components(paths, {"type": "none"}, "run")
    assert set(plots) == {"tic", "bpc"}
    for call in series.call_args_list:
        assert call.kwargs["interactivity"] == {CHROMATOGRAM_SPECTRUM_IDENTIFIER: "scan_id"}


@pytest.fixture
def identifications(tmp_path):
    path = tmp_path / "identifications.parquet"
    pl.DataFrame({
        "id_idx": [0, 1],
        "rt": [120.0, 1800.0],
        "precursor_mz": [500.25, 612.3],
        "sequence": ["PEPTIDEK", "ELVISK"],
        "sequence_display": ["PEPTIDEK", "ELVISK"],
        "charge": [2, 3],
        "score": [0.01, 0.2],
        "theoretical_mass": [998.5, 1833.9],
        "protein_accession": ["P1", "P2"],
        "scan_id": [42, -1],
    }).write_parquet(path)
    return path


def test_all_identifications_table(identifications, tmp_path):
    with patch("src.components.factory.Table") as table:
        create_all_identifications_table(str(identifications), str(tmp_path), "run")
    kwargs = table.call_args.kwargs
    assert kwargs["cache_id"] == "run_all_identifications_table"
    # Its own identifier: most spectra have no identification, so ``spectrum``
    # here would be cleared whenever such a spectrum is selected
    assert kwargs["interactivity"] == {ALL_IDENTIFICATIONS_IDENTIFIER: "id_idx"}
    assert kwargs["default_row"] == -1

    rows = kwargs["data"].collect()
    assert rows["rt_min"].to_list() == pytest.approx([2.0, 30.0])
    # An identification not linked to a spectrum has no scan
    assert rows["scan_id"].to_list() == [42, None]
    shown = {c["field"] for c in kwargs["column_definitions"]}
    assert shown <= set(rows.columns)
