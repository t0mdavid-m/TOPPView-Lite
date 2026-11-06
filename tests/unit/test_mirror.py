"""Tests for the mirror view data (experimental vs. theoretical spectrum)."""

import os
import time

import numpy as np
import polars as pl

from src.preprocessing.mirror import (
    build_mirror_data,
    ensure_mirror_input,
    get_mirror_paths,
    match_ions,
    mirror_cache_is_valid,
    theoretical_ions,
    _make_generator,
)


def test_match_ions_da_and_ppm():
    exp = np.array([100.0, 200.0, 300.0])
    theo = np.array([100.02, 250.0, 299.9])
    assert match_ions(exp, theo, 0.05, False).tolist() == [0, -1, -1]
    assert match_ions(exp, theo, 0.2, False).tolist() == [0, -1, 2]
    # 10 ppm of 100 Da is 0.001
    assert match_ions(exp, np.array([100.0005, 100.002]), 10, True).tolist() == [0, -1]


def test_match_ions_empty():
    assert match_ions(np.empty(0), np.array([1.0]), 0.1, False).tolist() == [-1]
    assert match_ions(np.array([1.0]), np.empty(0), 0.1, False).tolist() == []


def test_theoretical_ions_names_and_charges():
    tsg = _make_generator()
    mz, names = theoretical_ions(tsg, "PEPTIDEK", 2)
    # b2..b7 + y1..y7 (no b1 by default), singly charged only for a 2+ precursor
    assert len(mz) == 13
    assert all(name.endswith("+") and not name.endswith("++") for name in names)
    assert {n[0] for n in names} == {"b", "y"}

    mz3, names3 = theoretical_ions(tsg, "PEPTIDEK", 3)
    assert len(mz3) == 26
    assert any(n.endswith("++") for n in names3)


def test_theoretical_ions_bad_sequence():
    tsg = _make_generator()
    mz, names = theoretical_ions(tsg, "not a peptide!", 2)
    assert len(mz) == 0 and names == []


def _example():
    tsg = _make_generator()
    theo_mz, _ = theoretical_ions(tsg, "PEPTIDEK", 2)
    exp_mz = np.sort(np.concatenate([theo_mz[:3] + 0.01, [555.5]]))
    peaks = pl.DataFrame(
        {
            "scan_id": [5] * len(exp_mz),
            "mass": exp_mz.astype(np.float32),
            "intensity": np.arange(1, len(exp_mz) + 1, dtype=np.float32) * 10,
        }
    )
    ids = pl.DataFrame(
        {
            "id_idx": [0, 1],
            "sequence": ["PEPTIDEK", "PEPTIDEK"],
            "charge": [2, 2],
            "scan_id": [5, -1],  # second ID is not linked to a spectrum
        }
    )
    return ids, peaks


def test_build_mirror_data_halves_and_summary():
    ids, peaks = _example()
    mirror, summary = build_mirror_data(ids, peaks, tolerance=0.05, tolerance_ppm=False)

    top = mirror.filter(pl.col("mirror_top_id") == 0)
    bottom = mirror.filter(pl.col("mirror_bottom_id") == 0)
    assert top.height == 4
    assert bottom.height == 13
    assert (top["mirror_bottom_id"] == -1).all()
    assert (bottom["mirror_top_id"] == -1).all()
    assert mirror["mirror_peak_id"].n_unique() == mirror.height

    assert top["matched"].sum() == 3
    assert bottom["matched"].sum() == 3
    assert (top.filter(~pl.col("matched"))["label"] == "").all()
    assert (bottom.filter(pl.col("matched"))["label"] != "").all()
    # Theoretical sticks are drawn at the experimental base peak height
    assert bottom["intensity"].unique().to_list() == [top["intensity"].max()]

    assert summary.to_dicts() == [
        {"id_idx": 0, "matched_ions": 3, "total_ions": 13, "coverage": 300 / 13}
    ]


def test_build_mirror_data_without_linked_ids():
    ids, peaks = _example()
    mirror, summary = build_mirror_data(ids.with_columns(pl.lit(-1).alias("scan_id")), peaks)
    assert mirror.height == 0 and summary.height == 0
    assert "mirror_top_id" in mirror.columns


def test_ensure_mirror_input_caches(tmp_path):
    ids, peaks = _example()
    components = tmp_path / "components"
    components.mkdir()
    paths = {"spectrum_plot": components / "spectrum_plot.parquet"}
    peaks.write_parquet(paths["spectrum_plot"])
    id_paths = {"identifications": tmp_path / "identifications.parquet"}
    ids.write_parquet(id_paths["identifications"])

    mirror_paths, rebuilt = ensure_mirror_input(paths, id_paths)
    assert rebuilt
    assert mirror_paths == get_mirror_paths(paths)
    assert mirror_cache_is_valid(mirror_paths, id_paths["identifications"])

    _, rebuilt = ensure_mirror_input(paths, id_paths)
    assert not rebuilt

    # Newer identifications invalidate the cache
    future = time.time() + 10
    os.utime(id_paths["identifications"], (future, future))
    _, rebuilt = ensure_mirror_input(paths, id_paths)
    assert rebuilt
