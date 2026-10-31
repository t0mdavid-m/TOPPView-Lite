"""Unit tests for src/preprocessing/features.py (real pyOpenMS)."""

from pathlib import Path

import numpy as np
import polars as pl
import pyopenms as oms

from src.preprocessing.features import (
    extract_featurexml_to_parquet,
    feature_cache_is_valid,
    find_matching_featurexml,
    get_feature_cache_paths,
    link_features_to_ms1,
    load_features,
)


def _write_featurexml(path: Path) -> None:
    feature_map = oms.FeatureMap()
    for rt, mz, intensity, charge in [(600.0, 500.25, 1e6, 2), (1200.0, 700.5, 2e5, 3)]:
        feature = oms.Feature()
        feature.setRT(rt)
        feature.setMZ(mz)
        feature.setIntensity(intensity)
        feature.setCharge(charge)
        feature.setOverallQuality(0.9)
        hull = oms.ConvexHull2D()
        hull.setHullPoints(np.array([[rt - 6.0, mz - 0.01], [rt + 12.0, mz + 1.5]], dtype=np.float32))
        feature.setConvexHulls([hull])
        feature_map.push_back(feature)
    oms.FeatureXMLFile().store(str(path), feature_map)


def test_load_features_reads_centroid_and_hull_extent(tmp_path):
    path = tmp_path / "run.featureXML"
    _write_featurexml(path)

    df = load_features(path)

    assert df.height == 2
    first = df.row(0, named=True)
    assert first["rt"] == 10.0  # minutes
    assert first["mz"] == 500.25
    assert first["charge"] == 2
    assert (first["rt_start"], first["rt_end"]) == (9.9, 10.2)
    assert round(first["mz_end"] - first["mz_start"], 2) == 1.51


def test_link_features_to_nearest_ms1(sample_metadata_df):
    features = pl.DataFrame({"feature_id": [0, 1], "rt": [1.2, 2.9]})
    linked = link_features_to_ms1(features, sample_metadata_df)
    # MS1 scans are 1 (1.0 min) and 3 (3.0 min)
    assert linked["apex_scan_id"].to_list() == [1, 3]


def test_find_matching_featurexml_by_stem(tmp_path):
    feature_dir = tmp_path / "featureXML-files"
    feature_dir.mkdir()
    (feature_dir / "Run1.featureXML").write_text("")
    assert find_matching_featurexml(Path("run1.mzML"), tmp_path).name == "Run1.featureXML"
    assert find_matching_featurexml(Path("other.mzML"), tmp_path) is None


def test_extract_and_cache_validity(tmp_path, sample_metadata_df):
    path = tmp_path / "run.featureXML"
    _write_featurexml(path)
    metadata = tmp_path / "metadata.parquet"
    sample_metadata_df.write_parquet(metadata)
    paths = get_feature_cache_paths(tmp_path, path)

    assert not feature_cache_is_valid(path, paths)
    df = extract_featurexml_to_parquet(path, paths, metadata)
    assert feature_cache_is_valid(path, paths)
    assert "apex_scan_id" in df.columns
