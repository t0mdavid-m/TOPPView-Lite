"""featureXML support: extract detected features to parquet for the viewer.

A featureXML is matched to an mzML by file name (``sample.mzML`` ↔
``sample.featureXML`` in the workspace's ``featureXML-files`` folder), like
idXML files. Each feature becomes one row with its centroid, intensity,
charge, quality and RT/m/z extent (bounding box of its convex hulls), plus the
MS1 spectrum closest to its apex so a click in the features table can show
that spectrum.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from pyopenms import FeatureMap, FeatureXMLFile

FEATURE_DIR_NAME = "featureXML-files"


def find_matching_featurexml(mzml_path: Path, workspace: Path) -> Optional[Path]:
    """Return the featureXML with the same stem as ``mzml_path``, if any."""
    feature_dir = Path(workspace) / FEATURE_DIR_NAME
    if not feature_dir.exists():
        return None
    stem = Path(mzml_path).stem.lower()
    for candidate in sorted(feature_dir.iterdir()):
        if candidate.suffix.lower() == ".featurexml" and candidate.stem.lower() == stem:
            return candidate
    return None


def get_feature_cache_paths(workspace: Path, featurexml_path: Path) -> dict:
    cache_dir = Path(workspace) / ".cache" / "features" / Path(featurexml_path).stem
    return {
        "features": cache_dir / "features.parquet",
        "feature_info": cache_dir / "feature_info.json",
    }


def feature_cache_is_valid(featurexml_path: Path, feature_paths: dict) -> bool:
    target = feature_paths["features"]
    return target.exists() and target.stat().st_mtime >= Path(featurexml_path).stat().st_mtime


def load_features(featurexml_path: Path) -> pl.DataFrame:
    """Read a featureXML into one row per feature (RT in minutes)."""
    feature_map = FeatureMap()
    FeatureXMLFile().load(str(Path(featurexml_path).resolve()), feature_map)

    rows = {
        "feature_id": [], "rt": [], "mz": [], "intensity": [], "charge": [],
        "quality": [], "rt_start": [], "rt_end": [], "mz_start": [], "mz_end": [],
    }
    for index, feature in enumerate(feature_map):
        rt, mz = feature.getRT(), feature.getMZ()
        rt_lo, rt_hi, mz_lo, mz_hi = rt, rt, mz, mz
        hulls = feature.getConvexHulls()
        points = [hull.getHullPoints() for hull in hulls]
        points = [p for p in points if len(p)]
        if points:
            stacked = np.vstack(points)
            rt_lo, mz_lo = stacked.min(axis=0)
            rt_hi, mz_hi = stacked.max(axis=0)
        rows["feature_id"].append(index)
        rows["rt"].append(rt / 60.0)
        rows["mz"].append(mz)
        rows["intensity"].append(feature.getIntensity())
        rows["charge"].append(feature.getCharge())
        rows["quality"].append(feature.getOverallQuality())
        rows["rt_start"].append(rt_lo / 60.0)
        rows["rt_end"].append(rt_hi / 60.0)
        rows["mz_start"].append(mz_lo)
        rows["mz_end"].append(mz_hi)

    return pl.DataFrame(
        rows,
        schema={
            "feature_id": pl.Int64, "rt": pl.Float64, "mz": pl.Float64,
            "intensity": pl.Float64, "charge": pl.Int32, "quality": pl.Float64,
            "rt_start": pl.Float64, "rt_end": pl.Float64,
            "mz_start": pl.Float64, "mz_end": pl.Float64,
        },
    )


def link_features_to_ms1(features: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Add ``apex_scan_id``: the MS1 spectrum closest in RT to each feature."""
    ms1 = (
        metadata.filter(pl.col("ms_level") == 1)
        .select(
            pl.col("scan_id").alias("apex_scan_id"),
            pl.col("retention_time").cast(pl.Float64).alias("rt"),
        )
        .sort("rt")
    )
    if ms1.is_empty() or features.is_empty():
        return features.with_columns(pl.lit(-1, dtype=pl.Int32).alias("apex_scan_id"))
    return (
        features.sort("rt")
        .join_asof(ms1, on="rt", strategy="nearest")
        .with_columns(pl.col("apex_scan_id").cast(pl.Int32))
        .sort("feature_id")
    )


def extract_featurexml_to_parquet(
    featurexml_path: Path, feature_paths: dict, metadata_path: Path
) -> pl.DataFrame:
    """Write the features of ``featurexml_path`` (linked to its mzML) to parquet."""
    features = load_features(featurexml_path)
    features = link_features_to_ms1(features, pl.read_parquet(metadata_path))
    feature_paths["features"].parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(feature_paths["features"])
    with open(feature_paths["feature_info"], "w") as f:
        json.dump({"num_features": features.height, "source": Path(featurexml_path).name}, f)
    return features
