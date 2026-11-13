"""Chromatograms stored in the mzML file (SRM/MRM transitions, extracted ion
chromatograms, a TIC written by the instrument software).

They are read once into two parquet files next to the file's other component
inputs:

- ``stored_chromatograms.parquet``: one row per chromatogram (index, native ID,
  type, precursor and product m/z, peptide, number of points, RT range and
  highest intensity) and the ``group_id`` of its precursor. Chromatograms with
  the same precursor m/z and peptide, the transitions of one precursor, share
  a group.
- ``stored_chromatogram_points.parquet``: one row per point, RT in minutes like
  the rest of the viewer.
"""

import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
from pyopenms import MSExperiment, MzMLFile

# What ChromatogramSettings.getChromatogramType() means, in short
_TYPE_LABELS = {
    "MASS_CHROMATOGRAM": "Mass",
    "TOTAL_ION_CURRENT_CHROMATOGRAM": "TIC",
    "SELECTED_ION_CURRENT_CHROMATOGRAM": "SIC",
    "BASEPEAK_CHROMATOGRAM": "Base peak",
    "SELECTED_ION_MONITORING_CHROMATOGRAM": "SIM",
    "SELECTED_REACTION_MONITORING_CHROMATOGRAM": "SRM",
    "ELECTROMAGNETIC_RADIATION_CHROMATOGRAM": "EMR",
    "ABSORPTION_CHROMATOGRAM": "Absorption",
    "EMISSION_CHROMATOGRAM": "Emission",
}

INDEX_SCHEMA = {
    "chrom_index": pl.Int32,
    "native_id": pl.Utf8,
    "type": pl.Utf8,
    "precursor_mz": pl.Float64,
    "product_mz": pl.Float64,
    "peptide": pl.Utf8,
    "num_points": pl.Int32,
    "rt_start": pl.Float32,
    "rt_end": pl.Float32,
    "max_intensity": pl.Float32,
    "group_id": pl.Int32,
}

POINTS_SCHEMA = {
    "chrom_index": pl.Int32,
    "group_id": pl.Int32,
    "native_id": pl.Utf8,
    "retention_time": pl.Float32,
    "intensity": pl.Float32,
}


def stored_chromatogram_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    """The two parquet files, next to the file's other component inputs."""
    components_dir = Path(paths["spectra_table"]).parent
    return {
        "index": components_dir / "stored_chromatograms.parquet",
        "points": components_dir / "stored_chromatogram_points.parquet",
    }


def _type_label(chromatogram) -> str:
    name = str(chromatogram.getChromatogramType()).rsplit(".", 1)[-1]
    return _TYPE_LABELS.get(name, "Other")


def _peptide(precursor) -> str:
    for key in ("peptide_sequence", "PeptideRef"):
        if precursor.metaValueExists(key):
            return str(precursor.getMetaValue(key))
    return ""


def load_chromatograms(mzml_path: Path) -> MSExperiment:
    """The file's chromatograms, without reading its spectra."""
    exp = MSExperiment()
    handler = MzMLFile()
    options = handler.getOptions()
    options.setMSLevels([99])  # no spectrum has this level
    handler.setOptions(options)
    handler.load(str(Path(mzml_path).resolve()), exp)
    return exp


def chromatogram_frames(exp: MSExperiment) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The index and points frames of the experiment's chromatograms."""
    rows = []
    points = []
    groups: Dict[tuple, int] = {}
    for index, chromatogram in enumerate(exp.getChromatograms()):
        rt, intensity = chromatogram.get_peaks()
        rt = np.asarray(rt, dtype=np.float64) / 60.0
        intensity = np.asarray(intensity, dtype=np.float32)
        precursor = chromatogram.getPrecursor()
        precursor_mz = float(precursor.getMZ())
        peptide = _peptide(precursor)
        group = groups.setdefault((round(precursor_mz, 4), peptide), len(groups))
        native_id = chromatogram.getNativeID() or f"chromatogram {index + 1}"
        rows.append({
            "chrom_index": index,
            "native_id": native_id,
            "type": _type_label(chromatogram),
            "precursor_mz": precursor_mz,
            "product_mz": float(chromatogram.getProduct().getMZ()),
            "peptide": peptide,
            "num_points": len(rt),
            "rt_start": float(rt.min()) if len(rt) else None,
            "rt_end": float(rt.max()) if len(rt) else None,
            "max_intensity": float(intensity.max()) if len(intensity) else None,
            "group_id": group,
        })
        if len(rt):
            order = np.argsort(rt, kind="stable")
            points.append(pl.DataFrame({
                "chrom_index": np.full(len(rt), index, dtype=np.int32),
                "group_id": np.full(len(rt), group, dtype=np.int32),
                "native_id": [native_id] * len(rt),
                "retention_time": rt[order].astype(np.float32),
                "intensity": intensity[order],
            }, schema=POINTS_SCHEMA))
    index_df = pl.DataFrame(rows, schema=INDEX_SCHEMA)
    points_df = pl.concat(points) if points else pl.DataFrame(schema=POINTS_SCHEMA)
    return index_df, points_df


def stored_chromatograms_are_current(mzml_path: Path, paths: Dict[str, Path]) -> bool:
    files = stored_chromatogram_paths(paths).values()
    mzml_mtime = Path(mzml_path).stat().st_mtime
    return all(f.exists() and f.stat().st_mtime > mzml_mtime for f in files)


def ensure_stored_chromatograms(mzml_path: Path, paths: Dict[str, Path]) -> tuple[int, bool]:
    """Write the two parquet files unless they are newer than the mzML.

    Returns how many chromatograms the file stores and whether the files were
    written now. Writing them removes the Insight caches built from older ones.
    """
    files = stored_chromatogram_paths(paths)
    if stored_chromatograms_are_current(mzml_path, paths):
        count = pl.scan_parquet(files["index"]).select(pl.len()).collect().item()
        return count, False
    index_df, points_df = chromatogram_frames(load_chromatograms(mzml_path))
    files["index"].parent.mkdir(parents=True, exist_ok=True)
    points_df.write_parquet(files["points"])
    index_df.write_parquet(files["index"])
    # Insight reuses a cache with the same id and configuration even when its
    # data changed, so drop the caches built from the old files
    component_cache = Path(paths["component_cache"])
    if component_cache.is_dir():
        for cache in component_cache.glob("*stored_chromatogram_*"):
            shutil.rmtree(cache, ignore_errors=True)
    return index_df.height, True
