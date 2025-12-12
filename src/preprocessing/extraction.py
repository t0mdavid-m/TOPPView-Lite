"""mzML extraction to parquet files.

This module handles the first stage of preprocessing: extracting raw mzML data
into parquet files for efficient subsequent access.
"""

import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import polars as pl
from pyopenms import MSExperiment, MzMLFile


def get_cache_paths(workspace: Path, mzml_path: Path) -> dict:
    """Get all cache file paths for a given mzML file within a workspace.

    Cache structure:
    {workspace}/.cache/{file_stem}/
    ├── raw/
    │   ├── metadata.parquet
    │   ├── peaks.parquet
    │   └── im_info.json
    └── components/
        ├── spectra_table.parquet
        ├── peaks_table.parquet
        ├── spectrum_plot.parquet
        ├── heatmap_input.parquet
        ├── im_table.parquet
        └── component_cache/  (for openms_insight caches)
    """
    cache_dir = Path(workspace) / ".cache" / mzml_path.stem
    raw_dir = cache_dir / "raw"
    components_dir = cache_dir / "components"

    return {
        # Raw extraction
        "metadata": raw_dir / "metadata.parquet",
        "peaks": raw_dir / "peaks.parquet",
        "im_info": raw_dir / "im_info.json",
        # Component-specific inputs
        "spectra_table": components_dir / "spectra_table.parquet",
        "peaks_table": components_dir / "peaks_table.parquet",
        "spectrum_plot": components_dir / "spectrum_plot.parquet",
        "annotated_spectrum_plot": components_dir / "annotated_spectrum_plot.parquet",
        "heatmap_input": components_dir / "heatmap_input.parquet",
        "im_table": components_dir / "im_table.parquet",
        # Component cache directory (for Heatmap compression levels, etc.)
        "component_cache": components_dir / "component_cache",
    }


def raw_cache_is_valid(mzml_path: Path, paths: dict) -> bool:
    """Check if raw cache files exist and are newer than the mzML file."""
    metadata_pq = paths["metadata"]
    peaks_pq = paths["peaks"]
    im_info = paths["im_info"]

    if not metadata_pq.exists() or not peaks_pq.exists() or not im_info.exists():
        return False

    mzml_mtime = mzml_path.stat().st_mtime
    return (
        metadata_pq.stat().st_mtime > mzml_mtime
        and peaks_pq.stat().st_mtime > mzml_mtime
        and im_info.stat().st_mtime > mzml_mtime
    )


def get_faims_cv_from_spectrum(spectrum) -> Optional[float]:
    """Extract FAIMS compensation voltage from spectrum.

    Uses getDriftTime() which pyOpenMS uses for FAIMS CV values.
    The drift time unit indicates FAIMS (unit 3 = volt).
    """
    drift_time = spectrum.getDriftTime()
    if drift_time != 0.0 or spectrum.getDriftTimeUnit() == 3:
        return float(drift_time)

    # Fallback: check meta values (for other data formats)
    cv_names = [
        "FAIMS compensation voltage",
        "ion mobility drift time",
        "MS:1001581",
    ]
    for name in cv_names:
        if spectrum.metaValueExists(name):
            try:
                return float(spectrum.getMetaValue(name))
            except (ValueError, TypeError):
                pass

    return None


def detect_ion_mobility_type(exp: MSExperiment) -> tuple[str, str, list]:
    """Detect what type of ion mobility data is present in the experiment.

    Returns:
        (im_type, im_unit, cv_values_per_spectrum)
        im_type: 'faims', 'tims', or 'none'
        im_unit: Unit string (e.g., 'V' for FAIMS, 'Vs/cm2' for TIMS)
        cv_values: For FAIMS, list of CV values per spectrum index (None for no CV)
    """
    cv_values = []
    has_any_cv = False

    for spectrum in exp:
        cv = get_faims_cv_from_spectrum(spectrum)
        cv_values.append(cv)
        if cv is not None:
            has_any_cv = True

    if has_any_cv:
        return 'faims', 'V', cv_values

    # Check for TIMS ion mobility in FloatDataArrays
    im_array_names = [
        "ion mobility",
        "inverse reduced ion mobility",
        "drift time",
        "ion mobility drift time",
        "mean ion mobility array",
    ]

    for spectrum in exp:
        if spectrum.getMSLevel() != 1:
            continue
        float_arrays = spectrum.getFloatDataArrays()
        for fda in float_arrays:
            name = fda.getName().lower() if fda.getName() else ""
            for im_name in im_array_names:
                if im_name in name:
                    if "inverse" in name or "1/k0" in name:
                        return 'tims', 'Vs/cm2', []
                    elif "drift" in name:
                        return 'tims', 'ms', []
                    else:
                        return 'tims', '', []

    return 'none', '', []


def extract_mzml_to_parquet(
    mzml_path: Path,
    paths: dict,
    status_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Extract mzML data to raw parquet files using vectorized numpy operations.

    Also detects and extracts ion mobility (FAIMS or TIMS) data.

    Args:
        mzml_path: Path to the mzML file
        paths: Dictionary of cache paths from get_cache_paths()
        status_callback: Optional callback for progress updates
    """
    if status_callback:
        status_callback("Parsing mzML file...")

    exp = MSExperiment()
    MzMLFile().load(str(mzml_path), exp)

    # Detect ion mobility type
    if status_callback:
        status_callback("Detecting ion mobility data...")
    im_type, im_unit, cv_values = detect_ion_mobility_type(exp)

    # For FAIMS: create mapping from CV value to im_id
    cv_to_im_id = {}
    unique_cvs = []
    if im_type == 'faims':
        unique_cvs = sorted(set(cv for cv in cv_values if cv is not None))
        cv_to_im_id = {cv: idx for idx, cv in enumerate(unique_cvs)}

    # Count total peaks for pre-allocation
    total_peaks = sum(spec.size() for spec in exp)
    num_spectra = exp.size()

    if status_callback:
        status_callback(f"Processing {num_spectra:,} spectra with {total_peaks:,} peaks...")

    # Pre-allocate numpy arrays for peaks
    peak_ids = np.empty(total_peaks, dtype=np.int64)
    scan_ids = np.empty(total_peaks, dtype=np.int32)
    rts = np.empty(total_peaks, dtype=np.float32)
    mzs = np.empty(total_peaks, dtype=np.float32)
    intensities = np.empty(total_peaks, dtype=np.float32)
    peak_im_ids = np.empty(total_peaks, dtype=np.int32)

    # Pre-allocate arrays for metadata
    meta_scan_ids = np.empty(num_spectra, dtype=np.int32)
    meta_rts = np.empty(num_spectra, dtype=np.float32)
    meta_ms_levels = np.empty(num_spectra, dtype=np.int8)
    meta_precursor_mzs = np.empty(num_spectra, dtype=np.float32)
    meta_charges = np.empty(num_spectra, dtype=np.int8)
    meta_num_peaks = np.empty(num_spectra, dtype=np.int32)
    meta_im_ids = np.empty(num_spectra, dtype=np.int32)

    # Fill arrays
    peak_idx = 0
    for spec_idx, spectrum in enumerate(exp):
        rt = spectrum.getRT() / 60.0  # Convert to minutes
        ms_level = spectrum.getMSLevel()
        mz_array, intensity_array = spectrum.get_peaks()
        n_peaks = len(mz_array)
        scan_id = spec_idx + 1

        # Metadata
        meta_scan_ids[spec_idx] = scan_id
        meta_rts[spec_idx] = rt
        meta_ms_levels[spec_idx] = ms_level
        meta_num_peaks[spec_idx] = n_peaks

        # Ion mobility ID for this spectrum (FAIMS)
        if im_type == 'faims' and cv_values[spec_idx] is not None:
            im_id = cv_to_im_id[cv_values[spec_idx]]
        else:
            im_id = -1
        meta_im_ids[spec_idx] = im_id

        if ms_level > 1 and spectrum.getPrecursors():
            precursor = spectrum.getPrecursors()[0]
            meta_precursor_mzs[spec_idx] = precursor.getMZ()
            meta_charges[spec_idx] = precursor.getCharge()
        else:
            meta_precursor_mzs[spec_idx] = 0.0
            meta_charges[spec_idx] = 0

        # Peaks
        if n_peaks > 0:
            end_idx = peak_idx + n_peaks
            peak_ids[peak_idx:end_idx] = np.arange(peak_idx, end_idx)
            scan_ids[peak_idx:end_idx] = scan_id
            rts[peak_idx:end_idx] = rt
            mzs[peak_idx:end_idx] = mz_array
            intensities[peak_idx:end_idx] = intensity_array
            peak_im_ids[peak_idx:end_idx] = im_id
            peak_idx = end_idx

    # Create directories and write files
    if status_callback:
        status_callback("Writing parquet files...")

    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)

    metadata_df = pl.DataFrame({
        "scan_id": meta_scan_ids,
        "name": [f"Scan_{i}" for i in meta_scan_ids],
        "retention_time": meta_rts,
        "ms_level": meta_ms_levels,
        "precursor_mz": meta_precursor_mzs,
        "charge": meta_charges,
        "num_peaks": meta_num_peaks,
        "im_id": meta_im_ids,
    })
    metadata_df.write_parquet(paths["metadata"])

    # Write peaks
    if peak_idx > 0:
        peaks_df = pl.DataFrame({
            "peak_id": peak_ids[:peak_idx],
            "scan_id": scan_ids[:peak_idx],
            "retention_time": rts[:peak_idx],
            "mass": mzs[:peak_idx],
            "intensity": intensities[:peak_idx],
            "im_id": peak_im_ids[:peak_idx],
        })
        peaks_df.write_parquet(paths["peaks"])
    else:
        pl.DataFrame({
            "peak_id": pl.Series([], dtype=pl.Int64),
            "scan_id": pl.Series([], dtype=pl.Int32),
            "retention_time": pl.Series([], dtype=pl.Float32),
            "mass": pl.Series([], dtype=pl.Float32),
            "intensity": pl.Series([], dtype=pl.Float32),
            "im_id": pl.Series([], dtype=pl.Int32),
        }).write_parquet(paths["peaks"])

    # Write ion mobility info
    im_info = {
        "type": im_type,
        "unit": im_unit,
        "unique_values": unique_cvs if im_type == 'faims' else [],
        "num_dimensions": len(unique_cvs) if im_type == 'faims' else 0,
    }
    with open(paths["im_info"], "w") as f:
        json.dump(im_info, f, indent=2)

    if status_callback:
        if im_type != 'none':
            status_callback(f"Extracted {num_spectra:,} spectra with {im_type.upper()} data")
        else:
            status_callback(f"Extracted {num_spectra:,} spectra")
