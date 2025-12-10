"""Create component-specific input files from raw parquet data.

This module handles the second stage of preprocessing: creating projected
parquet files optimized for each visualization component.
"""

import json
from pathlib import Path
from typing import Optional, Callable

import polars as pl


def load_im_info(paths: dict) -> dict:
    """Load ion mobility info from JSON file."""
    im_info_path = paths["im_info"]
    if im_info_path.exists():
        with open(im_info_path) as f:
            return json.load(f)
    return {"type": "none", "unit": ""}


def component_cache_is_valid(paths: dict) -> bool:
    """Check if component-specific cache files exist and are newer than raw cache."""
    # Check raw files exist first
    if not paths["metadata"].exists() or not paths["peaks"].exists():
        return False

    raw_mtime = max(
        paths["metadata"].stat().st_mtime,
        paths["peaks"].stat().st_mtime,
        paths["im_info"].stat().st_mtime if paths["im_info"].exists() else 0,
    )

    component_files = [
        paths["spectra_table"],
        paths["peaks_table"],
        paths["spectrum_plot"],
        paths["heatmap_input"],
    ]

    for f in component_files:
        if not f.exists():
            return False
        if f.stat().st_mtime < raw_mtime:
            return False

    # Check if im_table needs to be created/updated
    im_info = load_im_info(paths)
    if im_info.get("type") != "none":
        im_table_path = paths["im_table"]
        if not im_table_path.exists():
            return False
        if im_table_path.stat().st_mtime < raw_mtime:
            return False

    return True


def create_component_inputs(
    paths: dict,
    status_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Create component-specific parquet files with only needed columns.

    This projects the raw data to smaller files for each component,
    reducing I/O and memory usage. Also creates the ion mobility summary table
    if IM data is present.

    Args:
        paths: Dictionary of cache paths from get_cache_paths()
        status_callback: Optional callback for progress updates
    """
    if status_callback:
        status_callback("Creating component input files...")

    # Ensure output directory exists
    paths["spectra_table"].parent.mkdir(parents=True, exist_ok=True)

    # Load ion mobility info
    im_info = load_im_info(paths)
    has_im = im_info.get("type") != "none"

    # Spectra table: all metadata columns + im_id if available
    if status_callback:
        status_callback("Creating spectra table input...")

    spectra_cols = [
        "scan_id", "name", "retention_time", "ms_level",
        "precursor_mz", "charge", "num_peaks"
    ]
    if has_im:
        spectra_cols.append("im_id")

    pl.scan_parquet(paths["metadata"]).select(spectra_cols).collect().write_parquet(
        paths["spectra_table"]
    )

    # Peaks table: columns for display + filter column (scan_id) + interactivity (peak_id)
    if status_callback:
        status_callback("Creating peaks table input...")

    peaks_table_cols = ["peak_id", "scan_id", "mass", "intensity"]
    if has_im:
        peaks_table_cols.append("im_id")

    pl.scan_parquet(paths["peaks"]).select(peaks_table_cols).filter(
        pl.col("intensity") > 0
    ).collect().write_parquet(paths["peaks_table"])

    # Spectrum plot: same columns as peaks table
    if status_callback:
        status_callback("Creating spectrum plot input...")

    spectrum_plot_cols = ["peak_id", "scan_id", "mass", "intensity"]
    if has_im:
        spectrum_plot_cols.append("im_id")

    pl.scan_parquet(paths["peaks"]).select(spectrum_plot_cols).filter(
        pl.col("intensity") > 0
    ).collect().write_parquet(paths["spectrum_plot"])

    # Heatmap: needs x (retention_time), y (mass), intensity, scan_id and peak_id
    if status_callback:
        status_callback("Creating heatmap input...")

    heatmap_cols = ["peak_id", "scan_id", "retention_time", "mass", "intensity"]
    if has_im:
        heatmap_cols.append("im_id")

    pl.scan_parquet(paths["peaks"]).select(heatmap_cols).filter(
        pl.col("intensity") > 0
    ).collect().write_parquet(paths["heatmap_input"])

    # Create ion mobility summary table if IM data exists
    if has_im:
        if status_callback:
            status_callback("Creating ion mobility summary table...")
        _create_im_summary_table(paths, im_info)

    if status_callback:
        status_callback("Component inputs created")


def _create_im_summary_table(paths: dict, im_info: dict) -> None:
    """Create ion mobility summary table with statistics per IM dimension.

    For FAIMS: Groups by CV value, shows spectra and peak counts.
    """
    im_type = im_info.get("type")
    im_unit = im_info.get("unit", "")
    unique_values = im_info.get("unique_values", [])

    if im_type == 'faims':
        # Get spectrum counts per IM dimension
        spectra_summary = (
            pl.scan_parquet(paths["metadata"])
            .filter(pl.col("im_id") >= 0)
            .group_by("im_id")
            .agg([
                pl.len().alias("num_spectra"),
            ])
        )

        # Get peak counts and average intensity per IM dimension
        peaks_summary = (
            pl.scan_parquet(paths["peaks"])
            .filter(pl.col("im_id") >= 0)
            .filter(pl.col("intensity") > 0)
            .group_by("im_id")
            .agg([
                pl.len().alias("num_peaks"),
                pl.col("intensity").mean().alias("avg_intensity"),
            ])
        )

        # Join summaries
        summary = spectra_summary.join(peaks_summary, on="im_id", how="left").collect()

        # Add CV value and label columns
        cv_values = []
        labels = []
        for im_id in summary["im_id"].to_list():
            if 0 <= im_id < len(unique_values):
                cv = unique_values[im_id]
                cv_values.append(cv)
                labels.append(f"CV: {cv}{im_unit}")
            else:
                cv_values.append(None)
                labels.append("Unknown")

        summary = summary.with_columns([
            pl.Series("cv_value", cv_values),
            pl.Series("im_label", labels),
        ])

        # Sort by CV value
        summary = summary.sort("cv_value")

        summary.write_parquet(paths["im_table"])
