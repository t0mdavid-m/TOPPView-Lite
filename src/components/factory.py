"""Factory for creating visualization components.

This module creates Table, LinePlot, and Heatmap components from
preprocessed parquet files.
"""

from pathlib import Path
from typing import Optional, Tuple

import polars as pl
from openms_insight import Table, LinePlot, Heatmap, preprocess_component


def create_components(
    paths: dict,
    im_info: dict,
    file_id: str = "",
    has_ids: bool = False,
) -> Tuple[Optional[Table], Table, Table, LinePlot, Heatmap]:
    """Create all visualization components for a preprocessed file.

    Args:
        paths: Dictionary of cache paths from get_cache_paths()
        im_info: Ion mobility info dict from load_im_info()
        file_id: Unique identifier for the file (used in cache IDs)
        has_ids: Whether identification data exists (enables annotation filters)

    Returns:
        Tuple of (im_table, spectra_table, peaks_table, spectrum_plot, heatmap)
        im_table is None if no ion mobility data is present.
    """
    has_im = im_info.get("type") != "none"
    cache_path = str(paths["component_cache"])

    # Use file_id prefix for cache_ids to separate different files
    prefix = f"{file_id}_" if file_id else ""

    # Ensure component cache directory exists
    paths["component_cache"].mkdir(parents=True, exist_ok=True)

    # Ion Mobility Table (optional)
    im_table = None
    if has_im and paths["im_table"].exists():
        im_table = Table(
            cache_id=f"{prefix}im_table",
            data=pl.scan_parquet(paths["im_table"]),
            cache_path=cache_path,
            interactivity={"im_dimension": "im_id"},
            column_definitions=[
                {"field": "im_label", "title": "Ion Mobility", "width": 100},
                {"field": "num_spectra", "title": "# Spectra", "sorter": "number", "width": 80},
                {"field": "num_peaks", "title": "# Peaks", "sorter": "number", "width": 80},
                {
                    "field": "avg_intensity",
                    "title": "Avg Int",
                    "sorter": "number",
                    "formatter": "money",
                    "formatterParams": {"precision": 0, "symbol": ""},
                    "width": 90,
                },
            ],
            title="Ion Mobility",
            index_field="im_id",
            default_row=0,
        )

    # Spectra Table
    spectra_filters = {"im_dimension": "im_id"} if has_im else None
    spectra_table = Table(
        cache_id=f"{prefix}spectra_table",
        data=pl.scan_parquet(paths["spectra_table"]),
        cache_path=cache_path,
        filters=spectra_filters,
        interactivity={"spectrum": "scan_id"},
        column_definitions=[
            {"field": "scan_id", "title": "Scan ID", "sorter": "number", "width": 80},
            {"field": "name", "title": "Name"},
            {
                "field": "retention_time",
                "title": "RT (min)",
                "sorter": "number",
                "formatter": "money",
                "formatterParams": {"precision": 2, "symbol": ""},
            },
            {"field": "ms_level", "title": "MS", "sorter": "number", "width": 50},
            {
                "field": "precursor_mz",
                "title": "Precursor m/z",
                "sorter": "number",
                "formatter": "money",
                "formatterParams": {"precision": 2, "symbol": ""},
            },
            {"field": "charge", "title": "z", "sorter": "number", "width": 50},
            {"field": "num_peaks", "title": "# Peaks", "sorter": "number", "width": 80},
        ],
        title="Spectra",
        index_field="scan_id",
        go_to_fields=["scan_id", "name"],
        default_row=0,
    )

    # Spectrum Plot
    # Use unified annotated data if available (has_ids and file exists)
    annotated_path = paths["annotated_spectrum_plot"]
    use_annotations = has_ids and annotated_path.exists()

    # Build LinePlot configuration
    plot_kwargs = {
        "cache_id": f"{prefix}spectrum_plot",
        "cache_path": cache_path,
        "interactivity": {"peak": "peak_id"},
        "x_column": "mass",
        "y_column": "intensity",
        "title": "Mass Spectrum",
        "x_label": "m/z",
        "y_label": "Intensity",
        "styling": {
            "unhighlightedColor": "#4A90D9",
            "selectedColor": "#F3A712",
        },
    }

    if use_annotations:
        # Use unified annotated data with identification filter
        # Note: im_dimension filtering happens via spectra_table → spectrum selection
        # The annotated data doesn't have im_id column, so we only filter by spectrum and identification
        plot_kwargs["data"] = pl.scan_parquet(annotated_path)
        plot_kwargs["filters"] = {"spectrum": "scan_id", "identification": "id_idx"}
        plot_kwargs["filter_defaults"] = {"identification": -1}
        plot_kwargs["highlight_column"] = "highlight"
        plot_kwargs["annotation_column"] = "annotation"
    else:
        # Use basic spectrum plot data (may have im_id column)
        plot_filters = {"spectrum": "scan_id"}
        if has_im:
            plot_filters["im_dimension"] = "im_id"
        plot_kwargs["data"] = pl.scan_parquet(paths["spectrum_plot"])
        plot_kwargs["filters"] = plot_filters

    spectrum_plot = LinePlot(**plot_kwargs)

    # Peaks Table
    peaks_filters = {"spectrum": "scan_id"}
    if has_im:
        peaks_filters["im_dimension"] = "im_id"

    peaks_table = Table(
        cache_id=f"{prefix}peaks_table",
        data=pl.scan_parquet(paths["peaks_table"]),
        cache_path=cache_path,
        filters=peaks_filters,
        interactivity={"peak": "peak_id"},
        column_definitions=[
            {
                "field": "mass",
                "title": "m/z",
                "sorter": "number",
                "formatter": "money",
                "formatterParams": {"precision": 4, "symbol": ""},
            },
            {
                "field": "intensity",
                "title": "Intensity",
                "sorter": "number",
                "formatter": "money",
                "formatterParams": {"precision": 2, "symbol": ""},
            },
        ],
        title="Peak List",
        index_field="peak_id",
        initial_sort=[{"column": "intensity", "dir": "desc"}],
        default_row=-1,
    )

    # Heatmap - use subprocess preprocessing to release memory after cache creation
    heatmap_filters = {"im_dimension": "im_id"} if has_im else None
    categorical_filters = ["im_dimension"] if has_im else None
    heatmap_cache_id = f"{prefix}peaks_heatmap"

    heatmap_kwargs = {
        "cache_id": heatmap_cache_id,
        "cache_path": cache_path,
        "filters": heatmap_filters,
        "categorical_filters": categorical_filters,
        "x_column": "retention_time",
        "y_column": "mass",
        "intensity_column": "intensity",
        "interactivity": {"spectrum": "scan_id", "peak": "peak_id"},
        "title": "Peak Map",
        "x_label": "Retention Time (min)",
        "y_label": "m/z",
        "min_points": 20000,
        "colorscale": "Portland",
    }

    # Try to load from cache first, otherwise preprocess in subprocess
    try:
        heatmap = Heatmap(**heatmap_kwargs)  # No data = load from cache
    except Exception:
        # Cache miss - preprocess in subprocess to release memory after
        preprocess_component(
            Heatmap,
            data_path=str(paths["heatmap_input"]),
            **heatmap_kwargs,
        )
        # Now load from cache
        heatmap = Heatmap(**heatmap_kwargs)

    return im_table, spectra_table, peaks_table, spectrum_plot, heatmap
