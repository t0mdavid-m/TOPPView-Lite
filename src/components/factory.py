"""Factory for creating visualization components.

This module creates Table, LinePlot, Heatmap, and SequenceView components
using the openms_insight patterns including cache reconstruction and
LinePlot.from_sequence_view() for linked annotated spectrum display.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import polars as pl
from openms_insight import Table, LinePlot, Heatmap, SequenceView


def create_ms_components(
    paths: dict,
    im_info: dict,
    file_id: str = "",
) -> Dict[str, Any]:
    """Create mzML visualization components (without identification data).

    Args:
        paths: Dictionary of cache paths from get_cache_paths()
        im_info: Ion mobility info dict from load_im_info()
        file_id: Unique identifier for the file (used in cache IDs)

    Returns:
        Dict with keys: im_table, spectra_table, peaks_table, spectrum_plot, heatmap
        im_table is None if no ion mobility data or only one IM dimension.
    """
    has_im = im_info.get("type") != "none"
    num_im_dimensions = im_info.get("num_dimensions", 0)
    # Only enable IM filtering when multiple dimensions exist
    use_im_filter = has_im and num_im_dimensions > 1
    cache_path = str(paths["component_cache"])
    prefix = f"{file_id}_" if file_id else ""

    # Ensure component cache directory exists
    paths["component_cache"].mkdir(parents=True, exist_ok=True)

    # Ion Mobility Table (only if multiple IM dimensions to filter by)
    im_table = None
    if use_im_filter and paths["im_table"].exists():
        im_table = Table(
            cache_id=f"{prefix}im_table",
            data_path=str(paths["im_table"]),
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

    # Spectra Table (basic, without ID counts)
    spectra_filters = {"im_dimension": "im_id"} if use_im_filter else None
    spectra_table = Table(
        cache_id=f"{prefix}spectra_table",
        data_path=str(paths["spectra_table"]),
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

    # Peaks Table
    peaks_filters = {"spectrum": "scan_id"}
    if use_im_filter:
        peaks_filters["im_dimension"] = "im_id"

    peaks_table = Table(
        cache_id=f"{prefix}peaks_table",
        data_path=str(paths["peaks_table"]),
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

    # Spectrum Plot (basic, without annotations)
    plot_filters = {"spectrum": "scan_id"}
    if use_im_filter:
        plot_filters["im_dimension"] = "im_id"

    spectrum_plot = LinePlot(
        cache_id=f"{prefix}spectrum_plot",
        data_path=str(paths["spectrum_plot"]),
        cache_path=cache_path,
        filters=plot_filters,
        interactivity={"peak": "peak_id"},
        x_column="mass",
        y_column="intensity",
        title="Mass Spectrum",
        x_label="m/z",
        y_label="Intensity",
        styling={
            "unhighlightedColor": "#4A90D9",
            "selectedColor": "#F3A712",
        },
    )

    # Heatmap - data_path triggers subprocess preprocessing for memory efficiency
    heatmap_filters = {"im_dimension": "im_id"} if use_im_filter else None
    categorical_filters = ["im_dimension"] if use_im_filter else None

    heatmap = Heatmap(
        cache_id=f"{prefix}peaks_heatmap",
        data_path=str(paths["heatmap_input"]),
        cache_path=cache_path,
        filters=heatmap_filters,
        categorical_filters=categorical_filters,
        x_column="retention_time",
        y_column="mass",
        intensity_column="intensity",
        interactivity={"spectrum": "scan_id", "peak": "peak_id"},
        title="Peak Map",
        x_label="Retention Time (min)",
        y_label="m/z",
        min_points=20000,
        colorscale="Portland",
    )

    return {
        "im_table": im_table,
        "spectra_table": spectra_table,
        "peaks_table": peaks_table,
        "spectrum_plot": spectrum_plot,
        "heatmap": heatmap,
    }


def create_id_components(
    paths: dict,
    id_paths: dict,
    im_info: dict,
    file_id: str = "",
    search_params: Optional[dict] = None,
) -> Dict[str, Any]:
    """Create identification-related components using SequenceView + LinePlot linking.

    This creates:
    - spectra_table_with_ids: Spectra table with num_ids column
    - id_table: Identifications table
    - sequence_view: SequenceView for peptide visualization
    - annotated_spectrum_plot: LinePlot linked to SequenceView for annotations

    Args:
        paths: Dictionary of cache paths from get_cache_paths()
        id_paths: Dictionary of identification cache paths
        im_info: Ion mobility info dict
        file_id: Unique identifier for the file
        search_params: Optional search parameters dict for annotation config

    Returns:
        Dict with keys: spectra_table_with_ids, id_table, sequence_view, annotated_spectrum_plot
    """
    has_im = im_info.get("type") != "none"
    num_im_dimensions = im_info.get("num_dimensions", 0)
    # Only enable IM filtering when multiple dimensions exist
    use_im_filter = has_im and num_im_dimensions > 1
    cache_path = str(paths["component_cache"])
    prefix = f"{file_id}_" if file_id else ""

    # Load identification data for SequenceView
    id_df = pl.read_parquet(id_paths["identifications"])

    # Spectra Table with ID counts
    spectra_filters = {"im_dimension": "im_id"} if use_im_filter else None
    spectra_table_with_ids = Table(
        cache_id=f"{prefix}spectra_table_with_ids",
        data_path=str(paths["spectra_table_with_ids"]),
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
            {"field": "num_ids", "title": "# IDs", "sorter": "number", "width": 60},
        ],
        title="Spectra",
        index_field="scan_id",
        go_to_fields=["scan_id", "name"],
        default_row=0,
    )

    # Identifications Table
    id_table = Table(
        cache_id=f"{prefix}id_table",
        data=id_df.lazy(),
        cache_path=cache_path,
        filters={"spectrum": "scan_id"},
        interactivity={"identification": "id_idx"},
        column_definitions=[
            {"field": "sequence_display", "title": "Sequence", "headerTooltip": True},
            {
                "field": "score",
                "title": "Score",
                "sorter": "number",
                "hozAlign": "right",
                "formatter": "money",
                "formatterParams": {"precision": 2, "symbol": ""},
            },
            {"field": "charge", "title": "z", "sorter": "number", "hozAlign": "right"},
            {"field": "protein_accession", "title": "Protein", "headerTooltip": True},
        ],
        index_field="id_idx",
        title="Identifications",
    )

    # Build annotation config from search params
    annotation_config = {
        "ion_types": ["b", "y"],
        "neutral_losses": True,
        "proton_loss_addition": True,
        "tolerance": 20.0,
        "tolerance_ppm": True,
    }
    if search_params:
        tol = search_params.get("fragment_mass_tolerance")
        if tol is not None:
            annotation_config["tolerance"] = float(tol)
        is_ppm = search_params.get("fragment_mass_tolerance_ppm")
        if is_ppm is not None:
            annotation_config["tolerance_ppm"] = bool(is_ppm)

    # SequenceView - handles fragment matching in Vue
    # Prepare sequence data (filtered by identification)
    sequence_data = id_df.lazy().select([
        pl.col("id_idx").alias("sequence_id"),
        "sequence",
        pl.col("charge").alias("precursor_charge"),
    ])

    # SequenceView with peaks filtered by spectrum selection
    sequence_view = SequenceView(
        cache_id=f"{prefix}sequence_view",
        sequence_data=sequence_data,
        peaks_data=pl.scan_parquet(paths["spectrum_plot"]),
        filters={
            "identification": "sequence_id",  # Filter sequence by selected identification
            "spectrum": "scan_id",             # Filter peaks by selected spectrum
        },
        interactivity={"peak": "peak_id"},
        deconvolved=False,  # m/z data, not deconvolved neutral masses
        annotation_config=annotation_config,
        cache_path=cache_path,
    )

    # Create annotated LinePlot linked to SequenceView
    # This automatically inherits filters and interactivity from SequenceView
    annotated_spectrum_plot = LinePlot.from_sequence_view(
        sequence_view,
        cache_id=f"{prefix}annotated_spectrum",
        cache_path=cache_path,
        title="Annotated Spectrum",
        styling={
            "unhighlightedColor": "#CCCCCC",
            "highlightColor": "#E74C3C",
            "selectedColor": "#F3A712",
        },
    )

    return {
        "spectra_table_with_ids": spectra_table_with_ids,
        "id_table": id_table,
        "sequence_view": sequence_view,
        "annotated_spectrum_plot": annotated_spectrum_plot,
    }


def reconstruct_ms_components(
    cache_path: str,
    file_id: str = "",
    num_im_dimensions: int = 0,
) -> Dict[str, Any]:
    """Reconstruct MS components from cache (no data needed).

    Uses openms_insight cache reconstruction mode - components reload
    their configuration and data references from the cached manifest.

    Args:
        cache_path: Path to component cache directory
        file_id: File identifier used when creating components
        num_im_dimensions: Number of ion mobility dimensions (table shown if > 1)

    Returns:
        Dict with keys: im_table, spectra_table, peaks_table, spectrum_plot, heatmap
    """
    prefix = f"{file_id}_" if file_id else ""

    # Reconstruct from cache (only cache_id and cache_path needed)
    # Only reconstruct IM table if multiple dimensions exist
    im_table = None
    if num_im_dimensions > 1:
        try:
            im_table = Table(cache_id=f"{prefix}im_table", cache_path=cache_path)
        except Exception:
            pass  # IM table may not exist

    spectra_table = Table(cache_id=f"{prefix}spectra_table", cache_path=cache_path)
    peaks_table = Table(cache_id=f"{prefix}peaks_table", cache_path=cache_path)
    spectrum_plot = LinePlot(cache_id=f"{prefix}spectrum_plot", cache_path=cache_path)
    heatmap = Heatmap(cache_id=f"{prefix}peaks_heatmap", cache_path=cache_path)

    return {
        "im_table": im_table,
        "spectra_table": spectra_table,
        "peaks_table": peaks_table,
        "spectrum_plot": spectrum_plot,
        "heatmap": heatmap,
    }


def reconstruct_id_components(
    cache_path: str,
    file_id: str = "",
) -> Dict[str, Any]:
    """Reconstruct identification components from cache.

    Args:
        cache_path: Path to component cache directory
        file_id: File identifier used when creating components

    Returns:
        Dict with keys: spectra_table_with_ids, id_table, sequence_view, annotated_spectrum_plot
    """
    prefix = f"{file_id}_" if file_id else ""

    spectra_table_with_ids = Table(
        cache_id=f"{prefix}spectra_table_with_ids",
        cache_path=cache_path
    )
    id_table = Table(cache_id=f"{prefix}id_table", cache_path=cache_path)
    sequence_view = SequenceView(
        cache_id=f"{prefix}sequence_view",
        cache_path=cache_path
    )
    annotated_spectrum_plot = LinePlot(
        cache_id=f"{prefix}annotated_spectrum",
        cache_path=cache_path
    )

    return {
        "spectra_table_with_ids": spectra_table_with_ids,
        "id_table": id_table,
        "sequence_view": sequence_view,
        "annotated_spectrum_plot": annotated_spectrum_plot,
    }
