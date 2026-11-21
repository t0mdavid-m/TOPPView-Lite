"""The precursor of the selected fragment spectrum, marked on the spectrum plot.

An MS2 (or higher) spectrum gets a dotted guide at its precursor m/z, labelled
with the precursor's charge. The most intense peak within
``PRECURSOR_TOLERANCE`` of it, usually precursor ions left unfragmented, gets a
dot. MS1 spectra and spectra without a precursor get no marker.

The marker is a render-time argument of the Insight ``LinePlot``
(``x_markers``), so the plot is rebuilt from its cache with the marker of the
spectrum on screen; its data and cache are unchanged.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
from openms_insight import LinePlot

#: Colour of the precursor guide and its label.
PRECURSOR_COLOR = "#d62728"
#: Distance (m/z) within which a peak counts as the precursor's peak.
PRECURSOR_TOLERANCE = 0.02


def precursor_markers(
    metadata_path: Union[str, Path], scan_id: Any
) -> Optional[List[Dict[str, Any]]]:
    """The ``x_markers`` of spectrum ``scan_id``: its precursor, or None."""
    if scan_id is None or not Path(metadata_path).exists():
        return None
    try:
        scan_id = int(scan_id)
    except (TypeError, ValueError):
        return None
    row = (
        pl.scan_parquet(metadata_path)
        .filter(pl.col("scan_id") == scan_id)
        .select("ms_level", "precursor_mz", "charge")
        .head(1)
        .collect()
    )
    if row.height == 0:
        return None
    ms_level, mz, charge = row.row(0)
    if ms_level is None or ms_level < 2 or mz is None or not mz > 0:
        return None
    label = "Precursor"
    if charge is not None and charge > 0:
        label += f" {int(charge)}+"
    return [{"x": round(float(mz), 4), "label": label, "color": PRECURSOR_COLOR}]


def with_precursor(
    plot: LinePlot,
    cache_id: str,
    cache_path: str,
    markers: Optional[List[Dict[str, Any]]],
) -> LinePlot:
    """``plot``, or the same plot from its cache with ``markers`` drawn."""
    if not markers:
        return plot
    return LinePlot(
        cache_id=cache_id,
        cache_path=cache_path,
        x_markers=markers,
        x_marker_tolerance=PRECURSOR_TOLERANCE,
    )
