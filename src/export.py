"""Export a filtered copy of an mzML file.

What is exported follows the viewer's tables: the spectra are the rows the
Spectra table shows after its column filters, and the m/z and intensity window
comes from the column filters of the Peak List table.

Uses the TOPP tool FileFilter when it is on PATH (the full Docker image and the
Windows installer ship it) and the table filters come down to an RT range and a
set of MS levels. Other selections (precursor m/z, charge, scan number, ...)
pick spectra FileFilter cannot express, and the pyOpenMS-only image has no TOPP
tools; the filter is then applied with pyOpenMS directly.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import pyopenms as oms

EXPORT_DIR_NAME = "exports"

Range = Tuple[float, float]

# Peak List columns whose filters become the exported m/z and intensity window.
PEAK_FILTER_COLUMNS = {"mass": "mz", "intensity": "int"}

# How far (seconds) the RT bounds handed to FileFilter may lie from the first and
# last selected spectrum. The metadata stores RT in minutes as float32, so the
# bounds are widened a little; no spectrum outside the selection may lie this
# close, or FileFilter is not used.
RT_TOLERANCE_SEC = 0.01


def filtered_output_path(workspace: Path, mzml_path: Path) -> Path:
    """Where the filtered copy of ``mzml_path`` is written."""
    return Path(workspace) / EXPORT_DIR_NAME / f"{Path(mzml_path).stem}_filtered.mzML"


def _range_arg(bounds: Optional[Range]) -> Optional[str]:
    if bounds is None:
        return None
    lo, hi = bounds
    return f"{'' if lo is None else lo}:{'' if hi is None else hi}"


def filefilter_command(
    in_path: Path,
    out_path: Path,
    rt_range_sec: Optional[Range] = None,
    mz_range: Optional[Range] = None,
    ms_levels: Optional[Iterable[int]] = None,
    int_range: Optional[Range] = None,
    executable: str = "FileFilter",
) -> list:
    """Build the FileFilter command line. ``None`` leaves that filter out."""
    cmd = [executable, "-in", str(in_path), "-out", str(out_path)]
    for flag, bounds in (("-rt", rt_range_sec), ("-mz", mz_range), ("-int", int_range)):
        arg = _range_arg(bounds)
        if arg is not None:
            cmd += [flag, arg]
    if ms_levels is not None:
        cmd += ["-peak_options:level", *[str(int(level)) for level in sorted(ms_levels)]]
    return cmd


def _within(values: np.ndarray, bounds: Optional[Range]) -> np.ndarray:
    mask = np.ones(len(values), dtype=bool)
    if bounds is None:
        return mask
    lo, hi = bounds
    if lo is not None:
        mask &= values >= lo
    if hi is not None:
        mask &= values <= hi
    return mask


def filter_with_pyopenms(
    in_path: Path,
    out_path: Path,
    rt_range_sec: Optional[Range] = None,
    mz_range: Optional[Range] = None,
    ms_levels: Optional[Iterable[int]] = None,
    int_range: Optional[Range] = None,
    scan_ids: Optional[Iterable[int]] = None,
) -> int:
    """Apply the filter with pyOpenMS. Returns the number of spectra kept.

    ``scan_ids`` are the viewer's 1-based spectrum numbers (file order).
    """
    levels = None if ms_levels is None else {int(level) for level in ms_levels}
    wanted = None if scan_ids is None else {int(s) for s in scan_ids}

    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(in_path), exp)

    kept = []
    for index, spec in enumerate(exp.getSpectra()):
        if wanted is not None and index + 1 not in wanted:
            continue
        if levels is not None and spec.getMSLevel() not in levels:
            continue
        if not _within(np.array([spec.getRT()]), rt_range_sec)[0]:
            continue
        if mz_range is not None or int_range is not None:
            mz, intensity = spec.get_peaks()
            mz, intensity = np.asarray(mz), np.asarray(intensity)
            mask = _within(mz, mz_range) & _within(intensity, int_range)
            if not mask.all():
                spec.set_peaks((mz[mask], intensity[mask]))
        kept.append(spec)

    exp.setSpectra(kept)
    oms.MzMLFile().store(str(out_path), exp)
    return len(kept)


def _check_range(bounds: Optional[Range]) -> None:
    if bounds is None:
        return
    lo, hi = bounds
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("The lower bound of a range is above its upper bound.")


def export_filtered_mzml(
    in_path: Path,
    out_path: Path,
    rt_range_sec: Optional[Range] = None,
    mz_range: Optional[Range] = None,
    ms_levels: Optional[Iterable[int]] = None,
    int_range: Optional[Range] = None,
    scan_ids: Optional[Iterable[int]] = None,
    use_topp: Optional[bool] = None,
) -> str:
    """Write a filtered mzML. Returns which backend was used ("FileFilter" or "pyOpenMS").

    ``scan_ids`` (a set of spectrum numbers) can only be applied with pyOpenMS.

    Raises:
        ValueError: if no MS level or spectrum is selected, or a range is empty.
        RuntimeError: if FileFilter fails.
    """
    if ms_levels is not None:
        ms_levels = sorted({int(level) for level in ms_levels})
        if not ms_levels:
            raise ValueError("Select at least one MS level.")
    if scan_ids is not None:
        scan_ids = sorted({int(s) for s in scan_ids})
        if not scan_ids:
            raise ValueError("No spectra are selected.")
    for bounds in (rt_range_sec, mz_range, int_range):
        _check_range(bounds)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("FileFilter")
    if use_topp is None:
        use_topp = executable is not None and scan_ids is None
    if use_topp and scan_ids is not None:
        raise ValueError("FileFilter cannot select spectra by number.")

    if use_topp:
        cmd = filefilter_command(
            in_path, out_path, rt_range_sec, mz_range, ms_levels, int_range,
            executable or "FileFilter",
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError(
                "FileFilter failed: " + (message[-1] if message else f"exit {result.returncode}")
            )
        return "FileFilter"

    filter_with_pyopenms(
        in_path, out_path, rt_range_sec, mz_range, ms_levels, int_range, scan_ids
    )
    return "pyOpenMS"


# =============================================================================
# Coupling to the viewer's tables
# =============================================================================


def peak_window(column_filters: Sequence[dict[str, Any]]) -> dict[str, Optional[Range]]:
    """The m/z and intensity window set by the Peak List's column filters.

    Returns ``{"mz": (lo, hi) | None, "int": (lo, hi) | None}``; a missing bound
    is ``None``. Filters on other columns do not restrict the export.
    """
    window: dict[str, list] = {key: [None, None] for key in PEAK_FILTER_COLUMNS.values()}
    for col_filter in column_filters or []:
        key = PEAK_FILTER_COLUMNS.get(col_filter.get("field"))
        value = col_filter.get("value")
        if key is None or value is None:
            continue
        if col_filter.get("type") == ">=":
            window[key][0] = float(value)
        elif col_filter.get("type") == "<=":
            window[key][1] = float(value)
    return {
        key: None if bounds == [None, None] else tuple(bounds)
        for key, bounds in window.items()
    }


def filefilter_equivalent(
    metadata: pl.DataFrame, scan_ids: Iterable[int]
) -> Optional[Tuple[Optional[Range], list]]:
    """Express a set of spectra as FileFilter's RT range and MS levels, if possible.

    ``metadata`` has one row per spectrum with ``scan_id``, ``retention_time``
    (minutes) and ``ms_level``. Returns ``(rt_range_sec, ms_levels)`` when exactly
    the spectra in ``scan_ids`` have one of those levels and an RT in that range
    (``rt_range_sec`` is ``None`` when every spectrum of those levels is
    selected), else ``None``.
    """
    selected = metadata["scan_id"].is_in(sorted({int(s) for s in scan_ids}))
    chosen = metadata.filter(selected)
    if chosen.is_empty():
        return None
    levels = sorted(int(v) for v in chosen["ms_level"].unique().to_list())
    same_level = metadata.filter(pl.col("ms_level").is_in(levels))
    if same_level.height == chosen.height:
        return None, levels

    rt_sec = metadata["retention_time"].cast(pl.Float64) * 60.0
    lo = float(rt_sec.filter(selected).min())
    hi = float(rt_sec.filter(selected).max())
    # No spectrum of these levels outside the selection may lie in the range or
    # near its bounds, where float rounding could let FileFilter keep it.
    margin = 2 * RT_TOLERANCE_SEC
    others = metadata.with_columns(rt_sec=rt_sec).filter(
        ~selected & pl.col("ms_level").is_in(levels)
    )
    if others.filter(pl.col("rt_sec").is_between(lo - margin, hi + margin)).height:
        return None
    return (lo - RT_TOLERANCE_SEC, hi + RT_TOLERANCE_SEC), levels


def export_table_selection(
    in_path: Path,
    out_path: Path,
    metadata: pl.DataFrame,
    scan_ids: Iterable[int],
    peak_filters: Sequence[dict[str, Any]] = (),
    use_topp: Optional[bool] = None,
) -> str:
    """Export the spectra ``scan_ids`` with the Peak List's m/z and intensity window.

    FileFilter is used when it is on PATH (or ``use_topp`` is true) and the
    selection is an RT range over some MS levels; otherwise pyOpenMS.
    """
    scan_ids = sorted({int(s) for s in scan_ids})
    if not scan_ids:
        raise ValueError("The Spectra table's filters leave no spectra to export.")
    window = peak_window(peak_filters)

    if use_topp is None:
        use_topp = shutil.which("FileFilter") is not None
    equivalent = filefilter_equivalent(metadata, scan_ids) if use_topp else None
    if equivalent is not None:
        rt_range_sec, levels = equivalent
        return export_filtered_mzml(
            in_path, out_path, rt_range_sec, window["mz"], levels, window["int"],
            use_topp=True,
        )
    selects_all = len(scan_ids) == metadata.height
    return export_filtered_mzml(
        in_path, out_path, mz_range=window["mz"], int_range=window["int"],
        scan_ids=None if selects_all else scan_ids, use_topp=False,
    )
