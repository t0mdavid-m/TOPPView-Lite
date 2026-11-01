"""Export a filtered copy of an mzML file (RT range, m/z range, MS levels).

Uses the TOPP tool FileFilter when it is on PATH (the full Docker image and the
Windows installer ship it). The pyOpenMS-only image has no TOPP tools, so the
same filter is then applied with pyOpenMS directly.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pyopenms as oms

EXPORT_DIR_NAME = "exports"


def filtered_output_path(workspace: Path, mzml_path: Path) -> Path:
    """Where the filtered copy of ``mzml_path`` is written."""
    return Path(workspace) / EXPORT_DIR_NAME / f"{Path(mzml_path).stem}_filtered.mzML"


def filefilter_command(
    in_path: Path,
    out_path: Path,
    rt_range_sec: Tuple[float, float],
    mz_range: Tuple[float, float],
    ms_levels: Iterable[int],
    executable: str = "FileFilter",
) -> list:
    """Build the FileFilter command line."""
    return [
        executable,
        "-in", str(in_path),
        "-out", str(out_path),
        "-rt", f"{rt_range_sec[0]}:{rt_range_sec[1]}",
        "-mz", f"{mz_range[0]}:{mz_range[1]}",
        "-peak_options:level", *[str(int(level)) for level in sorted(ms_levels)],
    ]


def filter_with_pyopenms(
    in_path: Path,
    out_path: Path,
    rt_range_sec: Tuple[float, float],
    mz_range: Tuple[float, float],
    ms_levels: Iterable[int],
) -> int:
    """Apply the same filter as FileFilter with pyOpenMS. Returns the number of spectra kept."""
    levels = {int(level) for level in ms_levels}
    rt_lo, rt_hi = rt_range_sec
    mz_lo, mz_hi = mz_range

    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(in_path), exp)

    kept = []
    for spec in exp.getSpectra():
        if spec.getMSLevel() not in levels or not rt_lo <= spec.getRT() <= rt_hi:
            continue
        mz, intensity = spec.get_peaks()
        mask = (mz >= mz_lo) & (mz <= mz_hi)
        if not mask.all():
            spec.set_peaks((np.asarray(mz)[mask], np.asarray(intensity)[mask]))
        kept.append(spec)

    exp.setSpectra(kept)
    oms.MzMLFile().store(str(out_path), exp)
    return len(kept)


def export_filtered_mzml(
    in_path: Path,
    out_path: Path,
    rt_range_sec: Tuple[float, float],
    mz_range: Tuple[float, float],
    ms_levels: Iterable[int],
    use_topp: Optional[bool] = None,
) -> str:
    """Write a filtered mzML. Returns which backend was used ("FileFilter" or "pyOpenMS").

    Raises:
        ValueError: if no MS level is selected or a range is empty.
        RuntimeError: if FileFilter fails.
    """
    ms_levels = sorted({int(level) for level in ms_levels})
    if not ms_levels:
        raise ValueError("Select at least one MS level.")
    if rt_range_sec[0] > rt_range_sec[1] or mz_range[0] > mz_range[1]:
        raise ValueError("The lower bound of a range is above its upper bound.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("FileFilter")
    if use_topp is None:
        use_topp = executable is not None

    if use_topp:
        cmd = filefilter_command(
            in_path, out_path, rt_range_sec, mz_range, ms_levels, executable or "FileFilter"
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            raise RuntimeError(
                "FileFilter failed: " + (message[-1] if message else f"exit {result.returncode}")
            )
        return "FileFilter"

    filter_with_pyopenms(in_path, out_path, rt_range_sec, mz_range, ms_levels)
    return "pyOpenMS"
