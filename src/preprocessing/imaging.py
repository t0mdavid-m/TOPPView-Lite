"""Mass spectrometry imaging (imzML).

An imzML file (with its binary ``.ibd`` beside it) holds one spectrum per
pixel. It is read with pyOpenMS' ``ImzMLFile`` and written as an mzML file
into the workspace's mzML folder, so every panel of the viewer works on it:
each pixel becomes a spectrum whose retention time is its index (imzML has no
retention time) and which keeps its position as the user parameters
``imzml:x`` / ``imzml:y``. Extraction copies these into the ``pixel_x`` /
``pixel_y`` columns of ``metadata.parquet``.

The ion image panel then shows, per pixel, the summed intensity of every peak
(TIC image) or of the peaks within a tolerance of one m/z (ion image).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

IMZML_DIR_NAME = "imzML-files"
PIXEL_COLUMNS = ("pixel_x", "pixel_y")
DEFAULT_PPM = 10.0


def imzml_dir(workspace: Path) -> Path:
    return Path(workspace, IMZML_DIR_NAME)


def find_pairs(directory: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    """The imzML files of a folder by stem, with their .ibd (None if missing)."""
    found: Dict[str, Dict[str, Optional[Path]]] = {}
    if not directory.exists():
        return found
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in (".imzml", ".ibd"):
            continue
        entry = found.setdefault(path.stem, {"imzML": None, "ibd": None})
        entry["imzML" if suffix == ".imzml" else "ibd"] = path
    return found


def convert_imzml_to_mzml(imzml_path: Path, mzml_path: Path) -> int:
    """Write an imzML file (its .ibd beside it) as mzML. Returns the pixel count.

    Each pixel's spectrum gets its index as retention time (seconds) and at
    least MS level 1 (the reader leaves 0); peaks of zero intensity, which
    some centroiders leave, are dropped.
    """
    from pyopenms import ImzMLFile, MSImagingExperiment, MzMLFile

    imaging = MSImagingExperiment()
    ImzMLFile().load(str(Path(imzml_path).resolve()), imaging)
    exp = imaging.getMSExperiment()
    spectra = exp.getSpectra()
    if not spectra:
        raise ValueError(f"{Path(imzml_path).name} holds no spectra")
    for index, spectrum in enumerate(spectra):
        spectrum.setRT(float(index))
        if spectrum.getMSLevel() < 1:
            spectrum.setMSLevel(1)
        _, intensity = spectrum.get_peaks()
        keep = np.asarray(intensity) > 0
        if len(keep) and not keep.all():
            # select() keeps the float data arrays aligned with the peaks
            spectrum.select(np.flatnonzero(keep).tolist())
    exp.setSpectra(spectra)
    mzml_path.parent.mkdir(parents=True, exist_ok=True)
    MzMLFile().store(str(mzml_path), exp)
    return len(spectra)


def convert_pairs(workspace: Path, mzml_dir: Path) -> List[str]:
    """Convert every complete imzML/ibd pair without a newer mzML. Returns messages."""
    messages = []
    for stem, pair in find_pairs(imzml_dir(workspace)).items():
        if pair["imzML"] is None or pair["ibd"] is None:
            missing = ".ibd" if pair["ibd"] is None else ".imzML"
            messages.append(f"{stem}: waiting for its {missing} file")
            continue
        target = mzml_dir / f"{stem}.mzML"
        newest = max(pair["imzML"].stat().st_mtime, pair["ibd"].stat().st_mtime)
        if target.exists() and target.stat().st_mtime >= newest:
            continue
        try:
            pixels = convert_imzml_to_mzml(pair["imzML"], target)
        except Exception as err:  # the reader raises plain RuntimeErrors
            messages.append(f"{stem}: could not be read ({err})")
            continue
        messages.append(f"{stem}: {pixels:,} pixels added as {target.name}")
    return messages


def pixel_position(spectrum) -> tuple:
    """(x, y) of an imaging spectrum, or (-1, -1) for any other spectrum."""
    if spectrum.metaValueExists("imzml:x") and spectrum.metaValueExists("imzml:y"):
        return int(spectrum.getMetaValue("imzml:x")), int(spectrum.getMetaValue("imzml:y"))
    return -1, -1


def has_pixels(paths: dict) -> bool:
    """True when the file's spectra are pixels of an image."""
    metadata = paths["metadata"]
    if not metadata.exists():
        return False
    frame = pl.scan_parquet(metadata)
    if "pixel_x" not in frame.collect_schema().names():
        return False
    return frame.filter(pl.col("pixel_x") >= 0).select(pl.len()).collect().item() > 0


def mz_window(mz: float, ppm: float) -> tuple:
    delta = abs(mz) * ppm * 1e-6
    return mz - delta, mz + delta


def ion_image_name(mz: Optional[float], ppm: float) -> str:
    """File stem / cache id part of one image (``tic`` for the TIC image)."""
    if mz is None:
        return "tic"
    return f"mz{mz:.4f}_ppm{ppm:g}".replace(".", "p")


def create_ion_image(paths: dict, mz: Optional[float], ppm: float = DEFAULT_PPM) -> Path:
    """Write one image (``scan_id``, ``pixel_x``, ``pixel_y``, ``intensity``).

    With ``mz`` None the intensity is the pixel's summed intensity (TIC),
    else the summed intensity of its peaks within ``ppm`` of ``mz``. Every
    pixel is listed, those without such a peak with 0. Written once per
    image next to the other component inputs.
    """
    out = paths["heatmap_input"].parent / f"ion_image_{ion_image_name(mz, ppm)}.parquet"
    if out.exists() and out.stat().st_mtime >= paths["peaks"].stat().st_mtime:
        return out
    pixels = (
        pl.scan_parquet(paths["metadata"])
        .filter(pl.col("pixel_x") >= 0)
        .select("scan_id", *PIXEL_COLUMNS)
    )
    peaks = pl.scan_parquet(paths["peaks"])
    if mz is not None:
        lo, hi = mz_window(mz, ppm)
        peaks = peaks.filter(pl.col("mass").is_between(lo, hi))
    sums = peaks.group_by("scan_id").agg(
        pl.col("intensity").cast(pl.Float64).sum().alias("intensity")
    )
    image = (
        pixels.join(sums, on="scan_id", how="left")
        .with_columns(pl.col("intensity").fill_null(0.0))
        .sort("scan_id")
        .collect()
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.write_parquet(out)
    return out


def peak_mz(paths: dict, peak_id: Optional[int]) -> Optional[float]:
    """m/z of a peak by its ``peak_id``, or None."""
    if peak_id is None:
        return None
    row = (
        pl.scan_parquet(paths["peaks"])
        .filter(pl.col("peak_id") == int(peak_id))
        .select("mass")
        .collect()
    )
    return float(row["mass"][0]) if row.height else None
