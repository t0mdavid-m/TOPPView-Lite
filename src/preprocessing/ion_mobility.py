"""Inputs for the ion mobility map and mobilogram of one spectrum.

Files whose spectra carry one ion mobility value per peak (timsTOF 1/K0, drift
time) keep it in the ``ion_mobility`` column of ``peaks.parquet``. From there
this module writes three component inputs:

- ``im_peaks.parquet``: every peak with an ion mobility value
  (``peak_id``, ``scan_id``, ``mass``, ``ion_mobility``, ``intensity``).
- ``im_peaks_binned.parquet``: the most intense peak per spectrum and
  m/z x ion mobility bin, the overview level of the map. A frame of a million
  peaks is shown as at most ``MZ_BINS * IM_BINS`` points; zooming in reaches
  the full-resolution peaks.
- ``mobilogram.parquet``: summed intensity per spectrum and ion mobility bin.
"""

from typing import Optional

import polars as pl

MZ_BINS = 1000
IM_BINS = 200
MOBILOGRAM_BINS = 200

_INPUTS = ("im_peaks", "im_peaks_binned", "mobilogram")


def has_peak_mobility(im_info: dict) -> bool:
    """True for files with one ion mobility value per peak (not FAIMS)."""
    return im_info.get("type") == "tims"


def im_axis_label(im_info: dict) -> str:
    """Axis label for the ion mobility values of a file."""
    unit = im_info.get("unit", "")
    if unit == "Vs/cm2":
        return "Ion mobility 1/K₀ (V·s/cm²)"
    if unit == "ms":
        return "Drift time (ms)"
    return "Ion mobility"


def _bin_index(column: str, lo: float, hi: float, bins: int) -> pl.Expr:
    width = (hi - lo) / bins if hi > lo else 1.0
    return (
        ((pl.col(column) - lo) / width).floor().clip(0, bins - 1).cast(pl.Int32)
    )


def create_im_inputs(paths: dict) -> bool:
    """Write the three ion mobility inputs. Returns False if there is no data."""
    peaks = pl.scan_parquet(paths["peaks"])
    if "ion_mobility" not in peaks.collect_schema().names():
        return False

    im_peaks = (
        peaks.filter(pl.col("ion_mobility").is_not_nan())
        .select("peak_id", "scan_id", "mass", "ion_mobility", "intensity")
        .collect()
    )
    paths["im_peaks"].parent.mkdir(parents=True, exist_ok=True)
    im_peaks.write_parquet(paths["im_peaks"])
    if im_peaks.height == 0:
        return False

    mz_lo, mz_hi = float(im_peaks["mass"].min()), float(im_peaks["mass"].max())
    im_lo = float(im_peaks["ion_mobility"].min())
    im_hi = float(im_peaks["ion_mobility"].max())

    binned = (
        im_peaks.lazy()
        .with_columns(
            _bin_index("mass", mz_lo, mz_hi, MZ_BINS).alias("_mz_bin"),
            _bin_index("ion_mobility", im_lo, im_hi, IM_BINS).alias("_im_bin"),
        )
        .sort("intensity", descending=True)
        .group_by("scan_id", "_mz_bin", "_im_bin", maintain_order=True)
        .first()
        .drop("_mz_bin", "_im_bin")
        .select(im_peaks.columns)
        .collect()
    )
    binned.write_parquet(paths["im_peaks_binned"])

    width = (im_hi - im_lo) / MOBILOGRAM_BINS if im_hi > im_lo else 1.0
    mobilogram = (
        im_peaks.lazy()
        .with_columns(
            _bin_index("ion_mobility", im_lo, im_hi, MOBILOGRAM_BINS).alias("_bin")
        )
        .group_by("scan_id", "_bin")
        .agg(pl.col("intensity").sum().cast(pl.Float64).alias("intensity"))
        .with_columns(
            (im_lo + (pl.col("_bin").cast(pl.Float64) + 0.5) * width).alias(
                "ion_mobility"
            )
        )
        .select("scan_id", "ion_mobility", "intensity")
        .sort("scan_id", "ion_mobility")
        .collect()
    )
    mobilogram.write_parquet(paths["mobilogram"])
    return True


def ensure_im_inputs(paths: dict, im_info: dict) -> Optional[float]:
    """Create the inputs when missing or older than the peaks.

    Returns the modification time of the inputs (part of the components'
    cache ids, so rebuilt inputs get new caches), or None without ion
    mobility data.
    """
    if not has_peak_mobility(im_info) or not paths["peaks"].exists():
        return None
    peaks_mtime = paths["peaks"].stat().st_mtime
    fresh = all(
        paths[key].exists() and paths[key].stat().st_mtime >= peaks_mtime
        for key in _INPUTS
    )
    if not fresh and not create_im_inputs(paths):
        return None
    if not all(paths[key].exists() for key in _INPUTS):
        return None
    if pl.scan_parquet(paths["im_peaks"]).select(pl.len()).collect().item() == 0:
        return None
    return max(paths[key].stat().st_mtime for key in _INPUTS)
