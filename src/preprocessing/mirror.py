"""Mirror view data: experimental spectrum against the theoretical spectrum of an ID.

For every identification linked to a spectrum, pyOpenMS' TheoreticalSpectrumGenerator
produces the b/y fragment ladder of the peptide. Each theoretical ion is matched to the
nearest experimental peak within the search's fragment tolerance. The result is one
parquet file that the Insight MirrorPlot reads:

- experimental rows (top half): ``mirror_top_id`` = id_idx, ``mirror_bottom_id`` = -1
- theoretical rows (bottom half): ``mirror_top_id`` = -1, ``mirror_bottom_id`` = id_idx

``matched`` marks annotated peaks and ``label`` holds the ion names. Theoretical
intensities are scaled to the base peak of the experimental spectrum so both halves
share one axis. A second, small file holds the matched / total ion counts per ID.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl
import pyopenms as oms

ION_TYPES = ("b", "y")
MAX_FRAGMENT_CHARGE = 2

MIRROR_SCHEMA = {
    "mirror_peak_id": pl.Int64,
    "mirror_top_id": pl.Int64,
    "mirror_bottom_id": pl.Int64,
    "mz": pl.Float64,
    "intensity": pl.Float64,
    "matched": pl.Boolean,
    "label": pl.Utf8,
}

SUMMARY_SCHEMA = {
    "id_idx": pl.Int64,
    "matched_ions": pl.Int64,
    "total_ions": pl.Int64,
    "coverage": pl.Float64,
}


def get_mirror_paths(paths: dict) -> dict:
    """Mirror cache files, next to the other component inputs of an mzML."""
    components_dir = Path(paths["spectrum_plot"]).parent
    return {
        "mirror": components_dir / "mirror.parquet",
        "summary": components_dir / "mirror_summary.parquet",
    }


def mirror_cache_is_valid(mirror_paths: dict, identifications_path: Path) -> bool:
    """The cache is valid while it is newer than the identification cache."""
    try:
        id_mtime = Path(identifications_path).stat().st_mtime
        return all(Path(p).stat().st_mtime >= id_mtime for p in mirror_paths.values())
    except FileNotFoundError:
        return False


def _make_generator() -> oms.TheoreticalSpectrumGenerator:
    tsg = oms.TheoreticalSpectrumGenerator()
    params = tsg.getParameters()
    for ion in ("a", "b", "c", "x", "y", "z"):
        params.setValue(f"add_{ion}_ions", "true" if ion in ION_TYPES else "false")
    params.setValue("add_metainfo", "true")
    tsg.setParameters(params)
    return tsg


def theoretical_ions(
    tsg: oms.TheoreticalSpectrumGenerator, sequence: str, charge: int
) -> tuple[np.ndarray, list[str]]:
    """Return m/z values and ion names (e.g. ``y3++``) of a peptide's fragment ladder."""
    try:
        aa_seq = oms.AASequence.fromString(sequence)
    except Exception:
        return np.empty(0), []
    if aa_seq.size() < 2:
        return np.empty(0), []
    max_charge = max(1, min(MAX_FRAGMENT_CHARGE, int(charge or 1) - 1))
    spec = oms.MSSpectrum()
    tsg.getSpectrum(spec, aa_seq, 1, max_charge)
    mz, _ = spec.get_peaks()
    names: list[str] = []
    for arr in spec.getStringDataArrays():
        if arr.getName() == "IonNames":
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in arr]
            break
    if len(names) != len(mz):
        names = [""] * len(mz)
    return np.asarray(mz, dtype=float), names


def match_ions(
    exp_mz: np.ndarray,
    theo_mz: np.ndarray,
    tolerance: float,
    tolerance_ppm: bool,
) -> np.ndarray:
    """For each theoretical m/z, the index of the nearest experimental peak in tolerance, else -1.

    ``exp_mz`` must be sorted ascending.
    """
    result = np.full(len(theo_mz), -1, dtype=np.int64)
    if len(exp_mz) == 0 or len(theo_mz) == 0:
        return result
    pos = np.searchsorted(exp_mz, theo_mz)
    left = np.clip(pos - 1, 0, len(exp_mz) - 1)
    right = np.clip(pos, 0, len(exp_mz) - 1)
    nearest = np.where(
        np.abs(exp_mz[left] - theo_mz) <= np.abs(exp_mz[right] - theo_mz), left, right
    )
    diff = np.abs(exp_mz[nearest] - theo_mz)
    allowed = theo_mz * tolerance * 1e-6 if tolerance_ppm else np.full(len(theo_mz), tolerance)
    hit = diff <= allowed
    result[hit] = nearest[hit]
    return result


def build_mirror_data(
    id_df: pl.DataFrame,
    peaks: pl.DataFrame,
    tolerance: float = 0.05,
    tolerance_ppm: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build mirror rows and per-ID match summary.

    Args:
        id_df: Identifications with id_idx, sequence, charge and scan_id (linked).
        peaks: Peaks with scan_id, mass (m/z) and intensity.
        tolerance: Fragment mass tolerance.
        tolerance_ppm: Whether the tolerance is in ppm.
    """
    linked = id_df.filter(pl.col("scan_id") > 0).select("id_idx", "sequence", "charge", "scan_id")
    if linked.height == 0:
        return pl.DataFrame(schema=MIRROR_SCHEMA), pl.DataFrame(schema=SUMMARY_SCHEMA)

    scan_ids = linked["scan_id"].unique().to_list()
    scan_peaks = (
        peaks.filter(pl.col("scan_id").is_in(scan_ids))
        .select("scan_id", "mass", "intensity")
        .sort("scan_id", "mass")
        .partition_by("scan_id", as_dict=True)
    )

    tsg = _make_generator()
    columns: dict[str, list] = {key: [] for key in MIRROR_SCHEMA}
    summary: dict[str, list] = {key: [] for key in SUMMARY_SCHEMA}
    next_peak_id = 0

    for id_idx, sequence, charge, scan_id in linked.iter_rows():
        exp = scan_peaks.get((scan_id,))
        exp_mz = exp["mass"].to_numpy().astype(float) if exp is not None else np.empty(0)
        exp_int = exp["intensity"].to_numpy().astype(float) if exp is not None else np.empty(0)
        theo_mz, theo_names = theoretical_ions(tsg, sequence, charge)
        hits = match_ions(exp_mz, theo_mz, tolerance, tolerance_ppm)

        exp_labels: list[list[str]] = [[] for _ in range(len(exp_mz))]
        for theo_idx, exp_idx in enumerate(hits):
            if exp_idx >= 0:
                exp_labels[exp_idx].append(theo_names[theo_idx])

        n_exp, n_theo = len(exp_mz), len(theo_mz)
        base_peak = float(exp_int.max()) if n_exp else 1.0

        columns["mirror_peak_id"].extend(range(next_peak_id, next_peak_id + n_exp + n_theo))
        next_peak_id += n_exp + n_theo
        columns["mirror_top_id"].extend([id_idx] * n_exp + [-1] * n_theo)
        columns["mirror_bottom_id"].extend([-1] * n_exp + [id_idx] * n_theo)
        columns["mz"].extend(exp_mz.tolist() + theo_mz.tolist())
        columns["intensity"].extend(exp_int.tolist() + [base_peak] * n_theo)
        columns["matched"].extend([bool(lbls) for lbls in exp_labels] + (hits >= 0).tolist())
        columns["label"].extend(
            [" / ".join(lbls) for lbls in exp_labels]
            + [name if hit >= 0 else "" for name, hit in zip(theo_names, hits)]
        )

        matched = int((hits >= 0).sum())
        summary["id_idx"].append(id_idx)
        summary["matched_ions"].append(matched)
        summary["total_ions"].append(n_theo)
        summary["coverage"].append(100.0 * matched / n_theo if n_theo else 0.0)

    return (
        pl.DataFrame(columns, schema=MIRROR_SCHEMA),
        pl.DataFrame(summary, schema=SUMMARY_SCHEMA),
    )


def ensure_mirror_input(
    paths: dict,
    id_paths: dict,
    search_params: Optional[dict] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> tuple[dict, bool]:
    """Write the mirror cache if it is missing or older than the IDs.

    Returns the mirror paths and whether the files were (re)written.
    """
    mirror_paths = get_mirror_paths(paths)
    if mirror_cache_is_valid(mirror_paths, id_paths["identifications"]):
        return mirror_paths, False

    if status_callback:
        status_callback("Generating theoretical spectra...")
    params = search_params or {}
    id_df = pl.read_parquet(id_paths["identifications"])
    peaks = pl.read_parquet(paths["spectrum_plot"], columns=["scan_id", "mass", "intensity"])
    mirror_df, summary_df = build_mirror_data(
        id_df,
        peaks,
        tolerance=float(params.get("fragment_mass_tolerance", 0.05)),
        tolerance_ppm=bool(params.get("fragment_mass_tolerance_ppm", False)),
    )
    mirror_df.write_parquet(mirror_paths["mirror"])
    summary_df.write_parquet(mirror_paths["summary"])
    return mirror_paths, True
