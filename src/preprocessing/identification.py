"""idXML extraction and spectrum linking for identification data.

This module handles loading identification data (idXML files) and linking
identifications to spectra based on retention time and precursor m/z matching.

Fragment matching and annotation is now handled by openms_insight.SequenceView
in the Vue frontend, so this module no longer precomputes fragment annotations.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any
import json

import numpy as np
import polars as pl
from pyopenms import IdXMLFile, AASequence, PeptideIdentificationList


# Default matching tolerances
DEFAULT_RT_TOLERANCE = 5.0  # seconds
DEFAULT_MZ_TOLERANCE = 0.5  # Daltons


def get_id_cache_paths(workspace: Path, idxml_path: Path) -> dict:
    """Get cache file paths for identification data.

    Cache structure:
    {workspace}/.cache/{file_stem}/
    └── identifications/
        ├── identifications.parquet
        ├── search_params.json
        └── id_info.json
    """
    cache_dir = Path(workspace) / ".cache" / idxml_path.stem
    id_dir = cache_dir / "identifications"

    return {
        "identifications": id_dir / "identifications.parquet",
        "search_params": id_dir / "search_params.json",
        "id_info": id_dir / "id_info.json",
    }


def id_cache_is_valid(idxml_path: Path, paths: dict) -> bool:
    """Check if identification cache files exist and are newer than the idXML file."""
    id_pq = paths["identifications"]
    id_info = paths["id_info"]

    if not id_pq.exists() or not id_info.exists():
        return False

    idxml_mtime = idxml_path.stat().st_mtime
    return (
        id_pq.stat().st_mtime > idxml_mtime
        and id_info.stat().st_mtime > idxml_mtime
    )


def sequence_to_mass_shift_format(sequence_str: str) -> str:
    """Convert sequence with named modifications to mass shift format.

    Converts e.g. 'SHC(Carbamidomethyl)IAEVEK' to 'SHC[+57.02]IAEVEK'.

    Args:
        sequence_str: Peptide sequence with named modifications (OpenMS format)

    Returns:
        Sequence with modifications shown as mass shifts [+X.XX] or [-X.XX]
    """
    try:
        aa_seq = AASequence.fromString(sequence_str)
        result = []

        for i in range(aa_seq.size()):
            residue = aa_seq.getResidue(i)
            one_letter = residue.getOneLetterCode()
            result.append(one_letter)

            mod = residue.getModification()
            if mod:
                diff_mono = mod.getDiffMonoMass()
                if diff_mono >= 0:
                    result.append(f"[+{diff_mono:.2f}]")
                else:
                    result.append(f"[{diff_mono:.2f}]")

        return "".join(result)
    except Exception:
        return sequence_str


def calculate_peptide_mass(sequence_str: str) -> float:
    """Calculate monoisotopic mass of a peptide sequence using pyOpenMS."""
    try:
        aa_seq = AASequence.fromString(sequence_str)
        return aa_seq.getMonoWeight()
    except Exception:
        return 0.0


def extract_search_parameters(protein_ids: list) -> Dict[str, Any]:
    """Extract search parameters from ProteinIdentification objects.

    Returns dict with:
    - search_engine: Name of the search engine
    - search_engine_version: Version string
    - fragment_mass_tolerance: Tolerance value
    - fragment_mass_tolerance_ppm: True if ppm, False if Da
    - precursor_mass_tolerance: Tolerance value
    - precursor_mass_tolerance_ppm: True if ppm, False if Da
    - fixed_modifications: List of fixed modifications
    - variable_modifications: List of variable modifications
    - enzyme: Digestion enzyme name
    - missed_cleavages: Number of allowed missed cleavages
    """
    params = {
        "search_engine": "",
        "search_engine_version": "",
        "fragment_mass_tolerance": 0.05,  # Default 0.05 Da
        "fragment_mass_tolerance_ppm": False,
        "precursor_mass_tolerance": 10.0,
        "precursor_mass_tolerance_ppm": True,
        "fixed_modifications": [],
        "variable_modifications": [],
        "enzyme": "",
        "missed_cleavages": 1,
    }

    if not protein_ids:
        return params

    def _to_str(val):
        """Convert bytes to str if needed."""
        if isinstance(val, bytes):
            return val.decode('utf-8')
        return str(val) if val is not None else ""

    try:
        prot_id = protein_ids[0]

        # Search engine info
        params["search_engine"] = _to_str(prot_id.getSearchEngine())
        params["search_engine_version"] = _to_str(prot_id.getSearchEngineVersion())

        # Get SearchParameters object
        sp = prot_id.getSearchParameters()

        # Tolerances
        params["fragment_mass_tolerance"] = float(sp.fragment_mass_tolerance)
        params["fragment_mass_tolerance_ppm"] = bool(sp.fragment_mass_tolerance_ppm)
        params["precursor_mass_tolerance"] = float(sp.precursor_mass_tolerance)
        params["precursor_mass_tolerance_ppm"] = bool(sp.precursor_mass_tolerance_ppm)

        # Modifications - convert each to string
        params["fixed_modifications"] = [_to_str(m) for m in sp.fixed_modifications]
        params["variable_modifications"] = [_to_str(m) for m in sp.variable_modifications]

        # Enzyme info
        if sp.digestion_enzyme:
            params["enzyme"] = _to_str(sp.digestion_enzyme.getName())
        params["missed_cleavages"] = int(sp.missed_cleavages)

    except Exception as e:
        print(f"Warning: Could not extract search parameters: {e}")

    return params


def extract_idxml_to_parquet(
    idxml_path: Path,
    paths: dict,
    status_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Extract idXML identification data to parquet files.

    Creates files:
    1. identifications.parquet - ID metadata (sequence, score, RT, etc.)
    2. search_params.json - Search parameters (tolerances, mods, enzyme)
    3. id_info.json - Summary info

    Note: Fragment mass calculation is now handled by SequenceView in Vue,
    so this function no longer precomputes fragment_masses.parquet.

    Args:
        idxml_path: Path to the idXML file
        paths: Dictionary of cache paths from get_id_cache_paths()
        status_callback: Optional callback for progress updates
    """
    if status_callback:
        status_callback("Loading idXML file...")

    protein_ids = []
    peptide_ids = PeptideIdentificationList()  # pyOpenMS 3.5+ requires this type
    # Use absolute path to avoid issues with relative path resolution
    IdXMLFile().load(str(Path(idxml_path).resolve()), protein_ids, peptide_ids)

    if status_callback:
        status_callback(f"Processing {len(peptide_ids)} peptide identifications...")

    # Extract search parameters
    search_params = extract_search_parameters(protein_ids)

    # Extract identification data
    id_data = []
    unique_sequences = set()

    for idx, pep_id in enumerate(peptide_ids):
        rt = pep_id.getRT()  # Already in seconds
        mz = pep_id.getMZ()

        # Get best hit (first hit after sorting by score)
        hits = pep_id.getHits()
        if not hits:
            continue

        best_hit = hits[0]
        sequence = best_hit.getSequence().toString()
        sequence_display = sequence_to_mass_shift_format(sequence)
        charge = best_hit.getCharge()
        score = best_hit.getScore()
        unique_sequences.add(sequence)

        # Extract protein accession if available
        protein_accessions = []
        for evidence in best_hit.getPeptideEvidences():
            protein_accessions.append(evidence.getProteinAccession())

        # Calculate theoretical mass
        theoretical_mass = calculate_peptide_mass(sequence)

        id_data.append({
            "id_idx": idx,
            "rt": rt,
            "precursor_mz": mz,
            "sequence": sequence,
            "sequence_display": sequence_display,
            "charge": charge,
            "score": score,
            "theoretical_mass": theoretical_mass,
            "protein_accession": ";".join(protein_accessions) if protein_accessions else "",
            "scan_id": -1,  # Will be filled by linking
        })

    if status_callback:
        status_callback("Writing parquet files...")

    # Create directories
    paths["identifications"].parent.mkdir(parents=True, exist_ok=True)

    # Write identifications DataFrame
    if id_data:
        id_df = pl.DataFrame(id_data)
        id_df.write_parquet(paths["identifications"])
    else:
        # Empty DataFrame with schema
        pl.DataFrame({
            "id_idx": pl.Series([], dtype=pl.Int64),
            "rt": pl.Series([], dtype=pl.Float64),
            "precursor_mz": pl.Series([], dtype=pl.Float64),
            "sequence": pl.Series([], dtype=pl.Utf8),
            "sequence_display": pl.Series([], dtype=pl.Utf8),
            "charge": pl.Series([], dtype=pl.Int32),
            "score": pl.Series([], dtype=pl.Float64),
            "theoretical_mass": pl.Series([], dtype=pl.Float64),
            "protein_accession": pl.Series([], dtype=pl.Utf8),
            "scan_id": pl.Series([], dtype=pl.Int32),
        }).write_parquet(paths["identifications"])

    # Write search parameters
    with open(paths["search_params"], "w") as f:
        json.dump(search_params, f, indent=2)

    # Write identification info
    id_info = {
        "num_identifications": len(id_data),
        "num_unique_sequences": len(unique_sequences),
        "num_proteins": len(protein_ids),
        "search_engine": search_params.get("search_engine", ""),
    }
    with open(paths["id_info"], "w") as f:
        json.dump(id_info, f, indent=2)

    if status_callback:
        status_callback(f"Extracted {len(id_data)} identifications ({len(unique_sequences)} unique sequences)")


def link_identifications_to_spectra(
    id_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    rt_tolerance: float = DEFAULT_RT_TOLERANCE,
    mz_tolerance: float = DEFAULT_MZ_TOLERANCE,
    status_callback: Optional[Callable[[str], None]] = None
) -> pl.DataFrame:
    """Link identifications to spectra based on RT and precursor m/z matching.

    For each identification, finds the best matching MS2 spectrum based on:
    - Retention time within rt_tolerance (seconds)
    - Precursor m/z within mz_tolerance (Daltons)
    - Best match selected by minimum RT difference

    Args:
        id_df: DataFrame with identification data
        metadata_df: DataFrame with spectrum metadata
        rt_tolerance: RT matching tolerance in seconds
        mz_tolerance: m/z matching tolerance in Daltons
        status_callback: Optional callback for progress updates

    Returns:
        DataFrame with scan_id column filled for matched IDs
    """
    if status_callback:
        status_callback("Linking identifications to spectra...")

    # Filter to MS2 spectra only
    ms2_df = metadata_df.filter(pl.col("ms_level") == 2)

    if ms2_df.height == 0:
        return id_df

    # Convert to numpy for faster matching
    id_rts = id_df["rt"].to_numpy()
    id_mzs = id_df["precursor_mz"].to_numpy()

    # MS2 metadata (convert RT from minutes back to seconds for matching)
    ms2_rts = ms2_df["retention_time"].to_numpy() * 60.0
    ms2_mzs = ms2_df["precursor_mz"].to_numpy()
    ms2_scan_ids = ms2_df["scan_id"].to_numpy()

    # Match each ID to best spectrum
    matched_scan_ids = []
    for i in range(len(id_rts)):
        id_rt = id_rts[i]
        id_mz = id_mzs[i]

        # Find spectra within tolerance
        rt_diff = np.abs(ms2_rts - id_rt)
        mz_diff = np.abs(ms2_mzs - id_mz)

        in_tolerance = (rt_diff <= rt_tolerance) & (mz_diff <= mz_tolerance)

        if np.any(in_tolerance):
            # Select best match by minimum RT difference
            valid_indices = np.where(in_tolerance)[0]
            best_idx = valid_indices[np.argmin(rt_diff[valid_indices])]
            matched_scan_ids.append(int(ms2_scan_ids[best_idx]))
        else:
            matched_scan_ids.append(-1)

    # Update DataFrame
    result = id_df.with_columns(pl.Series("scan_id", matched_scan_ids))

    if status_callback:
        num_matched = sum(1 for s in matched_scan_ids if s > 0)
        status_callback(f"Linked {num_matched}/{len(matched_scan_ids)} identifications")

    return result


def load_search_params(paths: dict) -> Dict[str, Any]:
    """Load search parameters from cache.

    Returns default values if file doesn't exist.
    """
    search_params_path = paths.get("search_params")
    if search_params_path and Path(search_params_path).exists():
        try:
            with open(search_params_path) as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "fragment_mass_tolerance": 0.05,
        "fragment_mass_tolerance_ppm": False,
    }


def find_matching_idxml(mzml_path: Path, workspace: Path) -> Optional[Path]:
    """Find matching idXML file for an mzML file based on filename.

    Looks for idXML files in the workspace's idXML-files directory
    that match the mzML filename stem.

    Args:
        mzml_path: Path to the mzML file
        workspace: Workspace path

    Returns:
        Path to matching idXML file, or None if not found
    """
    idxml_dir = workspace / "idXML-files"
    if not idxml_dir.exists():
        return None

    mzml_stem = mzml_path.stem

    # Try exact match first
    exact_match = idxml_dir / f"{mzml_stem}.idXML"
    if exact_match.exists():
        return exact_match

    # Try case-insensitive match
    for idxml_file in idxml_dir.glob("*.idXML"):
        if idxml_file.stem.lower() == mzml_stem.lower():
            return idxml_file

    # Try with common suffixes removed
    for suffix in ["_indexed", "_centroided", "_filtered"]:
        if mzml_stem.endswith(suffix):
            base_stem = mzml_stem[: -len(suffix)]
            for idxml_file in idxml_dir.glob("*.idXML"):
                if idxml_file.stem.lower() == base_stem.lower():
                    return idxml_file

    return None
