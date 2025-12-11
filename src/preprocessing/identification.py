"""idXML extraction and spectrum linking for identification data.

This module handles loading identification data (idXML files) and linking
identifications to spectra based on retention time and precursor m/z matching.

Uses pyOpenMS TheoreticalSpectrumGenerator for accurate fragment mass calculation
and extracts external peak annotations when available from the idXML file.
"""

from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any
import json

import numpy as np
import polars as pl
from pyopenms import (
    IdXMLFile,
    AASequence,
    TheoreticalSpectrumGenerator,
    MSSpectrum,
    Param,
)


# Default matching tolerances
DEFAULT_RT_TOLERANCE = 5.0  # seconds
DEFAULT_MZ_TOLERANCE = 0.5  # Daltons


def get_id_cache_paths(workspace: Path, idxml_path: Path) -> dict:
    """Get cache file paths for identification data.

    Cache structure:
    {workspace}/.cache/{file_stem}/
    └── identifications/
        ├── identifications.parquet
        ├── fragment_masses.parquet  (pre-computed for all sequences)
        ├── peak_annotations.parquet (external annotations from idXML)
        ├── search_params.json       (search engine parameters)
        └── id_info.json
    """
    cache_dir = Path(workspace) / ".cache" / idxml_path.stem
    id_dir = cache_dir / "identifications"

    return {
        "identifications": id_dir / "identifications.parquet",
        "fragment_masses": id_dir / "fragment_masses.parquet",
        "peak_annotations": id_dir / "peak_annotations.parquet",
        "search_params": id_dir / "search_params.json",
        "id_info": id_dir / "id_info.json",
    }


def id_cache_is_valid(idxml_path: Path, paths: dict) -> bool:
    """Check if identification cache files exist and are newer than the idXML file."""
    id_pq = paths["identifications"]
    frag_pq = paths["fragment_masses"]
    id_info = paths["id_info"]

    if not id_pq.exists() or not frag_pq.exists() or not id_info.exists():
        return False

    idxml_mtime = idxml_path.stat().st_mtime
    return (
        id_pq.stat().st_mtime > idxml_mtime
        and frag_pq.stat().st_mtime > idxml_mtime
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


def calculate_fragment_masses_tsg(
    sequence_str: str,
    max_charge: int = 1,
    ion_types: Optional[List[str]] = None,
) -> Dict[str, List[List[float]]]:
    """Calculate theoretical fragment masses using TheoreticalSpectrumGenerator.

    This uses pyOpenMS's TheoreticalSpectrumGenerator which properly handles:
    - Modifications (including variable and fixed)
    - Multiple charge states
    - All ion types (a, b, c, x, y, z)

    Args:
        sequence_str: Peptide sequence string (can include modifications)
        max_charge: Maximum charge state to consider (default 1 for neutral masses)
        ion_types: List of ion types to generate. Default: ['a', 'b', 'c', 'x', 'y', 'z']

    Returns:
        Dict with fragment_masses_a, fragment_masses_b, etc.
        Each is a list of lists (to support multiple charge states or ambiguous mods).
    """
    if ion_types is None:
        ion_types = ['a', 'b', 'c', 'x', 'y', 'z']

    try:
        aa_seq = AASequence.fromString(sequence_str)
        n = aa_seq.size()

        # Configure TheoreticalSpectrumGenerator
        tsg = TheoreticalSpectrumGenerator()
        params = tsg.getParameters()

        # Enable requested ion types
        params.setValue("add_a_ions", "true" if 'a' in ion_types else "false")
        params.setValue("add_b_ions", "true" if 'b' in ion_types else "false")
        params.setValue("add_c_ions", "true" if 'c' in ion_types else "false")
        params.setValue("add_x_ions", "true" if 'x' in ion_types else "false")
        params.setValue("add_y_ions", "true" if 'y' in ion_types else "false")
        params.setValue("add_z_ions", "true" if 'z' in ion_types else "false")
        params.setValue("add_metainfo", "true")  # Needed for ion names

        tsg.setParameters(params)

        # Generate spectrum for charge 1 (neutral masses)
        spec = MSSpectrum()
        tsg.getSpectrum(spec, aa_seq, 1, max_charge)

        # Initialize result dict
        result = {f'fragment_masses_{ion}': [[] for _ in range(n)] for ion in ion_types}

        # Get ion names from StringDataArrays
        ion_names = []
        sdas = spec.getStringDataArrays()
        for sda in sdas:
            if sda.getName() == "IonNames":
                for i in range(sda.size()):
                    # Values are bytes, decode to string
                    name = sda[i]
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    ion_names.append(name)
                break

        # Parse generated peaks and organize by ion type and position
        for i in range(spec.size()):
            peak = spec[i]
            mz = peak.getMZ()

            # Get ion annotation from StringDataArray
            ion_name = ion_names[i] if i < len(ion_names) else ""

            if not ion_name:
                continue

            # Parse ion name (e.g., "b3+", "y5++", "a2+")
            # Format: {ion_type}{number}{charge_suffix}
            ion_type = None
            ion_number = None

            for t in ion_types:
                if ion_name.lower().startswith(t):
                    ion_type = t
                    # Extract number after ion type letter
                    try:
                        num_str = ""
                        for c in ion_name[1:]:
                            if c.isdigit():
                                num_str += c
                            else:
                                break
                        if num_str:
                            ion_number = int(num_str)
                    except (ValueError, IndexError):
                        pass
                    break

            if ion_type and ion_number and 1 <= ion_number <= n:
                # Store mass at the appropriate position
                # For prefix ions (a, b, c): index = ion_number - 1
                # For suffix ions (x, y, z): index = ion_number - 1
                idx = ion_number - 1
                key = f'fragment_masses_{ion_type}'
                if idx < len(result[key]):
                    result[key][idx].append(mz)

        # Ensure each position has at least an empty list or single mass
        # Convert empty lists to contain at least one value if we have any data
        for ion_type in ion_types:
            key = f'fragment_masses_{ion_type}'
            for i in range(n):
                if not result[key][i]:
                    # Try to compute fallback using AASequence directly
                    result[key][i] = []

        return result

    except Exception as e:
        print(f"Error calculating fragments with TSG for {sequence_str}: {e}")
        # Return empty structure
        return {f'fragment_masses_{ion}': [] for ion in (ion_types or ['a', 'b', 'c', 'x', 'y', 'z'])}


def calculate_fragment_masses(sequence_str: str) -> dict:
    """Calculate theoretical fragment masses for a peptide using pyOpenMS.

    This is a wrapper that uses TheoreticalSpectrumGenerator for accurate masses.
    Falls back to direct AASequence calculation if TSG fails.

    Returns:
        Dict with fragment_masses_a, fragment_masses_b, etc.
        Each is a list of lists (to support ambiguous modifications).
    """
    # First try TheoreticalSpectrumGenerator
    result = calculate_fragment_masses_tsg(sequence_str)

    # Check if we got valid results
    has_data = any(
        any(masses for masses in result.get(f'fragment_masses_{ion}', []))
        for ion in ['a', 'b', 'c', 'x', 'y', 'z']
    )

    if has_data:
        return result

    # Fallback: use direct AASequence calculation (original method)
    try:
        from pyopenms import Residue
        aa_seq = AASequence.fromString(sequence_str)
        n = aa_seq.size()

        fallback_result = {}

        # Prefix ions (a, b, c)
        for ion_type, res_type in [('a', Residue.ResidueType.AIon),
                                    ('b', Residue.ResidueType.BIon),
                                    ('c', Residue.ResidueType.CIon)]:
            masses = []
            for i in range(n):
                prefix = aa_seq.getPrefix(i + 1)
                mass = prefix.getMonoWeight(res_type, 0)
                masses.append([mass])
            fallback_result[f'fragment_masses_{ion_type}'] = masses

        # Suffix ions (x, y, z)
        for ion_type, res_type in [('x', Residue.ResidueType.XIon),
                                    ('y', Residue.ResidueType.YIon),
                                    ('z', Residue.ResidueType.ZIon)]:
            masses = []
            for i in range(n):
                suffix = aa_seq.getSuffix(i + 1)
                mass = suffix.getMonoWeight(res_type, 0)
                masses.append([mass])
            fallback_result[f'fragment_masses_{ion_type}'] = masses

        return fallback_result

    except Exception as e:
        print(f"Error calculating fragments for {sequence_str}: {e}")
        return {
            'fragment_masses_a': [],
            'fragment_masses_b': [],
            'fragment_masses_c': [],
            'fragment_masses_x': [],
            'fragment_masses_y': [],
            'fragment_masses_z': [],
        }


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


def extract_peak_annotations(peptide_hit, id_idx: int) -> List[Dict[str, Any]]:
    """Extract peak annotations from a PeptideHit using getPeakAnnotations() API.

    Args:
        peptide_hit: pyOpenMS PeptideHit object
        id_idx: Index of the identification this hit belongs to

    Returns:
        List of dicts with: id_idx, mz, annotation, charge, ion_type
    """
    annotations = []

    try:
        peak_annotations = peptide_hit.getPeakAnnotations()

        if not peak_annotations:
            return annotations

        for peak_ann in peak_annotations:
            ann_mz = peak_ann.mz
            ion_name = str(peak_ann.annotation) if hasattr(peak_ann, 'annotation') else ""
            charge = peak_ann.charge if hasattr(peak_ann, 'charge') else 1

            # Determine ion type from annotation name
            ion_type = "unknown"
            ion_name_lower = ion_name.lower()
            for t in ['a', 'b', 'c', 'x', 'y', 'z']:
                if ion_name_lower.startswith(t):
                    ion_type = t
                    break

            annotations.append({
                "id_idx": id_idx,
                "mz": ann_mz,
                "annotation": ion_name,
                "charge": charge,
                "ion_type": ion_type,
            })

    except Exception as e:
        # getPeakAnnotations() may not be available or may fail
        pass

    return annotations


def extract_idxml_to_parquet(
    idxml_path: Path,
    paths: dict,
    status_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Extract idXML identification data to parquet files.

    Creates files:
    1. identifications.parquet - ID metadata (sequence, score, RT, etc.)
    2. fragment_masses.parquet - Pre-computed fragment masses (using TSG)
    3. peak_annotations.parquet - External peak annotations from search engine
    4. search_params.json - Search parameters (tolerances, mods, enzyme)
    5. id_info.json - Summary info

    Args:
        idxml_path: Path to the idXML file
        paths: Dictionary of cache paths from get_id_cache_paths()
        status_callback: Optional callback for progress updates
    """
    if status_callback:
        status_callback("Loading idXML file...")

    protein_ids = []
    peptide_ids = []
    IdXMLFile().load(str(idxml_path), protein_ids, peptide_ids)

    if status_callback:
        status_callback(f"Processing {len(peptide_ids)} peptide identifications...")

    # Extract search parameters
    search_params = extract_search_parameters(protein_ids)

    # Extract identification data and peak annotations
    id_data = []
    all_peak_annotations = []
    unique_sequences = set()
    has_external_annotations = False

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

        # Extract external peak annotations
        peak_anns = extract_peak_annotations(best_hit, idx)
        if peak_anns:
            has_external_annotations = True
            all_peak_annotations.extend(peak_anns)

    # Pre-compute fragment masses for all unique sequences using TSG
    if status_callback:
        status_callback(f"Computing fragment masses for {len(unique_sequences)} unique sequences...")

    fragment_data = []
    for i, sequence in enumerate(unique_sequences):
        if status_callback and (i + 1) % 100 == 0:
            status_callback(f"Computing fragments: {i + 1}/{len(unique_sequences)}")

        frag_masses = calculate_fragment_masses(sequence)
        # Store as JSON strings for the list columns
        fragment_data.append({
            "sequence": sequence,
            "theoretical_mass": calculate_peptide_mass(sequence),
            "fragment_masses_a": json.dumps(frag_masses.get("fragment_masses_a", [])),
            "fragment_masses_b": json.dumps(frag_masses.get("fragment_masses_b", [])),
            "fragment_masses_c": json.dumps(frag_masses.get("fragment_masses_c", [])),
            "fragment_masses_x": json.dumps(frag_masses.get("fragment_masses_x", [])),
            "fragment_masses_y": json.dumps(frag_masses.get("fragment_masses_y", [])),
            "fragment_masses_z": json.dumps(frag_masses.get("fragment_masses_z", [])),
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

    # Write fragment masses DataFrame
    if fragment_data:
        frag_df = pl.DataFrame(fragment_data)
        frag_df.write_parquet(paths["fragment_masses"])
    else:
        pl.DataFrame({
            "sequence": pl.Series([], dtype=pl.Utf8),
            "theoretical_mass": pl.Series([], dtype=pl.Float64),
            "fragment_masses_a": pl.Series([], dtype=pl.Utf8),
            "fragment_masses_b": pl.Series([], dtype=pl.Utf8),
            "fragment_masses_c": pl.Series([], dtype=pl.Utf8),
            "fragment_masses_x": pl.Series([], dtype=pl.Utf8),
            "fragment_masses_y": pl.Series([], dtype=pl.Utf8),
            "fragment_masses_z": pl.Series([], dtype=pl.Utf8),
        }).write_parquet(paths["fragment_masses"])

    # Write peak annotations DataFrame
    if all_peak_annotations:
        ann_df = pl.DataFrame(all_peak_annotations)
        ann_df.write_parquet(paths["peak_annotations"])
    else:
        pl.DataFrame({
            "id_idx": pl.Series([], dtype=pl.Int64),
            "mz": pl.Series([], dtype=pl.Float64),
            "annotation": pl.Series([], dtype=pl.Utf8),
            "charge": pl.Series([], dtype=pl.Int32),
            "ion_type": pl.Series([], dtype=pl.Utf8),
        }).write_parquet(paths["peak_annotations"])

    # Write search parameters
    with open(paths["search_params"], "w") as f:
        json.dump(search_params, f, indent=2)

    # Write identification info
    id_info = {
        "num_identifications": len(id_data),
        "num_unique_sequences": len(unique_sequences),
        "num_proteins": len(protein_ids),
        "has_external_annotations": has_external_annotations,
        "num_annotations": len(all_peak_annotations),
        "search_engine": search_params.get("search_engine", ""),
    }
    with open(paths["id_info"], "w") as f:
        json.dump(id_info, f, indent=2)

    if status_callback:
        msg = f"Extracted {len(id_data)} identifications ({len(unique_sequences)} unique sequences)"
        if has_external_annotations:
            msg += f", {len(all_peak_annotations)} peak annotations"
        status_callback(msg)


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


def load_peak_annotations(paths: dict, id_idx: int) -> Optional[pl.DataFrame]:
    """Load external peak annotations for a specific identification.

    Args:
        paths: Cache paths dict
        id_idx: Identification index to filter by

    Returns:
        DataFrame with peak annotations for this ID, or None if not available
    """
    ann_path = paths.get("peak_annotations")
    if not ann_path or not Path(ann_path).exists():
        return None

    try:
        ann_df = pl.read_parquet(ann_path)
        if ann_df.height == 0:
            return None

        filtered = ann_df.filter(pl.col("id_idx") == id_idx)
        if filtered.height == 0:
            return None

        return filtered
    except Exception:
        return None


def _parse_openms_sequence(sequence_str: str) -> Tuple[List[str], List[Optional[float]]]:
    """Parse OpenMS sequence format to extract residues and modification mass shifts.

    Converts e.g. 'SHC(Carbamidomethyl)IAEVEK' to:
    - residues: ['S', 'H', 'C', 'I', 'A', 'E', 'V', 'E', 'K']
    - modifications: [None, None, 57.02, None, None, None, None, None, None]

    Args:
        sequence_str: Peptide sequence in OpenMS format with modifications in parentheses

    Returns:
        Tuple of (residues list, modifications list where None means unmodified)
    """
    try:
        aa_seq = AASequence.fromString(sequence_str)
        residues = []
        modifications = []

        for i in range(aa_seq.size()):
            residue = aa_seq.getResidue(i)
            one_letter = residue.getOneLetterCode()
            residues.append(one_letter)

            mod = residue.getModification()
            if mod:
                diff_mono = mod.getDiffMonoMass()
                modifications.append(round(diff_mono, 2))
            else:
                modifications.append(None)

        return residues, modifications
    except Exception:
        # On any error, fallback to extracting single letters
        residues = []
        modifications = []
        i = 0
        while i < len(sequence_str):
            if sequence_str[i].isupper():
                residues.append(sequence_str[i])
                modifications.append(None)
                i += 1
            elif sequence_str[i] == '(':
                end = sequence_str.find(')', i)
                if end > i:
                    i = end + 1
                else:
                    i += 1
            else:
                i += 1
        return residues, modifications


def get_sequence_data_for_identification(
    id_row: dict,
    peaks_df: pl.DataFrame,
    fragment_masses_df: Optional[pl.DataFrame] = None,
    peak_annotations_df: Optional[pl.DataFrame] = None,
    search_params: Optional[Dict[str, Any]] = None,
) -> Tuple[dict, List[float], float]:
    """Get sequence data and observed masses for a specific identification.

    Args:
        id_row: Dictionary with identification data (from id_df.row())
        peaks_df: DataFrame with all peaks data
        fragment_masses_df: Optional pre-computed fragment masses DataFrame.
            If provided, looks up cached masses instead of computing.
        peak_annotations_df: Optional external peak annotations DataFrame.
            If provided, includes external annotations in the result.
        search_params: Optional search parameters dict.

    Returns:
        Tuple of (sequence_data dict, observed_masses list, precursor_mass)

    The sequence_data dict contains:
        - sequence: List of amino acid characters
        - modifications: List of mass shifts per position (None for unmodified)
        - theoretical_mass: Peptide mass
        - fixed_modifications: List of modifications
        - fragment_masses_[a,b,c,x,y,z]: Pre-computed fragment masses
        - external_annotations: List of external peak annotations (if available)
        - fragment_tolerance: Tolerance value from search params
        - fragment_tolerance_ppm: Whether tolerance is in ppm
    """
    sequence = id_row["sequence"]
    scan_id = id_row["scan_id"]
    id_idx = id_row.get("id_idx", -1)
    precursor_mass = id_row.get("theoretical_mass", 0.0)

    # Parse sequence to extract residues and modification mass shifts
    residues, modifications = _parse_openms_sequence(sequence)

    # Try to get pre-computed fragment masses from cache
    if fragment_masses_df is not None:
        cached_row = fragment_masses_df.filter(pl.col("sequence") == sequence)
        if cached_row.height > 0:
            row = cached_row.row(0, named=True)
            sequence_data = {
                "sequence": residues,
                "modifications": modifications,
                "theoretical_mass": row.get("theoretical_mass", precursor_mass),
                "fixed_modifications": [],
                "fragment_masses_a": json.loads(row["fragment_masses_a"]),
                "fragment_masses_b": json.loads(row["fragment_masses_b"]),
                "fragment_masses_c": json.loads(row["fragment_masses_c"]),
                "fragment_masses_x": json.loads(row["fragment_masses_x"]),
                "fragment_masses_y": json.loads(row["fragment_masses_y"]),
                "fragment_masses_z": json.loads(row["fragment_masses_z"]),
            }
        else:
            # Fallback: compute if not in cache
            fragment_data = calculate_fragment_masses(sequence)
            sequence_data = {
                "sequence": residues,
                "modifications": modifications,
                "theoretical_mass": precursor_mass,
                "fixed_modifications": [],
                **fragment_data,
            }
    else:
        # No cache provided, compute fragment masses
        fragment_data = calculate_fragment_masses(sequence)
        sequence_data = {
            "sequence": residues,
            "modifications": modifications,
            "theoretical_mass": precursor_mass,
            "fixed_modifications": [],
            **fragment_data,
        }

    # Add external peak annotations if available
    if peak_annotations_df is not None and id_idx >= 0:
        id_anns = peak_annotations_df.filter(pl.col("id_idx") == id_idx)
        if id_anns.height > 0:
            sequence_data["external_annotations"] = id_anns.to_dicts()

    # Add search parameters if available
    if search_params:
        sequence_data["fragment_tolerance"] = search_params.get("fragment_mass_tolerance", 0.05)
        sequence_data["fragment_tolerance_ppm"] = search_params.get("fragment_mass_tolerance_ppm", False)

    # Get observed masses from peaks
    observed_masses = []
    if scan_id > 0:
        spectrum_peaks = peaks_df.filter(pl.col("scan_id") == scan_id)
        if spectrum_peaks.height > 0:
            observed_masses = spectrum_peaks["mass"].to_list()

    return sequence_data, observed_masses, id_row.get("precursor_mz", 0.0)


PROTON_MASS = 1.007276  # Daltons


def compute_peak_annotations_for_spectrum(
    peaks_df: pl.DataFrame,
    scan_id: int,
    sequence_data: dict,
    precursor_charge: int = 2,
    tolerance: float = 20.0,
    tolerance_ppm: bool = True,
    ion_types: Optional[List[str]] = None,
) -> Dict[float, Dict[str, Any]]:
    """Compute fragment ion annotations for peaks in a spectrum.

    Matches observed peaks against theoretical fragment masses considering
    charge states from 1 to precursor_charge.

    Args:
        peaks_df: DataFrame with all peaks (must have scan_id, mass, intensity columns)
        scan_id: Scan ID to filter peaks for
        sequence_data: Dict with fragment_masses_[a,b,c,x,y,z] (from get_sequence_data_for_identification)
        precursor_charge: Maximum charge state to consider
        tolerance: Mass tolerance for matching
        tolerance_ppm: If True, tolerance is in ppm; if False, in Daltons
        ion_types: List of ion types to consider (default: ['b', 'y'])

    Returns:
        Dict mapping m/z values to annotation data:
        {mz_value: {'highlight': True, 'annotation': 'b3¹⁺'}, ...}
        Can be passed directly to LinePlot.set_dynamic_annotations()
    """
    if ion_types is None:
        ion_types = ['b', 'y']  # Most common ion types

    # Filter to spectrum peaks
    spectrum_peaks = peaks_df.filter(pl.col("scan_id") == scan_id)
    if spectrum_peaks.height == 0:
        return {}

    # Get observed m/z values
    observed_mz = spectrum_peaks["mass"].to_numpy()

    # Superscript digits for charge display
    superscript = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                   '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}

    def to_superscript(n: int) -> str:
        return ''.join(superscript.get(d, d) for d in str(n))

    # Initialize result dict: mz -> {highlight, annotation}
    annotations_dict: Dict[float, Dict[str, Any]] = {}

    # Build list of theoretical fragments with their labels
    # Each entry: (theoretical_mz, label, ion_type, ion_number)
    theoretical_fragments = []

    for ion_type in ion_types:
        key = f"fragment_masses_{ion_type}"
        fragment_list = sequence_data.get(key, [])

        for ion_number, masses in enumerate(fragment_list, start=1):
            if not masses:
                continue

            # masses is a list (can have multiple for ambiguous mods)
            for neutral_mass in masses:
                if neutral_mass <= 0:
                    continue

                # Generate m/z for each charge state
                for charge in range(1, precursor_charge + 1):
                    theoretical_mz = (neutral_mass + charge * PROTON_MASS) / charge
                    charge_str = to_superscript(charge) + "⁺"
                    label = f"{ion_type}{ion_number}{charge_str}"
                    theoretical_fragments.append((theoretical_mz, label, ion_type, ion_number))

    # Match observed peaks to theoretical fragments
    for frag_mz, label, ion_type, ion_number in theoretical_fragments:
        # Find peaks within tolerance
        if tolerance_ppm:
            tol_da = frag_mz * tolerance / 1e6
        else:
            tol_da = tolerance

        for obs_mz in observed_mz:
            mass_diff = abs(obs_mz - frag_mz)
            if mass_diff <= tol_da:
                # Use float as key
                mz_key = float(obs_mz)
                if mz_key not in annotations_dict:
                    annotations_dict[mz_key] = {'highlight': True, 'annotation': label}
                else:
                    # Peak already matched - append this annotation if different
                    existing = annotations_dict[mz_key]['annotation']
                    if label not in existing:
                        annotations_dict[mz_key]['annotation'] = f"{existing}/{label}"

    return annotations_dict


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
