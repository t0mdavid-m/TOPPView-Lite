"""Unit tests for src/preprocessing/identification.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import polars as pl

from src.preprocessing.identification import (
    get_id_cache_paths,
    id_cache_is_valid,
    link_identifications_to_spectra,
    load_search_params,
    find_matching_idxml,
    DEFAULT_RT_TOLERANCE,
    DEFAULT_MZ_TOLERANCE,
)


class TestGetIdCachePaths:
    """Tests for get_id_cache_paths function."""

    def test_returns_correct_structure(self, temp_workspace):
        """Test that all expected paths are returned."""
        idxml_path = Path("/path/to/sample.idXML")
        paths = get_id_cache_paths(temp_workspace, idxml_path)

        assert "identifications" in paths
        assert "search_params" in paths
        assert "id_info" in paths

    def test_paths_include_file_stem(self, temp_workspace):
        """Test that paths are organized by file stem."""
        idxml_path = Path("/path/to/my_sample.idXML")
        paths = get_id_cache_paths(temp_workspace, idxml_path)

        assert "my_sample" in str(paths["identifications"])
        assert paths["identifications"].suffix == ".parquet"
        assert paths["search_params"].suffix == ".json"
        assert paths["id_info"].suffix == ".json"

    def test_paths_in_identifications_subdir(self, temp_workspace):
        """Test that paths are in identifications subdirectory."""
        idxml_path = Path("/path/to/sample.idXML")
        paths = get_id_cache_paths(temp_workspace, idxml_path)

        assert "identifications" in str(paths["identifications"].parent.name)


class TestIdCacheIsValid:
    """Tests for id_cache_is_valid function."""

    def test_returns_false_when_files_missing(self, temp_workspace):
        """Test that missing files return False."""
        idxml_path = temp_workspace / "idXML-files" / "test.idXML"
        idxml_path.parent.mkdir(exist_ok=True)
        idxml_path.touch()

        paths = get_id_cache_paths(temp_workspace, idxml_path)

        assert id_cache_is_valid(idxml_path, paths) is False

    def test_returns_true_when_cache_newer(self, temp_workspace):
        """Test that newer cache files return True."""
        import time

        idxml_path = temp_workspace / "idXML-files" / "test.idXML"
        idxml_path.parent.mkdir(exist_ok=True)
        idxml_path.touch()

        paths = get_id_cache_paths(temp_workspace, idxml_path)
        paths["identifications"].parent.mkdir(parents=True, exist_ok=True)

        # Sleep briefly to ensure mtime difference
        time.sleep(0.01)

        # Create cache files (newer than idxml)
        pl.DataFrame({"id_idx": [0]}).write_parquet(paths["identifications"])
        with open(paths["id_info"], "w") as f:
            json.dump({"num_identifications": 1}, f)

        assert id_cache_is_valid(idxml_path, paths) is True

    def test_returns_false_when_cache_older(self, temp_workspace):
        """Test that older cache files return False."""
        import time

        idxml_path = temp_workspace / "idXML-files" / "test.idXML"
        idxml_path.parent.mkdir(exist_ok=True)

        paths = get_id_cache_paths(temp_workspace, idxml_path)
        paths["identifications"].parent.mkdir(parents=True, exist_ok=True)

        # Create cache files first
        pl.DataFrame({"id_idx": [0]}).write_parquet(paths["identifications"])
        with open(paths["id_info"], "w") as f:
            json.dump({"num_identifications": 1}, f)

        # Sleep and then touch idxml (makes it newer)
        time.sleep(0.01)
        idxml_path.touch()

        assert id_cache_is_valid(idxml_path, paths) is False


class TestLinkIdentificationsToSpectra:
    """Tests for link_identifications_to_spectra function."""

    def test_links_matching_spectra(self, sample_metadata_df, sample_id_df):
        """Test that identifications are linked to correct spectra."""
        # ID RT is in seconds, metadata RT is in minutes
        # ID 0: rt=120s (2min), mz=500.5 -> should match scan_id=2 (rt=2min, mz=500.5)
        # ID 1: rt=240s (4min), mz=600.6 -> should match scan_id=4 (rt=4min, mz=600.6)

        # Reset scan_id to test linking
        id_df = sample_id_df.with_columns(pl.lit(-1).alias("scan_id"))

        result = link_identifications_to_spectra(
            id_df,
            sample_metadata_df,
            rt_tolerance=DEFAULT_RT_TOLERANCE,
            mz_tolerance=DEFAULT_MZ_TOLERANCE,
        )

        # Check that IDs were linked to correct scans
        result_dict = {
            row["id_idx"]: row["scan_id"]
            for row in result.iter_rows(named=True)
        }

        assert result_dict[0] == 2  # First ID matches scan 2
        assert result_dict[1] == 4  # Second ID matches scan 4
        assert result_dict[2] == 5  # Third ID matches scan 5

    def test_no_match_returns_negative_one(self, sample_metadata_df):
        """Test that unmatched IDs get scan_id=-1."""
        id_df = pl.DataFrame({
            "id_idx": [0],
            "rt": [9999.0],  # Way outside tolerance
            "precursor_mz": [9999.0],
            "sequence": ["TEST"],
            "scan_id": [-1],
        })

        result = link_identifications_to_spectra(
            id_df,
            sample_metadata_df,
            rt_tolerance=DEFAULT_RT_TOLERANCE,
            mz_tolerance=DEFAULT_MZ_TOLERANCE,
        )

        assert result["scan_id"][0] == -1

    def test_handles_empty_ms2_spectra(self):
        """Test handling when no MS2 spectra exist."""
        metadata_df = pl.DataFrame({
            "scan_id": [1, 2, 3],
            "retention_time": [1.0, 2.0, 3.0],
            "ms_level": [1, 1, 1],  # All MS1
            "precursor_mz": [0.0, 0.0, 0.0],
        })

        id_df = pl.DataFrame({
            "id_idx": [0],
            "rt": [60.0],
            "precursor_mz": [500.0],
            "scan_id": [-1],
        })

        result = link_identifications_to_spectra(id_df, metadata_df)

        # Should return original DataFrame unchanged
        assert result["scan_id"][0] == -1

    def test_status_callback_called(self, sample_metadata_df):
        """Test that status callback is invoked."""
        id_df = pl.DataFrame({
            "id_idx": [0],
            "rt": [120.0],
            "precursor_mz": [500.5],
            "scan_id": [-1],
        })

        callback = MagicMock()
        link_identifications_to_spectra(
            id_df,
            sample_metadata_df,
            status_callback=callback
        )

        assert callback.call_count >= 1


class TestLoadSearchParams:
    """Tests for load_search_params function."""

    def test_loads_existing_params(self, temp_cache_paths):
        """Test loading existing search parameters."""
        params = {
            "fragment_mass_tolerance": 0.02,
            "fragment_mass_tolerance_ppm": True,
            "enzyme": "Trypsin",
        }

        temp_cache_paths["search_params"].parent.mkdir(parents=True, exist_ok=True)
        with open(temp_cache_paths["search_params"], "w") as f:
            json.dump(params, f)

        result = load_search_params(temp_cache_paths)

        assert result["fragment_mass_tolerance"] == 0.02
        assert result["fragment_mass_tolerance_ppm"] is True
        assert result["enzyme"] == "Trypsin"

    def test_returns_defaults_when_missing(self, temp_cache_paths):
        """Test that defaults are returned for missing file."""
        result = load_search_params(temp_cache_paths)

        assert "fragment_mass_tolerance" in result
        assert "fragment_mass_tolerance_ppm" in result
        assert result["fragment_mass_tolerance"] == 0.05
        assert result["fragment_mass_tolerance_ppm"] is False

    def test_handles_invalid_json(self, temp_cache_paths):
        """Test handling of invalid JSON file."""
        temp_cache_paths["search_params"].parent.mkdir(parents=True, exist_ok=True)
        with open(temp_cache_paths["search_params"], "w") as f:
            f.write("not valid json {{{")

        result = load_search_params(temp_cache_paths)

        # Should return defaults on error
        assert result["fragment_mass_tolerance"] == 0.05


class TestFindMatchingIdxml:
    """Tests for find_matching_idxml function."""

    def test_finds_exact_match(self, temp_workspace):
        """Test finding idXML with exact name match."""
        mzml_path = temp_workspace / "mzML-files" / "sample.mzML"
        idxml_path = temp_workspace / "idXML-files" / "sample.idXML"

        mzml_path.touch()
        idxml_path.touch()

        result = find_matching_idxml(mzml_path, temp_workspace)

        assert result == idxml_path

    def test_returns_none_when_no_match(self, temp_workspace):
        """Test returning None when no matching idXML exists."""
        mzml_path = temp_workspace / "mzML-files" / "sample.mzML"

        result = find_matching_idxml(mzml_path, temp_workspace)

        assert result is None

    def test_returns_none_when_no_idxml_dir(self, temp_workspace):
        """Test returning None when idXML directory doesn't exist."""
        # Remove idXML directory
        (temp_workspace / "idXML-files").rmdir()

        mzml_path = temp_workspace / "mzML-files" / "sample.mzML"

        result = find_matching_idxml(mzml_path, temp_workspace)

        assert result is None

    def test_case_insensitive_match(self, temp_workspace):
        """Test case-insensitive matching."""
        mzml_path = temp_workspace / "mzML-files" / "SAMPLE.mzML"
        idxml_path = temp_workspace / "idXML-files" / "sample.idXML"

        mzml_path.touch()
        idxml_path.touch()

        result = find_matching_idxml(mzml_path, temp_workspace)

        assert result == idxml_path

    def test_matches_with_suffix_removed(self, temp_workspace):
        """Test matching when mzML has common suffixes."""
        mzml_path = temp_workspace / "mzML-files" / "sample_indexed.mzML"
        idxml_path = temp_workspace / "idXML-files" / "sample.idXML"

        mzml_path.touch()
        idxml_path.touch()

        result = find_matching_idxml(mzml_path, temp_workspace)

        assert result == idxml_path
