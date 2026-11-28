"""Unit tests for src/preprocessing/component_inputs.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import polars as pl

from src.preprocessing.component_inputs import (
    load_im_info,
    component_cache_is_valid,
    create_component_inputs,
    create_spectra_table_with_ids,
    ensure_spectrum_summary,
    SPECTRUM_SUMMARY_COLUMNS,
    spectra_table_is_current,
)


def _current_spectra_table() -> pl.DataFrame:
    """A spectra table with every column a current cache has."""
    return pl.DataFrame({"scan_id": [1], **{c: [1.0] for c in SPECTRUM_SUMMARY_COLUMNS}})


class TestLoadImInfo:
    """Tests for load_im_info function."""

    def test_loads_existing_info(self, temp_cache_paths):
        """Test loading existing ion mobility info."""
        im_info = {
            "type": "faims",
            "unit": " V",
            "unique_values": [-35.0, -45.0],
        }

        with open(temp_cache_paths["im_info"], "w") as f:
            json.dump(im_info, f)

        result = load_im_info(temp_cache_paths)

        assert result["type"] == "faims"
        assert result["unit"] == " V"
        assert result["unique_values"] == [-35.0, -45.0]

    def test_returns_none_type_when_missing(self, temp_cache_paths):
        """Test that missing file returns type=none."""
        # Don't create the file
        result = load_im_info(temp_cache_paths)

        assert result["type"] == "none"
        assert result["unit"] == ""


class TestComponentCacheIsValid:
    """Tests for component_cache_is_valid function."""

    def test_returns_false_when_raw_missing(self, temp_cache_paths):
        """Test that missing raw files return False."""
        # Don't create any files
        assert component_cache_is_valid(temp_cache_paths) is False

    def test_returns_false_when_component_files_missing(self, populated_cache):
        """Test that missing component files return False."""
        # populated_cache has raw files but no component files
        assert component_cache_is_valid(populated_cache) is False

    def test_returns_true_when_all_valid(self, populated_cache):
        """Test that valid cache returns True."""
        import time

        # Sleep to ensure component files are newer
        time.sleep(0.01)

        # Create component files (newer than raw)
        _current_spectra_table().write_parquet(populated_cache["spectra_table"])
        pl.DataFrame({"peak_id": [1]}).write_parquet(populated_cache["peaks_table"])
        pl.DataFrame({"peak_id": [1]}).write_parquet(populated_cache["spectrum_plot"])
        pl.DataFrame({"peak_id": [1]}).write_parquet(populated_cache["heatmap_input"])

        assert component_cache_is_valid(populated_cache) is True

    def test_returns_false_when_im_table_missing_but_needed(self, temp_cache_paths):
        """Test that missing IM table returns False when IM data present."""
        import time

        # Write raw data
        metadata_df = pl.DataFrame({
            "scan_id": [1], "name": ["s1"], "retention_time": [1.0],
            "ms_level": [1], "precursor_mz": [0.0], "charge": [0],
            "num_peaks": [10], "im_id": [0],
        })
        peaks_df = pl.DataFrame({
            "peak_id": [1], "scan_id": [1], "mass": [100.0],
            "intensity": [1000.0], "retention_time": [1.0], "im_id": [0],
        })

        metadata_df.write_parquet(temp_cache_paths["metadata"])
        peaks_df.write_parquet(temp_cache_paths["peaks"])

        # Write IM info indicating FAIMS data
        with open(temp_cache_paths["im_info"], "w") as f:
            json.dump({"type": "faims", "unit": " V", "unique_values": [-35.0]}, f)

        time.sleep(0.01)

        # Create component files except im_table
        _current_spectra_table().write_parquet(temp_cache_paths["spectra_table"])
        pl.DataFrame({"peak_id": [1]}).write_parquet(temp_cache_paths["peaks_table"])
        pl.DataFrame({"peak_id": [1]}).write_parquet(temp_cache_paths["spectrum_plot"])
        pl.DataFrame({"peak_id": [1]}).write_parquet(temp_cache_paths["heatmap_input"])
        # Don't create im_table

        assert component_cache_is_valid(temp_cache_paths) is False


class TestCreateComponentInputs:
    """Tests for create_component_inputs function."""

    def test_creates_all_component_files(self, populated_cache):
        """Test that all component files are created."""
        create_component_inputs(populated_cache)

        assert populated_cache["spectra_table"].exists()
        assert populated_cache["peaks_table"].exists()
        assert populated_cache["spectrum_plot"].exists()
        assert populated_cache["heatmap_input"].exists()

    def test_spectra_table_has_correct_columns(self, populated_cache):
        """Test spectra table has expected columns."""
        create_component_inputs(populated_cache)

        df = pl.read_parquet(populated_cache["spectra_table"])

        expected_cols = ["scan_id", "name", "retention_time", "ms_level",
                        "precursor_mz", "charge", "num_peaks"]
        for col in expected_cols:
            assert col in df.columns

    def test_spectra_table_summarises_each_spectrum(self, temp_cache_paths):
        """TIC, base peak and m/z range come from the spectrum's peaks above zero."""
        pl.DataFrame({
            "scan_id": [1, 2], "name": ["s1", "s2"], "retention_time": [1.0, 2.0],
            "ms_level": [1, 2], "precursor_mz": [0.0, 500.0], "charge": [0, 2],
            "num_peaks": [3, 0],
        }).write_parquet(temp_cache_paths["metadata"])
        pl.DataFrame({
            "peak_id": [1, 2, 3], "scan_id": [1, 1, 1], "mass": [100.0, 250.5, 900.0],
            "intensity": [10.0, 30.0, 0.0], "retention_time": [1.0, 1.0, 1.0],
        }).write_parquet(temp_cache_paths["peaks"])

        create_component_inputs(temp_cache_paths)
        df = pl.read_parquet(temp_cache_paths["spectra_table"]).sort("scan_id")

        first = df.row(0, named=True)
        assert first["tic"] == 40.0
        assert first["base_peak_mz"] == 250.5
        assert first["base_peak_intensity"] == 30.0
        # The zero-intensity peak at 900 is not part of the range
        assert (first["mz_min"], first["mz_max"]) == (100.0, 250.5)
        # A spectrum without peaks keeps its row, with empty summary values
        second = df.row(1, named=True)
        assert second["scan_id"] == 2 and second["tic"] is None
        assert spectra_table_is_current(temp_cache_paths["spectra_table"])

    def test_older_caches_get_the_summary_columns(self, populated_cache):
        """Spectra tables written before the summary columns existed get them in place."""
        create_component_inputs(populated_cache)
        current = pl.read_parquet(populated_cache["spectra_table"]).sort("scan_id")
        old = current.drop(SPECTRUM_SUMMARY_COLUMNS)
        old.write_parquet(populated_cache["spectra_table"])
        old.with_columns(pl.lit(1).alias("num_ids")).write_parquet(
            populated_cache["spectra_table_with_ids"]
        )
        assert not spectra_table_is_current(populated_cache["spectra_table"])

        ensure_spectrum_summary(populated_cache)

        for key in ("spectra_table", "spectra_table_with_ids"):
            upgraded = pl.read_parquet(populated_cache[key]).sort("scan_id")
            assert upgraded.select(SPECTRUM_SUMMARY_COLUMNS).equals(
                current.select(SPECTRUM_SUMMARY_COLUMNS)
            )
        assert "num_ids" in pl.read_parquet(populated_cache["spectra_table_with_ids"]).columns
        # The component cache stays valid: nothing has to be processed again
        assert component_cache_is_valid(populated_cache) is True

    def test_faims_spectra_get_their_cv(self, populated_cache, sample_im_info_faims):
        """FAIMS files: each spectrum's compensation voltage, also added to older caches."""
        import json

        pl.read_parquet(populated_cache["metadata"]).with_columns(
            pl.Series("im_id", [0, 1, 2, 0, -1])
        ).write_parquet(populated_cache["metadata"])
        pl.read_parquet(populated_cache["peaks"]).with_columns(
            pl.lit(0).alias("im_id")
        ).write_parquet(populated_cache["peaks"])
        with open(populated_cache["im_info"], "w") as f:
            json.dump(sample_im_info_faims, f)

        create_component_inputs(populated_cache)
        df = pl.read_parquet(populated_cache["spectra_table"]).sort("scan_id")
        assert df["cv"].to_list() == [-35.0, -45.0, -55.0, -35.0, None]
        assert spectra_table_is_current(populated_cache["spectra_table"], sample_im_info_faims)

        df.drop("cv").write_parquet(populated_cache["spectra_table"])
        assert not spectra_table_is_current(populated_cache["spectra_table"], sample_im_info_faims)
        ensure_spectrum_summary(populated_cache)
        upgraded = pl.read_parquet(populated_cache["spectra_table"]).sort("scan_id")
        assert upgraded["cv"].to_list() == [-35.0, -45.0, -55.0, -35.0, None]

    def test_peaks_table_filters_zero_intensity(self, populated_cache, sample_peaks_df):
        """Test that zero intensity peaks are filtered."""
        # Add a zero intensity peak
        peaks_with_zero = pl.concat([
            sample_peaks_df,
            pl.DataFrame({
                "peak_id": [100],
                "scan_id": [1],
                "mass": [999.0],
                "intensity": [0.0],
                "retention_time": [1.0],
            })
        ])
        peaks_with_zero.write_parquet(populated_cache["peaks"])

        create_component_inputs(populated_cache)

        df = pl.read_parquet(populated_cache["peaks_table"])

        # Zero intensity peak should be filtered out
        assert df.filter(pl.col("intensity") == 0).height == 0
        assert df.filter(pl.col("peak_id") == 100).height == 0

    def test_heatmap_has_retention_time(self, populated_cache):
        """Test heatmap input has retention_time column."""
        create_component_inputs(populated_cache)

        df = pl.read_parquet(populated_cache["heatmap_input"])

        assert "retention_time" in df.columns
        assert "mass" in df.columns
        assert "intensity" in df.columns

    def test_status_callback_called(self, populated_cache):
        """Test that status callback is invoked."""
        callback = MagicMock()

        create_component_inputs(populated_cache, status_callback=callback)

        assert callback.call_count >= 1

    def test_creates_im_table_when_faims_data(self, temp_cache_paths):
        """Test that IM table is created for FAIMS data."""
        # Create raw data with IM
        metadata_df = pl.DataFrame({
            "scan_id": [1, 2],
            "name": ["s1", "s2"],
            "retention_time": [1.0, 2.0],
            "ms_level": [2, 2],
            "precursor_mz": [500.0, 600.0],
            "charge": [2, 2],
            "num_peaks": [10, 20],
            "im_id": [0, 1],
        })

        peaks_df = pl.DataFrame({
            "peak_id": [0, 1, 2, 3],
            "scan_id": [1, 1, 2, 2],
            "mass": [100.0, 200.0, 300.0, 400.0],
            "intensity": [1000.0, 2000.0, 1500.0, 500.0],
            "retention_time": [1.0, 1.0, 2.0, 2.0],
            "im_id": [0, 0, 1, 1],
        })

        metadata_df.write_parquet(temp_cache_paths["metadata"])
        peaks_df.write_parquet(temp_cache_paths["peaks"])

        with open(temp_cache_paths["im_info"], "w") as f:
            json.dump({
                "type": "faims",
                "unit": " V",
                "unique_values": [-35.0, -45.0],
            }, f)

        create_component_inputs(temp_cache_paths)

        assert temp_cache_paths["im_table"].exists()
        im_df = pl.read_parquet(temp_cache_paths["im_table"])
        assert "im_id" in im_df.columns
        assert "im_label" in im_df.columns
        assert "num_spectra" in im_df.columns


class TestCreateSpectraTableWithIds:
    """Tests for create_spectra_table_with_ids function."""

    def test_creates_num_ids_column(self, populated_cache, sample_id_df, sample_im_info_none):
        """Test that num_ids column is added."""
        # First create base spectra table
        create_component_inputs(populated_cache)

        create_spectra_table_with_ids(
            populated_cache,
            sample_id_df,
            sample_im_info_none
        )

        df = pl.read_parquet(populated_cache["spectra_table_with_ids"])

        assert "num_ids" in df.columns

    def test_counts_ids_correctly(self, populated_cache, sample_im_info_none):
        """Test that ID counts are correct."""
        # First create base spectra table
        create_component_inputs(populated_cache)

        # Create ID data with multiple IDs for same scan
        id_df = pl.DataFrame({
            "id_idx": [0, 1, 2, 3],
            "scan_id": [2, 2, 4, -1],  # scan 2 has 2 IDs, scan 4 has 1, one unmatched
            "sequence": ["A", "B", "C", "D"],
        })

        create_spectra_table_with_ids(
            populated_cache,
            id_df,
            sample_im_info_none
        )

        df = pl.read_parquet(populated_cache["spectra_table_with_ids"])

        scan_2_ids = df.filter(pl.col("scan_id") == 2)["num_ids"][0]
        scan_4_ids = df.filter(pl.col("scan_id") == 4)["num_ids"][0]
        scan_1_ids = df.filter(pl.col("scan_id") == 1)["num_ids"][0]

        assert scan_2_ids == 2
        assert scan_4_ids == 1
        assert scan_1_ids == 0

    def test_unmatched_ids_not_counted(self, populated_cache, sample_im_info_none):
        """Test that IDs with scan_id=-1 are not counted."""
        create_component_inputs(populated_cache)

        id_df = pl.DataFrame({
            "id_idx": [0, 1],
            "scan_id": [-1, -1],  # All unmatched
            "sequence": ["A", "B"],
        })

        create_spectra_table_with_ids(
            populated_cache,
            id_df,
            sample_im_info_none
        )

        df = pl.read_parquet(populated_cache["spectra_table_with_ids"])

        # All spectra should have 0 IDs
        assert df["num_ids"].sum() == 0

    def test_first_identification_of_each_spectrum(self, populated_cache, sample_im_info_none):
        """Sequence and score come from the first identification (idXML order) of a spectrum."""
        create_component_inputs(populated_cache)
        id_df = pl.DataFrame({
            "id_idx": [3, 0, 1, 2],
            "scan_id": [2, 2, 4, -1],
            "sequence": ["LATER", "FIRST", "OTHER", "UNMATCHED"],
            "score": [0.5, 0.9, 12.0, 1.0],
        })

        create_spectra_table_with_ids(populated_cache, id_df, sample_im_info_none)

        df = pl.read_parquet(populated_cache["spectra_table_with_ids"]).sort("scan_id")
        rows = {r["scan_id"]: r for r in df.iter_rows(named=True)}
        assert (rows[2]["sequence"], rows[2]["score"]) == ("FIRST", 0.9)
        assert (rows[4]["sequence"], rows[4]["score"]) == ("OTHER", 12.0)
        assert rows[1]["sequence"] is None and rows[1]["score"] is None
        assert spectra_table_is_current(
            populated_cache["spectra_table_with_ids"], sample_im_info_none, with_ids=True
        )
        # A table written before these columns existed is not current
        assert not spectra_table_is_current(
            populated_cache["spectra_table"], sample_im_info_none, with_ids=True
        )

    def test_status_callback_called(self, populated_cache, sample_id_df, sample_im_info_none):
        """Test that status callback is invoked."""
        create_component_inputs(populated_cache)

        callback = MagicMock()
        create_spectra_table_with_ids(
            populated_cache,
            sample_id_df,
            sample_im_info_none,
            status_callback=callback
        )

        assert callback.call_count >= 1


class TestChromatogramInput:
    """Tests for the MS1 TIC/BPC input."""

    def test_tic_and_bpc_per_ms1_spectrum(self, populated_cache):
        from src.preprocessing.component_inputs import create_chromatogram_input

        create_chromatogram_input(populated_cache)
        df = pl.read_parquet(populated_cache["chromatogram"])

        # Only MS1 spectra (scans 1 and 3), sorted by retention time
        assert df["scan_id"].to_list() == [1, 3]
        assert df["tic"].to_list() == [4500.0, 1800.0]
        assert df["bpc"].to_list() == [2000.0, 1200.0]
        assert "im_id" not in df.columns

    def test_ms1_without_peaks_gets_zero(self, populated_cache, sample_peaks_df):
        from src.preprocessing.component_inputs import create_chromatogram_input

        sample_peaks_df.filter(pl.col("scan_id") != 3).write_parquet(
            populated_cache["peaks"]
        )
        create_chromatogram_input(populated_cache)
        df = pl.read_parquet(populated_cache["chromatogram"])

        assert df.filter(pl.col("scan_id") == 3)["tic"].to_list() == [0.0]

    def test_ensure_builds_missing_input_for_old_caches(self, populated_cache):
        from src.preprocessing.component_inputs import ensure_chromatogram_input

        assert not populated_cache["chromatogram"].exists()
        ensure_chromatogram_input(populated_cache)
        assert populated_cache["chromatogram"].exists()

    def test_create_component_inputs_writes_chromatogram(self, populated_cache):
        create_component_inputs(populated_cache)
        assert populated_cache["chromatogram"].exists()
