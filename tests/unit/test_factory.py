"""Unit tests for src/components/factory.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import polars as pl


class TestCreateMsComponents:
    """Tests for create_ms_components function."""

    @patch('src.components.factory.Table')
    @patch('src.components.factory.LinePlot')
    @patch('src.components.factory.Heatmap')
    def test_returns_all_expected_components(self, mock_heatmap, mock_lineplot, mock_table, populated_cache, sample_im_info_none):
        """Test that all MS components are returned."""
        from src.components.factory import create_ms_components

        # Create component input files
        from src.preprocessing.component_inputs import create_component_inputs
        create_component_inputs(populated_cache)

        result = create_ms_components(
            populated_cache,
            sample_im_info_none,
            file_id="test"
        )

        assert "im_table" in result
        assert "spectra_table" in result
        assert "peaks_table" in result
        assert "spectrum_plot" in result
        assert "heatmap" in result

    @patch('src.components.factory.Table')
    @patch('src.components.factory.LinePlot')
    @patch('src.components.factory.Heatmap')
    def test_im_table_is_none_without_im_data(self, mock_heatmap, mock_lineplot, mock_table, populated_cache, sample_im_info_none):
        """Test that im_table is None when no IM data present."""
        from src.components.factory import create_ms_components

        # Create component input files
        from src.preprocessing.component_inputs import create_component_inputs
        create_component_inputs(populated_cache)

        result = create_ms_components(
            populated_cache,
            sample_im_info_none,
            file_id="test"
        )

        assert result["im_table"] is None

    @patch('src.components.factory.Table')
    @patch('src.components.factory.LinePlot')
    @patch('src.components.factory.Heatmap')
    def test_creates_table_with_correct_cache_id(self, mock_heatmap, mock_lineplot, mock_table, populated_cache, sample_im_info_none):
        """Test that Table is created with correct cache_id prefix."""
        from src.components.factory import create_ms_components

        # Create component input files
        from src.preprocessing.component_inputs import create_component_inputs
        create_component_inputs(populated_cache)

        create_ms_components(
            populated_cache,
            sample_im_info_none,
            file_id="myfile"
        )

        # Check that Table was called with cache_id starting with prefix
        table_calls = mock_table.call_args_list
        cache_ids = [call.kwargs.get('cache_id', '') for call in table_calls]

        assert any("myfile_spectra_table" in cid for cid in cache_ids)
        assert any("myfile_peaks_table" in cid for cid in cache_ids)

    @patch('src.components.factory.Table')
    @patch('src.components.factory.LinePlot')
    @patch('src.components.factory.Heatmap')
    def test_heatmap_uses_data_path(self, mock_heatmap, mock_lineplot, mock_table, populated_cache, sample_im_info_none):
        """Test that Heatmap is created with data_path for subprocess preprocessing."""
        from src.components.factory import create_ms_components

        # Create component input files
        from src.preprocessing.component_inputs import create_component_inputs
        create_component_inputs(populated_cache)

        create_ms_components(
            populated_cache,
            sample_im_info_none,
            file_id="test"
        )

        # Check Heatmap was called with data_path
        heatmap_call = mock_heatmap.call_args
        assert 'data_path' in heatmap_call.kwargs


class TestCreateIdComponents:
    """Tests for create_id_components function."""

    @patch('src.components.factory.Table')
    @patch('src.components.factory.SequenceView')
    @patch('src.components.factory.LinePlot')
    def test_returns_all_expected_components(self, mock_lineplot, mock_sv, mock_table, populated_cache, sample_im_info_none):
        """Test that all ID components are returned."""
        from src.components.factory import create_id_components

        # Create required files
        from src.preprocessing.component_inputs import create_component_inputs
        create_component_inputs(populated_cache)

        # Create mock id_paths
        id_dir = populated_cache["component_cache"].parent / "identifications"
        id_dir.mkdir(parents=True, exist_ok=True)

        id_paths = {
            "identifications": id_dir / "identifications.parquet",
            "search_params": id_dir / "search_params.json",
        }

        # Create test ID data
        id_df = pl.DataFrame({
            "id_idx": [0, 1],
            "sequence": ["PEPTIDE", "SEQUENCE"],
            "charge": [2, 3],
            "score": [0.95, 0.88],
            "scan_id": [1, 2],
        })
        id_df.write_parquet(id_paths["identifications"])

        # Mock LinePlot.from_sequence_view
        mock_lineplot.from_sequence_view = MagicMock(return_value=MagicMock())

        result = create_id_components(
            populated_cache,
            id_paths,
            sample_im_info_none,
            file_id="test",
            search_params=None
        )

        assert "spectra_table_with_ids" in result
        assert "id_table" in result
        assert "sequence_view" in result
        assert "annotated_spectrum_plot" in result

    @patch('src.components.factory.Table')
    @patch('src.components.factory.SequenceView')
    @patch('src.components.factory.LinePlot')
    def test_uses_search_params_for_annotation_config(self, mock_lineplot, mock_sv, mock_table, populated_cache, sample_im_info_none):
        """Test that search parameters are used for annotation config."""
        from src.components.factory import create_id_components

        # Create required files
        from src.preprocessing.component_inputs import create_component_inputs
        create_component_inputs(populated_cache)

        id_dir = populated_cache["component_cache"].parent / "identifications"
        id_dir.mkdir(parents=True, exist_ok=True)

        id_paths = {
            "identifications": id_dir / "identifications.parquet",
        }

        id_df = pl.DataFrame({
            "id_idx": [0],
            "sequence": ["PEPTIDE"],
            "charge": [2],
            "score": [0.95],
            "scan_id": [1],
        })
        id_df.write_parquet(id_paths["identifications"])

        mock_lineplot.from_sequence_view = MagicMock(return_value=MagicMock())

        search_params = {
            "fragment_mass_tolerance": 15.0,
            "fragment_mass_tolerance_ppm": True,
        }

        create_id_components(
            populated_cache,
            id_paths,
            sample_im_info_none,
            file_id="test",
            search_params=search_params
        )

        # Check SequenceView was called with annotation_config
        sv_call = mock_sv.call_args
        annotation_config = sv_call.kwargs.get('annotation_config', {})

        assert annotation_config.get('tolerance') == 15.0
        assert annotation_config.get('tolerance_ppm') is True


class TestReconstructMsComponents:
    """Tests for reconstruct_ms_components function."""

    @patch('src.components.factory.Table')
    @patch('src.components.factory.LinePlot')
    @patch('src.components.factory.Heatmap')
    def test_reconstructs_with_cache_only(self, mock_heatmap, mock_lineplot, mock_table):
        """Test that components are reconstructed with only cache_id and cache_path."""
        from src.components.factory import reconstruct_ms_components

        result = reconstruct_ms_components(
            cache_path="/path/to/cache",
            file_id="test",
            has_im=False
        )

        assert "spectra_table" in result
        assert "peaks_table" in result
        assert "spectrum_plot" in result
        assert "heatmap" in result
        assert result["im_table"] is None

        # Verify Table was called with only cache_id and cache_path
        table_calls = mock_table.call_args_list
        for call in table_calls:
            kwargs = call.kwargs
            assert 'cache_id' in kwargs
            assert 'cache_path' in kwargs
            # Should NOT have data or data_path (reconstruction mode)
            assert 'data' not in kwargs or kwargs.get('data') is None

    @patch('src.components.factory.Table')
    @patch('src.components.factory.LinePlot')
    @patch('src.components.factory.Heatmap')
    def test_reconstructs_im_table_when_has_im(self, mock_heatmap, mock_lineplot, mock_table):
        """Test that IM table is reconstructed when has_im=True."""
        from src.components.factory import reconstruct_ms_components

        result = reconstruct_ms_components(
            cache_path="/path/to/cache",
            file_id="test",
            has_im=True
        )

        assert result["im_table"] is not None


class TestReconstructIdComponents:
    """Tests for reconstruct_id_components function."""

    @patch('src.components.factory.Table')
    @patch('src.components.factory.SequenceView')
    @patch('src.components.factory.LinePlot')
    def test_reconstructs_all_id_components(self, mock_lineplot, mock_sv, mock_table):
        """Test that all ID components are reconstructed."""
        from src.components.factory import reconstruct_id_components

        result = reconstruct_id_components(
            cache_path="/path/to/cache",
            file_id="test"
        )

        assert "spectra_table_with_ids" in result
        assert "id_table" in result
        assert "sequence_view" in result
        assert "annotated_spectrum_plot" in result

    @patch('src.components.factory.Table')
    @patch('src.components.factory.SequenceView')
    @patch('src.components.factory.LinePlot')
    def test_uses_correct_cache_ids(self, mock_lineplot, mock_sv, mock_table):
        """Test that correct cache IDs are used with prefix."""
        from src.components.factory import reconstruct_id_components

        reconstruct_id_components(
            cache_path="/path/to/cache",
            file_id="myfile"
        )

        # Check Table calls have correct cache_ids
        table_calls = mock_table.call_args_list
        cache_ids = [call.kwargs.get('cache_id', '') for call in table_calls]

        assert any("myfile_spectra_table_with_ids" in cid for cid in cache_ids)
        assert any("myfile_id_table" in cid for cid in cache_ids)

        # Check SequenceView
        sv_call = mock_sv.call_args
        assert "myfile_sequence_view" in sv_call.kwargs.get('cache_id', '')
