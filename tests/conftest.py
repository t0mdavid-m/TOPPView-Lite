"""Shared pytest fixtures for TOPPView-Lite tests."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import polars as pl

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Create mock for pyopenms to avoid dependency on actual OpenMS installation
mock_pyopenms = MagicMock()
mock_pyopenms.__version__ = "3.0.0"

# Mock AASequence
mock_aa_sequence = MagicMock()
mock_aa_sequence.fromString = MagicMock(return_value=MagicMock(
    getMonoWeight=MagicMock(return_value=1234.56),
    size=MagicMock(return_value=0),
))
mock_pyopenms.AASequence = mock_aa_sequence
mock_pyopenms.IdXMLFile = MagicMock

sys.modules['pyopenms'] = mock_pyopenms


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory with required structure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create required directories
    (workspace / "mzML-files").mkdir()
    (workspace / "idXML-files").mkdir()

    return workspace


@pytest.fixture
def temp_cache_paths(temp_workspace):
    """Create cache paths dictionary pointing to temp directories."""
    cache_dir = temp_workspace / ".cache" / "test_file"
    raw_dir = cache_dir / "raw"
    components_dir = cache_dir / "components"
    id_dir = cache_dir / "identifications"

    raw_dir.mkdir(parents=True)
    components_dir.mkdir(parents=True)
    id_dir.mkdir(parents=True)

    return {
        "metadata": raw_dir / "metadata.parquet",
        "peaks": raw_dir / "peaks.parquet",
        "im_info": raw_dir / "im_info.json",
        "spectra_table": components_dir / "spectra_table.parquet",
        "peaks_table": components_dir / "peaks_table.parquet",
        "spectrum_plot": components_dir / "spectrum_plot.parquet",
        "heatmap_input": components_dir / "heatmap_input.parquet",
        "im_table": components_dir / "im_table.parquet",
        "spectra_table_with_ids": components_dir / "spectra_table_with_ids.parquet",
        "component_cache": components_dir / "component_cache",
        "search_params": id_dir / "search_params.json",
        "identifications": id_dir / "identifications.parquet",
        "id_info": id_dir / "id_info.json",
    }


@pytest.fixture
def sample_metadata_df():
    """Create a sample metadata DataFrame."""
    return pl.DataFrame({
        "scan_id": [1, 2, 3, 4, 5],
        "name": ["scan=1", "scan=2", "scan=3", "scan=4", "scan=5"],
        "retention_time": [1.0, 2.0, 3.0, 4.0, 5.0],  # in minutes
        "ms_level": [1, 2, 1, 2, 2],
        "precursor_mz": [0.0, 500.5, 0.0, 600.6, 700.7],
        "charge": [0, 2, 0, 3, 2],
        "num_peaks": [100, 50, 80, 60, 40],
    })


@pytest.fixture
def sample_peaks_df():
    """Create a sample peaks DataFrame."""
    return pl.DataFrame({
        "peak_id": list(range(10)),
        "scan_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        "mass": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "intensity": [1000.0, 2000.0, 1500.0, 500.0, 800.0, 1200.0, 600.0, 400.0, 300.0, 200.0],
        "retention_time": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
    })


@pytest.fixture
def sample_id_df():
    """Create a sample identifications DataFrame."""
    return pl.DataFrame({
        "id_idx": [0, 1, 2],
        "rt": [120.0, 240.0, 300.0],  # in seconds
        "precursor_mz": [500.5, 600.6, 700.7],
        "sequence": ["PEPTIDE", "SEQUENCE", "ANOTHER"],
        "sequence_display": ["PEPTIDE", "SEQUENCE", "ANOTHER"],
        "charge": [2, 3, 2],
        "score": [0.95, 0.88, 0.75],
        "theoretical_mass": [799.36, 950.41, 700.33],
        "protein_accession": ["PROT1", "PROT2", "PROT3"],
        "scan_id": [2, 4, 5],
    })


@pytest.fixture
def sample_im_info_none():
    """Sample ion mobility info for data without IM."""
    return {"type": "none", "unit": ""}


@pytest.fixture
def sample_im_info_faims():
    """Sample ion mobility info for FAIMS data."""
    return {
        "type": "faims",
        "unit": " V",
        "num_dimensions": 3,
        "unique_values": [-35.0, -45.0, -55.0],
    }


@pytest.fixture
def populated_cache(temp_cache_paths, sample_metadata_df, sample_peaks_df, sample_im_info_none):
    """Create a populated cache with sample data."""
    # Write raw data
    sample_metadata_df.write_parquet(temp_cache_paths["metadata"])
    sample_peaks_df.write_parquet(temp_cache_paths["peaks"])

    with open(temp_cache_paths["im_info"], "w") as f:
        json.dump(sample_im_info_none, f)

    return temp_cache_paths


@pytest.fixture
def mock_streamlit():
    """Mock essential Streamlit components for testing."""
    with patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.session_state', create=True, new={}) as mock_session_state, \
         patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.info') as mock_info:

        mock_session_state["workspace"] = str(Path.cwd())
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        yield {
            'tabs': mock_tabs,
            'columns': mock_columns,
            'session_state': mock_session_state,
            'selectbox': mock_selectbox,
            'spinner': mock_spinner,
            'warning': mock_warning,
            'info': mock_info,
        }
