# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TOPPView-Lite is a lightweight web-based mass spectrometry data viewer built with Streamlit. It provides interactive visualization of mzML files (peak maps, spectra, tables) with optional peptide identification (idXML) support including fragment ion annotations and sequence coverage visualization.

## Development Commands

```bash
# Setup (conda recommended)
conda create -n toppview-lite python=3.10 -y
conda activate toppview-lite
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Run tests
pytest tests/                           # All tests
pytest tests/test_simple_workflow.py    # Single file
pytest tests/ -k "test_name"            # Single test by name

# Docker deployment
docker-compose up -d --build

# Docker build (standalone)
docker build -f Dockerfile --no-cache -t toppview-lite:latest .

# Workspace cleanup (removes old cache files)
python clean-up-workspaces.py
```

## Architecture

### Page Structure

The app uses Streamlit's multi-page navigation (`st.navigation`):

- `content/welcome.py` - Landing page
- `content/upload.py` - File upload and preprocessing pipeline
- `content/viewer.py` - Main visualization page with linked components

### Data Flow

```
mzML file → extract_mzml_to_parquet() → raw cache (metadata.pq, peaks.pq, im_info.json)
                                      ↓
                          create_component_inputs() → component cache (spectra_table.pq, etc.)
                                      ↓
                          create_ms_components() → Table, LinePlot, Heatmap
                                      ↓
idXML file → extract_idxml_to_parquet() → id cache (identifications.pq, search_params.json)
                          ↓
       link_identifications_to_spectra() → scan_id linking via RT/mz matching
                          ↓
       create_id_components() → SequenceView + LinePlot.from_sequence_view()
                                      ↓
                          viewer.py renders with StateManager for cross-component linking
```

### Cache Structure

All cache files live under `{workspace}/.cache/{file_stem}/`:

```
raw/
├── metadata.parquet    # Spectrum metadata (scan_id, rt, ms_level, etc.)
├── peaks.parquet       # All peaks (peak_id, scan_id, mass, intensity)
└── im_info.json        # Ion mobility info (FAIMS/TIMS detection)

components/
├── spectra_table.parquet           # Base spectra metadata
├── spectra_table_with_ids.parquet  # With num_ids column (if IDs present)
├── peaks_table.parquet
├── spectrum_plot.parquet
├── heatmap_input.parquet
├── im_table.parquet                # If ion mobility data present
└── component_cache/                # openms_insight internal caches

identifications/        # If idXML uploaded
├── identifications.parquet  # id_idx, sequence, charge, score, scan_id
├── search_params.json       # tolerances for annotation config
└── id_info.json
```

### Key Modules

- `src/preprocessing/extraction.py` - mzML parsing with pyOpenMS, ion mobility detection
- `src/preprocessing/component_inputs.py` - Creates projected parquet files for each component
- `src/preprocessing/identification.py` - idXML parsing, RT/mz spectrum linking (fragment matching now in Vue)
- `src/components/factory.py` - Creates openms_insight components using new patterns

### Component Factory Pattern (openms_insight 0.1.2+)

The factory module provides four functions:

```python
# Create MS visualization components
create_ms_components(paths, im_info, file_id) → {im_table, spectra_table, peaks_table, spectrum_plot, heatmap}

# Create identification components using SequenceView + LinePlot linking
create_id_components(paths, id_paths, im_info, file_id, search_params) → {spectra_table_with_ids, id_table, sequence_view, annotated_spectrum_plot}

# Reconstruct from cache (no data needed)
reconstruct_ms_components(cache_path, file_id, has_im)
reconstruct_id_components(cache_path, file_id)
```

### Cross-Component Linking

Components share selection state via `StateManager(session_key="viewer_state")`:

- `im_dimension` → filters by ion mobility (FAIMS CV)
- `spectrum` → filters by scan_id
- `peak` → highlights peak in plot and table
- `identification` → filters SequenceView and annotated spectrum

Filter propagation:
- IM Table → Spectra Table, Heatmap
- Spectra Table → Peaks Table, Spectrum Plot, ID Table
- ID Table → SequenceView, Annotated Spectrum Plot

### SequenceView + LinePlot Integration

Fragment matching is handled in Vue by SequenceView. The annotated spectrum plot is created using:

```python
# SequenceView handles fragment matching in Vue
sequence_view = SequenceView(
    cache_id=f"{file_id}_sequence_view",
    sequence_data=id_df.lazy().select([...]),
    peaks_data=pl.scan_parquet(paths["spectrum_plot"]),
    filters={"identification": "sequence_id", "spectrum": "scan_id"},
    interactivity={"peak": "peak_id"},
    annotation_config=annotation_config,
    cache_path=cache_path,
)

# LinePlot.from_sequence_view() creates linked annotated plot
annotated_spectrum_plot = LinePlot.from_sequence_view(
    sequence_view,
    cache_id=f"{file_id}_annotated_spectrum",
    cache_path=cache_path,
)

# Render with linking
sv_result = sequence_view(key="sv", state_manager=state_manager)
annotated_spectrum_plot(key="plot", state_manager=state_manager, sequence_view_key="sv")
```

### State Management Patterns

When switching between files in the viewer:

```python
from openms_insight.rendering.bridge import clear_component_cache

# Clear component render caches and state
clear_component_cache()
state_manager = StateManager(session_key="viewer_state")
state_manager.clear()
```

## Configuration

- `settings.json` - App configuration (name, analytics, workspace settings)
- `.streamlit/config.toml` - Streamlit server config (dark theme enabled)
- `default-parameters.json` - Default workflow parameters

## Dependencies

Core: streamlit, pyopenms, polars, openms_insight>=0.1.2, plotly

The app uses Polars for data processing and Apache Arrow serialization for efficient data transfer to Vue.js components.

## CI/CD

GitHub Actions workflows (`.github/workflows/`):
- `ci.yml` - Runs pytest on push
- `build-docker-images.yml` - Docker image builds
- `build-windows-executable-app.yaml` - Windows MSI installer generation
