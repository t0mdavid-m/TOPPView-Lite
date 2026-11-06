# TOPPView-Lite

TOPPView-Lite is a browser-based mzML viewer built on the OpenMS streamlit-template.
It ships three pages (`content/welcome.py`, `content/upload.py`, `content/viewer.py`);
the template's demo pages are not registered. mzML and idXML files are converted to
parquet caches by `src/preprocessing/` (pyOpenMS), and `src/components/factory.py`
builds the OpenMS-Insight components (Table, LinePlot, Heatmap, SequenceView) that
`content/viewer.py` links through one `StateManager`. Unit tests for those modules
live in `tests/unit/`; `tests/test_pages.py` smoke-tests the three pages.

Template-level changes go to OpenMS/streamlit-template first and arrive here by
merging `template/main`.
