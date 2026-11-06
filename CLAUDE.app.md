# TOPPView-Lite

TOPPView-Lite is a browser-based mzML viewer built on the OpenMS streamlit-template.
It ships four pages (`content/welcome.py`, `content/upload.py`, `content/viewer.py`,
`content/tools.py`);
the template's demo pages are not registered. mzML and idXML files are converted to
parquet caches by `src/preprocessing/` (pyOpenMS), and `src/components/factory.py`
builds the OpenMS-Insight components (Table, LinePlot, Heatmap, SequenceView) that
`content/viewer.py` links through one `StateManager`. Unit tests for those modules
live in `tests/unit/`; `tests/test_pages.py` smoke-tests the pages.

The Tools page offers fixed pyOpenMS operations with parameter forms generated from the
algorithms' `Param` objects (`src/operations.py`), run in a separate interpreter
(`python -m src.operations`, not multiprocessing: Streamlit registers each page as
`__main__`, so a spawned child would re-run the page). Its Python console
(`src/console.py`, `src/console_runner.py`) runs arbitrary code and must only be shown
when `online_deployment` is false; `tests/test_pages.py` asserts that.

Template-level changes go to OpenMS/streamlit-template first and arrive here by
merging `template/main`.
