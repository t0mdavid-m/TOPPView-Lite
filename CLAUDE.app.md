# TOPPView-Lite

TOPPView-Lite is a browser-based mzML viewer built on the OpenMS streamlit-template.
It ships five pages (`content/welcome.py`, `content/upload.py`, `content/viewer.py`,
`content/layout.py`, `content/tools.py`);
the template's demo pages are not registered. mzML, idXML and featureXML files are converted to
parquet caches by `src/preprocessing/` (pyOpenMS), and `src/components/factory.py`
builds the OpenMS-Insight components (Table, SeriesPlot, Heatmap, SequenceView, MirrorPlot).

The viewer is modular: `src/components/viewer_panels.py` holds one function per component
(`PANELS`) and the catalog the Layout page offers (`CATALOG`, names are stored in saved
layouts, so never rename one). The Layout page saves `<workspace>/layouts/viewer.json` with
the template's layout manager (`src/layout_manager/`); `content/viewer.py` builds a
`ViewerContext` for the selected file and draws the saved rows with `render_layout`,
skipping components the file cannot show (no idXML, no ion mobility, no featureXML). All
panels share the `viewer_state` `StateManager`; a panel must not rely on another one having
been drawn first. A new component needs a `CATALOG` entry, a `PANELS` function and a place in
`DEFAULT_LAYOUT` (`tests/unit/test_viewer_panels.py` checks the three agree).
Unit tests live in `tests/unit/`; `tests/test_pages.py` smoke-tests the pages.

The Tools page offers fixed pyOpenMS operations with parameter forms generated from the
algorithms' `Param` objects (`src/operations.py`), run in a separate interpreter
(`python -m src.operations`, not multiprocessing: Streamlit registers each page as
`__main__`, so a spawned child would re-run the page). Its Python console
(`src/console.py`, `src/console_runner.py`) runs arbitrary code and must only be shown
when `online_deployment` is false; `tests/test_pages.py` asserts that.

Template-level changes go to OpenMS/streamlit-template first and arrive here by
merging `template/main`.
