"""Child process for the local Python console. Not imported by the app.

Usage: python -I console_runner.py <context.json>

The context holds the user's code and the paths of the open files. The script
runs with a namespace of loaded data and helpers that write new files into the
workspace. Only the local app starts this (see content/tools.py).
"""

import json
import sys
import traceback
from pathlib import Path


def main(context_path: str) -> int:
    ctx = json.loads(Path(context_path).read_text())
    workspace = Path(ctx["workspace"])

    import numpy as np
    import pandas as pd
    import polars as pl
    import pyopenms as oms

    def _target(folder: str, name: str, suffix: str) -> Path:
        name = Path(name).name  # no directories
        if not name.endswith(suffix):
            name += suffix
        path = workspace / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_mzml(experiment, name):
        """Write an MSExperiment as a new mzML in the workspace (shown on the upload page)."""
        path = _target("mzML-files", name, ".mzML")
        oms.MzMLFile().store(str(path), experiment)
        print(f"Saved {path.name}")
        return path

    def save_features(feature_map, name):
        """Write a FeatureMap as featureXML; name it like the mzML to show it in the viewer."""
        path = _target("featureXML-files", name, ".featureXML")
        feature_map.ensureUniqueId()
        oms.FeatureXMLFile().store(str(path), feature_map)
        print(f"Saved {path.name}")
        return path

    exp = oms.MSExperiment()
    oms.MzMLFile().load(ctx["mzml_path"], exp)

    feature_map = None
    if ctx.get("featurexml_path"):
        feature_map = oms.FeatureMap()
        oms.FeatureXMLFile().load(ctx["featurexml_path"], feature_map)

    protein_ids = []
    # pyOpenMS 3.5+ needs PeptideIdentificationList here
    peptide_ids = oms.PeptideIdentificationList() if hasattr(oms, "PeptideIdentificationList") else []
    if ctx.get("idxml_path"):
        oms.IdXMLFile().load(ctx["idxml_path"], protein_ids, peptide_ids)

    scan_id = ctx.get("scan_id")
    namespace = {
        "__name__": "__console__",
        "oms": oms,
        "pyopenms": oms,
        "np": np,
        "pd": pd,
        "pl": pl,
        "exp": exp,
        "feature_map": feature_map,
        "peptide_ids": peptide_ids,
        "protein_ids": protein_ids,
        "scan_id": scan_id,
        "spectrum_index": scan_id - 1 if scan_id else None,
        "mzml_path": ctx["mzml_path"],
        "stem": Path(ctx["mzml_path"]).stem,
        "save_mzml": save_mzml,
        "save_features": save_features,
    }
    try:
        exec(compile(ctx["code"], "<console>", "exec"), namespace)  # noqa: S102 - local app only
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
