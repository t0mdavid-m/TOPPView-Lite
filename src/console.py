"""Local Python console: snippets and the child-process launcher.

The console runs arbitrary code, so the Tools page shows it only in the local app
(``online_deployment: false``). Code always runs in a separate interpreter
(``src/console_runner.py``) with a timeout, never inside the Streamlit process.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RUNNER = Path(__file__).with_name("console_runner.py")

DEFAULT_CODE = """\
# Loaded: exp (MSExperiment), feature_map, peptide_ids, protein_ids
# Selection from the viewer: scan_id, spectrum_index (None if nothing selected)
# Modules: oms (pyopenms), np, pd, pl
# Write results: save_mzml(exp, name), save_features(feature_map, name)
print(f"{stem}: {exp.getNrSpectra()} spectra")
if spectrum_index is not None:
    spec = exp[spectrum_index]
    print(f"Selected scan {scan_id}: MS{spec.getMSLevel()}, RT {spec.getRT():.1f} s, {spec.size()} peaks")
"""

SNIPPETS = {
    "Overview": DEFAULT_CODE,
    "TIC of MS1 spectra": """\
rows = [(s.getRT() / 60, sum(s.get_peaks()[1])) for s in exp if s.getMSLevel() == 1]
tic = pd.DataFrame(rows, columns=["rt_min", "tic"])
print(tic.describe())
""",
    "Top 10 peaks of the selected spectrum": """\
if spectrum_index is None:
    print("Select a spectrum in the viewer first.")
else:
    mz, inten = exp[spectrum_index].get_peaks()
    top = pd.DataFrame({"mz": mz, "intensity": inten}).nlargest(10, "intensity")
    print(top.to_string(index=False))
""",
    "Precursor list": """\
rows = []
for i, s in enumerate(exp):
    if s.getMSLevel() == 2 and s.getPrecursors():
        p = s.getPrecursors()[0]
        rows.append((i + 1, s.getRT() / 60, p.getMZ(), p.getCharge()))
df = pd.DataFrame(rows, columns=["scan_id", "rt_min", "precursor_mz", "charge"])
print(df.to_string(index=False, max_rows=50))
""",
    "Centroid with PeakPickerHiRes": """\
picker = oms.PeakPickerHiRes()
params = picker.getParameters()
params.setValue("signal_to_noise", 1.0)
picker.setParameters(params)
picked = oms.MSExperiment()
picker.pickExperiment(exp, picked, True)
save_mzml(picked, f"{stem}_centroided")
""",
    "FeatureFinderMetabo": """\
exp.sortSpectra(True)
mtd = oms.MassTraceDetection()
p = mtd.getParameters(); p.setValue("mass_error_ppm", 10.0); mtd.setParameters(p)
traces = []
mtd.run(exp, traces, 0)
split = []
oms.ElutionPeakDetection().detectPeaks(traces, split)
features, chroms = oms.FeatureMap(), []
oms.FeatureFindingMetabo().run(split, features, chroms)
features.setUniqueIds()
print(f"{features.size()} features")
save_features(features, stem)  # same name as the mzML: the viewer shows it
""",
}


@dataclass
class ConsoleResult:
    output: str
    returncode: int
    timed_out: bool = False


def run_script(
    code: str,
    workspace: Path,
    mzml_path: Path,
    featurexml_path: Optional[Path] = None,
    idxml_path: Optional[Path] = None,
    scan_id: Optional[int] = None,
    timeout: float = 300,
) -> ConsoleResult:
    """Run ``code`` in a separate Python process and return its combined output."""
    # The child runs with the workspace as cwd, so every path must be absolute.
    workspace = Path(workspace).resolve()
    resolve = lambda p: str(Path(p).resolve()) if p else None  # noqa: E731
    context = {
        "code": code,
        "workspace": str(workspace),
        "mzml_path": resolve(mzml_path),
        "featurexml_path": resolve(featurexml_path),
        "idxml_path": resolve(idxml_path),
        "scan_id": int(scan_id) if scan_id is not None else None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ctx_path = Path(tmp, "context.json")
        ctx_path.write_text(json.dumps(context))
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(ctx_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace),
            )
        except subprocess.TimeoutExpired as err:
            out = (err.stdout or b"")
            out = out.decode() if isinstance(out, bytes) else out
            return ConsoleResult(out + f"\n[Stopped after {timeout:.0f} s]", -1, True)
    output = proc.stdout + (("\n" + proc.stderr) if proc.stderr.strip() else "")
    return ConsoleResult(output.strip(), proc.returncode)
