"""Processing operations offered on the Tools page.

Each operation is a fixed pyOpenMS pipeline with its algorithm parameters exposed
as a form. This is the hosted-safe alternative to a Python console: users choose
parameters, never code. Operations write new files into the workspace (centroided
mzML into ``mzML-files``, features into ``featureXML-files``), where the upload
page and the viewer pick them up like any other file.

Streamlit-free so it can run in a child process and be unit tested.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pyopenms as oms

MZML_DIR_NAME = "mzML-files"
FEATURE_DIR_NAME = "featureXML-files"

# pyOpenMS Param value types (ParamValue::ValueType)
_STRING, _INT, _DOUBLE, _STRING_LIST, _INT_LIST, _DOUBLE_LIST = range(6)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._\- ]+$")


# =============================================================================
# Parameter forms
# =============================================================================


def _s(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def param_entries(param: oms.Param, prefix: str = "") -> List[Dict[str, Any]]:
    """Describe every entry of a pyOpenMS Param for a generated form.

    Each dict has key, value, type ("string", "int", "float", "bool", "list"),
    description, advanced, choices (valid strings), min and max.
    """
    entries = []
    for raw_key in param.keys():
        key = _s(raw_key)
        entry = param.getEntry(raw_key)
        value_type = param.getValueType(raw_key)
        value = param.getValue(raw_key)
        try:
            choices = [_s(v) for v in entry.valid_strings]
        except TypeError:
            # pyOpenMS 3.6 stores flags as booleans, which have no string choices
            choices = []
        info: Dict[str, Any] = {
            "key": f"{prefix}{key}",
            "value": value,
            "description": _s(param.getDescription(raw_key)),
            "advanced": b"advanced" in entry.tags or "advanced" in entry.tags,
            "choices": choices,
            "min": None,
            "max": None,
        }
        if isinstance(value, bool):
            info["type"] = "bool"
        elif value_type == _STRING:
            info["value"] = _s(value)
            if sorted(choices) == ["false", "true"]:
                info["type"] = "bool"
                info["value"] = info["value"] == "true"
            else:
                info["type"] = "string"
        elif value_type == _INT:
            info["type"] = "int"
            info["min"], info["max"] = entry.min_int, entry.max_int
        elif value_type == _DOUBLE:
            info["type"] = "float"
            info["min"], info["max"] = entry.min_float, entry.max_float
        else:
            info["type"] = "list"
            info["item_type"] = {_STRING_LIST: "string", _INT_LIST: "int"}.get(value_type, "float")
            info["value"] = [_s(v) if isinstance(v, bytes) else v for v in value]
        entries.append(info)
    return entries


def _coerce(value: Any, entry: Dict[str, Any]) -> Any:
    kind = entry["type"]
    if kind == "bool":
        flag = value in (True, "true", "True", 1)
        # pyOpenMS 3.6 keeps flags as booleans; older Params as "true"/"false"
        return flag if isinstance(entry["value"], bool) and not entry["choices"] else ("true" if flag else "false")
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "list":
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        cast = {"int": int, "float": float}.get(entry.get("item_type"), str)
        return [cast(v) for v in value]
    value = str(value)
    if entry["choices"] and value not in entry["choices"]:
        raise ValueError(f"{entry['key']}: '{value}' is not one of {entry['choices']}")
    return value


def apply_overrides(param: oms.Param, overrides: Dict[str, Any], prefix: str = "") -> oms.Param:
    """Return a copy of ``param`` with the overrides that belong to ``prefix`` applied.

    Unknown keys raise ValueError, so a form can never set something the
    algorithm does not define.
    """
    result = oms.Param(param)
    entries = {e["key"]: e for e in param_entries(param, prefix)}
    for key, value in overrides.items():
        if prefix and not key.startswith(prefix):
            continue
        if key not in entries:
            raise ValueError(f"Unknown parameter: {key}")
        result.setValue(key[len(prefix):], _coerce(value, entries[key]))
    return result


# =============================================================================
# Operations
# =============================================================================


@dataclass(frozen=True)
class Operation:
    id: str
    name: str
    description: str
    output_dir: str
    output_suffix: str
    sections: Dict[str, Callable[[], Any]]  # prefix -> algorithm factory
    run: Callable[[Path, Path, Dict[str, oms.Param]], str]

    def entries(self) -> List[Dict[str, Any]]:
        out = []
        for prefix, factory in self.sections.items():
            out.extend(param_entries(factory().getParameters(), prefix))
        return out

    def params(self, overrides: Dict[str, Any]) -> Dict[str, oms.Param]:
        known = {e["key"] for e in self.entries()}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")
        return {
            prefix: apply_overrides(factory().getParameters(), overrides, prefix)
            for prefix, factory in self.sections.items()
        }

    def default_output_name(self, mzml_path: Path) -> str:
        return f"{Path(mzml_path).stem}{self.output_suffix}"


def _load(path: Path) -> oms.MSExperiment:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(path), exp)
    exp.sortSpectra(True)
    return exp


def _run_peak_picking(in_path: Path, out_path: Path, params: Dict[str, oms.Param]) -> str:
    exp = _load(in_path)
    picker = oms.PeakPickerHiRes()
    picker.setParameters(params[""])
    picked = picker.pickExperiment(exp, True)
    oms.MzMLFile().store(str(out_path), picked)
    return f"Centroided {picked.getNrSpectra()} spectra."


def _mass_traces(exp: oms.MSExperiment, params: Dict[str, oms.Param]) -> list:
    mtd = oms.MassTraceDetection()
    mtd.setParameters(params["mtd:"])
    return mtd.run(exp, 0)


def _run_mass_traces(in_path: Path, out_path: Path, params: Dict[str, oms.Param]) -> str:
    exp = _load(in_path)
    traces = _mass_traces(exp, params)
    fmap = oms.FeatureMap()
    for i, trace in enumerate(traces):
        feature = oms.Feature()
        feature.setRT(trace.getCentroidRT())
        feature.setMZ(trace.getCentroidMZ())
        feature.setIntensity(trace.computePeakArea())
        feature.setOverallQuality(1.0)
        feature.setUniqueId(i + 1)
        hull = oms.ConvexHull2D()
        hull.setHullPoints(trace.getConvexhull().getHullPoints())
        feature.setConvexHulls([hull])
        fmap.push_back(feature)
    fmap.ensureUniqueId()
    fmap.setPrimaryMSRunPath([str(in_path).encode()])
    oms.FeatureXMLFile().store(str(out_path), fmap)
    return f"Detected {len(traces)} mass traces."


def _run_feature_finder_metabo(in_path: Path, out_path: Path, params: Dict[str, oms.Param]) -> str:
    exp = _load(in_path)
    traces = _mass_traces(exp, params)
    epd = oms.ElutionPeakDetection()
    epd.setParameters(params["epd:"])
    split = epd.detectPeaks(traces)
    ffm = oms.FeatureFindingMetabo()
    ffm.setParameters(params["ffm:"])
    fmap = oms.FeatureMap()
    ffm.run(split, fmap)
    fmap.setUniqueIds()
    fmap.ensureUniqueId()
    fmap.setPrimaryMSRunPath([str(in_path).encode()])
    oms.FeatureXMLFile().store(str(out_path), fmap)
    return f"Found {fmap.size()} features from {len(split)} mass traces."


OPERATIONS: Dict[str, Operation] = {
    op.id: op
    for op in [
        Operation(
            id="peak_picking",
            name="Centroid spectra (PeakPickerHiRes)",
            description="Pick peaks in profile spectra. Writes a new centroided mzML.",
            output_dir=MZML_DIR_NAME,
            output_suffix="_centroided.mzML",
            sections={"": oms.PeakPickerHiRes},
            run=_run_peak_picking,
        ),
        Operation(
            id="feature_finder_metabo",
            name="Detect features (FeatureFinderMetabo)",
            description=(
                "Mass trace detection, elution peak splitting and isotope pattern "
                "assembly on centroided MS1 data. Writes a featureXML that the viewer "
                "shows next to the mzML."
            ),
            output_dir=FEATURE_DIR_NAME,
            output_suffix=".featureXML",
            sections={
                "mtd:": oms.MassTraceDetection,
                "epd:": oms.ElutionPeakDetection,
                "ffm:": oms.FeatureFindingMetabo,
            },
            run=_run_feature_finder_metabo,
        ),
        Operation(
            id="mass_trace_detection",
            name="Detect mass traces (MassTraceDetection)",
            description="Mass traces of centroided MS1 data, written as one feature per trace.",
            output_dir=FEATURE_DIR_NAME,
            output_suffix=".featureXML",
            sections={"mtd:": oms.MassTraceDetection},
            run=_run_mass_traces,
        ),
    ]
}


def output_path(workspace: Path, operation: Operation, name: str) -> Path:
    """Validated output path inside the operation's workspace folder."""
    name = name.strip()
    if not name or not _SAFE_NAME.match(name) or name.startswith("."):
        raise ValueError("Use a plain file name (letters, digits, '.', '_', '-', spaces).")
    suffix = "." + operation.output_suffix.rsplit(".", 1)[-1]
    if not name.endswith(suffix):
        name += suffix
    return Path(workspace) / operation.output_dir / name


def run_operation(operation_id: str, in_path: Path, out_path: Path, overrides: Dict[str, Any]) -> str:
    """Run an operation in this process. Returns a one-line summary."""
    operation = OPERATIONS[operation_id]
    params = operation.params(overrides)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return operation.run(Path(in_path), Path(out_path), params)


_RESULT_MARKER = "@@OPERATION_RESULT@@"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def run_operation_isolated(
    operation_id: str,
    in_path: Path,
    out_path: Path,
    overrides: Dict[str, Any],
    timeout: Optional[float] = 3600,
) -> str:
    """Run an operation in a separate Python process so a crash or hang cannot take the app down.

    Uses a fresh interpreter (``python -m src.operations``) rather than
    multiprocessing: Streamlit registers each page script as ``__main__``, and a
    spawned multiprocessing child would re-execute that page.

    Raises RuntimeError with the child's error message, or TimeoutError.
    """
    request = json.dumps(
        {
            "operation": operation_id,
            "in": str(Path(in_path).resolve()),
            "out": str(Path(out_path).resolve()),
            "overrides": overrides,
        }
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.operations"],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired as err:
        raise TimeoutError(f"Stopped after {timeout:.0f} s.") from err
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            status, message = json.loads(line[len(_RESULT_MARKER):])
            if status != "ok":
                raise RuntimeError(message)
            return message
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
    raise RuntimeError(f"The operation ended without a result: {tail[0]}")


def _main() -> int:
    request = json.loads(sys.stdin.read())
    try:
        result = ("ok", run_operation(request["operation"], request["in"], request["out"], request["overrides"]))
    except Exception as err:  # reported to the page
        result = ("error", f"{type(err).__name__}: {err}")
    print(_RESULT_MARKER + json.dumps(result), flush=True)
    return 0 if result[0] == "ok" else 1


if __name__ == "__main__":
    sys.exit(_main())
