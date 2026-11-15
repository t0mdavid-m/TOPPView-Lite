"""Axes of the peak map: retention time in minutes or seconds, and swapped axes.

Each choice is a Heatmap of its own (its own cache and zoom identifier), because
the heatmap's resolution levels are built from the columns it plots. Switching
keeps the region on screen: the zoom of the previous choice is converted.

Streamlit-free, so the conversion is tested on its own.
"""

from typing import Any, Dict, List, Optional, Tuple

RT_UNIT_LABELS = {"min": "RT in minutes", "s": "RT in seconds"}

# Seconds per unit of the retention time the heatmap input holds (minutes).
_SECONDS = {"min": 60.0, "s": 1.0}

DEFAULT_AXES: Tuple[str, bool] = ("min", False)


def axes_suffix(rt_unit: str, swapped: bool) -> str:
    """Suffix of the cache and zoom identifier of a choice ('' for the default)."""
    if (rt_unit, swapped) == DEFAULT_AXES:
        return ""
    return f"_rt_{rt_unit}" + ("_swapped" if swapped else "")


def zoom_identifier(rt_unit: str, swapped: bool) -> str:
    """The state key a choice keeps its zoom under ('heatmap_zoom' by default)."""
    return "heatmap_zoom" + axes_suffix(rt_unit, swapped)


def rt_label(rt_unit: str) -> str:
    return f"Retention Time ({rt_unit})"


def _range(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if low == high or (low == -1 and high == -1):
        return None
    return [min(low, high), max(low, high)]


def _axis(zoom: Dict[str, Any], axis: str) -> Optional[List[float]]:
    return _range(zoom.get(axis, zoom.get(f"{axis}Range")))


def convert_zoom(
    zoom: Any, source: Tuple[str, bool], target: Tuple[str, bool]
) -> Optional[Dict[str, List[float]]]:
    """The zoom of ``source`` axes, expressed for ``target`` axes.

    ``source`` and ``target`` are ``(rt_unit, swapped)``. Returns the value to
    store under the target's zoom identifier in both forms the heatmap reads
    (``x``/``y`` and ``xRange``/``yRange``, ``[-1, -1]`` for an axis shown in
    full), or None for the full view.
    """
    if not isinstance(zoom, dict):
        return None
    src_unit, src_swapped = source
    dst_unit, dst_swapped = target
    rt = _axis(zoom, "y" if src_swapped else "x")
    mz = _axis(zoom, "x" if src_swapped else "y")
    if rt is None and mz is None:
        return None
    if rt is not None:
        factor = _SECONDS[src_unit] / _SECONDS[dst_unit]
        rt = [rt[0] * factor, rt[1] * factor]
    x, y = (mz, rt) if dst_swapped else (rt, mz)
    result: Dict[str, List[float]] = {
        "xRange": x or [-1, -1],
        "yRange": y or [-1, -1],
    }
    if x:
        result["x"] = x
    if y:
        result["y"] = y
    return result
