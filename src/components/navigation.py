"""Spectrum navigation: first/last, previous/next, previous/next MS1 or MS2.

Streamlit-free so it can be unit tested; content/viewer.py wires it to buttons
that write the ``spectrum`` selection shared by all Insight components.
"""

from typing import Optional, Sequence

# (action, button label, help text)
NAVIGATION_ACTIONS = [
    ("first", "⏮", "First spectrum"),
    ("prev", "◀", "Previous spectrum"),
    ("next", "▶", "Next spectrum"),
    ("last", "⏭", "Last spectrum"),
    ("prev_ms1", "◀ MS1", "Previous MS1 spectrum"),
    ("next_ms1", "MS1 ▶", "Next MS1 spectrum"),
    ("prev_ms2", "◀ MS2", "Previous MS2 spectrum"),
    ("next_ms2", "MS2 ▶", "Next MS2 spectrum"),
]


def navigate(
    scan_ids: Sequence[int],
    ms_levels: Sequence[int],
    current: Optional[int],
    action: str,
) -> Optional[int]:
    """Return the scan to select for ``action``, or None when there is none.

    ``scan_ids`` must be in acquisition order, with ``ms_levels`` aligned to
    it. With no current selection, "next"-type actions start before the first
    spectrum and "prev"-type actions after the last one.
    """
    n = len(scan_ids)
    if n == 0:
        return None
    if action == "first":
        return scan_ids[0]
    if action == "last":
        return scan_ids[-1]

    if action.startswith("prev"):
        step = -1
    elif action.startswith("next"):
        step = 1
    else:
        raise ValueError(f"Unknown navigation action: {action!r}")
    level = {"ms1": 1, "ms2": 2}.get(action.split("_", 1)[1]) if "_" in action else None

    try:
        position = list(scan_ids).index(current)
    except ValueError:
        position = -1 if step == 1 else n

    i = position + step
    while 0 <= i < n:
        if level is None or ms_levels[i] == level:
            return scan_ids[i]
        i += step
    return None
