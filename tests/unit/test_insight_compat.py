"""Compatibility shims for the installed openms-insight release."""

import polars as pl


def test_sequenceview_payload_cache_key(tmp_path):
    """openms-insight 0.4.0 needs _get_render_key on SequenceView to render it."""
    import src.components.factory  # noqa: F401  (applies the shim)
    from openms_insight import SequenceView
    from openms_insight.rendering.bridge import _payload_cache_key

    sv = SequenceView(
        cache_id="compat_sv",
        sequence_data=pl.LazyFrame({"scan_id": [1], "sequence": ["PEPTIDER"], "precursor_charge": [2]}),
        peaks_data=pl.LazyFrame({"scan_id": [1], "peak_id": [1], "mass": [200.5], "intensity": [1.0]}),
        cache_path=str(tmp_path),
        filters={"spectrum": "scan_id"},
    )
    key = _payload_cache_key(sv, {"spectrum": 1}, set(sv.get_state_dependencies()))
    assert ("__render_key__", None) in key
