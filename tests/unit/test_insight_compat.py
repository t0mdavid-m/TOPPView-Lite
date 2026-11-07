"""The installed openms-insight release renders every component the viewer uses."""

import polars as pl


def test_sequenceview_payload_cache_key(tmp_path):
    """A SequenceView, which has no render key of its own, gets a payload cache key."""
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
