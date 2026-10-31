"""Unit tests for src/components/navigation.py."""

import pytest

from src.components.navigation import navigate

SCANS = [1, 2, 3, 4, 5, 6]
LEVELS = [1, 2, 2, 1, 2, 1]


@pytest.mark.parametrize(
    "current, action, expected",
    [
        (3, "first", 1),
        (3, "last", 6),
        (3, "prev", 2),
        (3, "next", 4),
        (1, "prev", None),
        (6, "next", None),
        (3, "next_ms1", 4),
        (3, "prev_ms1", 1),
        (4, "next_ms2", 5),
        (4, "prev_ms2", 3),
        (5, "next_ms2", None),
        (1, "prev_ms1", None),
    ],
)
def test_navigate(current, action, expected):
    assert navigate(SCANS, LEVELS, current, action) == expected


def test_without_selection_next_starts_at_the_beginning():
    assert navigate(SCANS, LEVELS, None, "next_ms2") == 2
    assert navigate(SCANS, LEVELS, None, "prev_ms1") == 6


def test_empty_file():
    assert navigate([], [], None, "next") is None


def test_unknown_action():
    with pytest.raises(ValueError):
        navigate(SCANS, LEVELS, 1, "sideways")
