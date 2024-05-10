#!/usr/bin/env python
"""
Test _utils.py
"""

import pytest
import numpy as np

from matplotlib_set_diagrams._utils import (
    get_subset_ids,
    get_subsets,
    blend_colors,
    rgba_to_grayscale,
    get_text_alignment,
)

def test_get_subset_ids():
    actual = get_subset_ids(2)
    desired = [(1, 0), (0, 1), (1, 1)]
    assert set(actual) == set(desired)

    actual = get_subset_ids(3)
    desired = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    assert set(actual) == set(desired)


def test_get_subsets():
    actual = get_subsets([{0, 1}, {1, 2}])
    desired = {
        (1, 0) : {0},
        (0, 1) : {2},
        (1, 1) : {1},
    }
    assert actual == desired

    sets = [
        {0, 1, 2, 3},
        {4, 1, 5, 3},
        {6, 2, 5, 3},
    ]
    actual = get_subsets(sets)
    desired = {
        (1, 0, 0) : {0},
        (0, 1, 0) : {4},
        (0, 0, 1) : {6},
        (1, 1, 0) : {1},
        (1, 0, 1) : {2},
        (0, 1, 1) : {5},
        (1, 1, 1) : {3},
    }
    assert actual == desired


def test_blend_colors():
    np.testing.assert_allclose(blend_colors([(0, 0, 0, 0), (0, 0, 0, 0)]), (0, 0, 0, 0))
    np.testing.assert_allclose(blend_colors([(1, 1, 1, 1), (1, 1, 1, 1)]), (1, 1, 1, 1))


def test_rgba_to_grayscale():
    assert np.isclose(rgba_to_grayscale(1, 1, 1, 1), 1)
    assert np.isclose(rgba_to_grayscale(0, 0, 0, 1), 0)
    assert np.isclose(rgba_to_grayscale(1, 1, 1, 0), 0)


def test_get_text_alignment():
    assert get_text_alignment(1, 0) == ("left", "center")
    assert get_text_alignment(0, 1) == ("center", "bottom")
    assert get_text_alignment(-1, 0) == ("right", "center")
    assert get_text_alignment(0, -1) == ("center", "top")
