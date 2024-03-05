#!/usr/bin/env python
"""
Test _main.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from matplotlib_euler._main import (
    get_subset_sizes,
    blend_colors,
    rgba_to_grayscale,
    get_text_alignment,
    EulerDiagram,
)

def test_get_subset_sizes():
    s1 = {0, 1}
    s2 = {1, 2}
    desired = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 1,
    }
    assert get_subset_sizes([s1, s2]) == desired


def test_blend_colors():
    np.testing.assert_allclose(blend_colors([(0, 0, 0, 0), (0, 0, 0, 0)]), (0, 0, 0, 0))
    np.testing.assert_allclose(blend_colors([(255, 255, 255, 1), (255, 255, 255, 1)]), (255, 255, 255, 1))


def test_rgba_to_grayscale():
    assert np.isclose(rgba_to_grayscale(1, 1, 1, 1), 1)
    assert np.isclose(rgba_to_grayscale(0, 0, 0, 1), 0)
    assert np.isclose(rgba_to_grayscale(1, 1, 1, 0), 0)


def test_get_text_alignment():
    assert get_text_alignment(1, 0) == ("left", "center")
    assert get_text_alignment(0, 1) == ("center", "bottom")
    assert get_text_alignment(-1, 0) == ("right", "center")
    assert get_text_alignment(0, -1) == ("center", "top")


def test_EulerDiagram():
    #__init__
    #_get_set_sizes
    #_get_radii
    #_initialize_origins
    #_get_subset_geometries
    #_get_origins
    #cost_function
    #cost_function.constraint_function
    #_evaluate
    #_evaluate.get_cost
    #_pretty_print_performance
    #_initialize_axis
    #_draw_subsets
    #_draw_subset_labels
    #_draw_set_labels
    pass
