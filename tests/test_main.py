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
    EulerDiagramBase,
    EulerDiagram,
)

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


def test_EulerDiagramBase():
    # EulerDiagramBase._get_set_sizes
    # EulerDiagramBase._get_radii
    # EulerDiagramBase._initialize_origins
    # EulerDiagramBase._get_subset_geometries
    # EulerDiagramBase._get_origins
    # EulerDiagramBase.cost_function
    # EulerDiagramBase.cost_function.constraint_function
    # EulerDiagramBase._evaluate
    # EulerDiagramBase._evaluate.get_cost
    # EulerDiagramBase._pretty_print_performance
    # EulerDiagramBase._initialize_axis
    # EulerDiagramBase._get_subset_colors
    # EulerDiagramBase._draw_subsets
    # EulerDiagramBase._draw_subset_labels
    # EulerDiagramBase._draw_set_labels
    pass


def test_EulerDiagram():
    # EulerDiagram._get_subset_sizes
    pass

    pass
