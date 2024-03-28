#!/usr/bin/env python
"""
Test _main.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from matplotlib_set_diagrams._main import (
    blend_colors,
    rgba_to_grayscale,
    get_text_alignment,
    SetDiagram,
    EulerDiagramBase,
    EulerDiagram,
    EulerWordCloud,
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


def test_SetDiagram():
    # SetDiagram.__init__
    # SetDiagram._get_subset_geometries
    # SetDiagram._get_subset_colors
    # SetDiagram._initialize_axis
    # SetDiagram._draw_subsets
    # SetDiagram._draw_subset_labels
    # SetDiagram._draw_set_labels
    pass


def test_EulerDiagramBase():
    # EulerDiagramBase.__init__
    # EulerDiagramBase._get_layout
    # EulerDiagramBase._initialize_layout
    # EulerDiagramBase._get_set_sizes
    # EulerDiagramBase._initialize_radii
    # EulerDiagramBase._initialize_origins
    # EulerDiagramBase._optimize_layout
    # EulerDiagramBase._optimize_layout.cost_function
    # EulerDiagramBase._optimize_layout.constraint_function
    # EulerDiagramBase._get_set_labels
    # EulerDiagramBase._get_subset_labels
    pass


def test_EulerDiagram():
    # EulerDiagram.__init__
    # EulerDiagram._get_subset_sizes
    pass


def test_EulerWordCloud():
    # EulerWordCloud.__init__
    # EulerWordCloud._draw_subsets
    # EulerWordCloud._draw_subset_labels
    # EulerWordCloud._get_subsets
    # EulerWordCloud._get_wordcloud
    pass
