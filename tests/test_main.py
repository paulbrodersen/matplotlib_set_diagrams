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
    VennDiagram,
    VennWordCloud,
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


@pytest.mark.mpl_image_compare
def test_SetDiagram():
    fig, ax = plt.subplots()
    SetDiagram([(0,0), (1, 0)], [0.66, 0.66], ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_EulerDiagramBase():
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15,10))
    axes = axes.ravel()

    axes[0].set_title("|A| = |B|")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    EulerDiagramBase(subset_sizes, ax=axes[0])

    axes[1].set_title("|A| > |B|")
    subset_sizes = {
        (1, 0) : 2,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    EulerDiagramBase(subset_sizes, ax=axes[1])

    axes[2].set_title(r"A $\supset$ B")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 0,
        (1, 1) : 0.5,
    }
    EulerDiagramBase(subset_sizes, ax=axes[2])

    axes[3].set_title(r"A $\sqcup$ B")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0,
    }
    EulerDiagramBase(subset_sizes, ax=axes[3])

    axes[4].set_title("|A| = |B| = |C|")
    subset_sizes = {
        (1, 0, 0) : 1,
        (0, 1, 0) : 1,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0.5,
        (1, 0, 1) : 0.5,
        (0, 1, 1) : 0.5,
        (1, 1, 1) : 0.25,
    }
    EulerDiagramBase(subset_sizes, ax=axes[4])

    axes[5].set_title("|A| = |B| = |C| = |D|")
    subset_sizes = {
        (1, 0, 0, 0) : 1,
        (0, 1, 0, 0) : 1,
        (0, 0, 1, 0) : 1,
        (1, 1, 0, 0) : 0.5,
        (1, 0, 1, 0) : 0.5,
        (0, 1, 1, 0) : 0.5,
        (1, 1, 1, 0) : 0.25,

        (1, 0, 0, 0) : 1,
        (0, 1, 0, 0) : 1,
        (0, 0, 0, 1) : 1,
        (1, 1, 0, 0) : 0.5,
        (1, 0, 0, 1) : 0.5,
        (0, 1, 0, 1) : 0.5,
        (1, 1, 0, 1) : 0.25,

        (1, 0, 0, 0) : 1,
        (0, 0, 1, 0) : 1,
        (0, 0, 0, 1) : 1,
        (1, 0, 1, 0) : 0.5,
        (1, 0, 0, 1) : 0.5,
        (0, 0, 1, 1) : 0.5,
        (1, 0, 1, 1) : 0.25,

        (0, 1, 0, 0) : 1,
        (0, 0, 1, 0) : 1,
        (0, 0, 0, 1) : 1,
        (0, 1, 1, 0) : 0.5,
        (0, 1, 0, 1) : 0.5,
        (0, 0, 1, 1) : 0.5,
        (0, 1, 1, 1) : 0.25,

        (1, 1, 1, 1) : 0.125,
    }
    EulerDiagramBase(subset_sizes, ax=axes[5])
    return fig


@pytest.mark.mpl_image_compare
def test_EulerDiagram():
    fig, axes = plt.subplots(1, 3)

    # canonical example
    sets = [
        {"Lorem", "ipsum", "dolor"},
        {"dolor", "sit", "amet"}
    ]
    EulerDiagram(sets, ax=axes[0])

    # no intersection
    sets = [
        {"Lorem", "ipsum", "dolor"},
        {"sit", "amet"}
    ]
    EulerDiagram(sets, ax=axes[1])

    # empty set
    sets = [
        {"Lorem", "ipsum", "dolor", "sit", "amet"},
        {}
    ]
    EulerDiagram(sets, ax=axes[2])

    return fig


@pytest.mark.mpl_image_compare
def test_EulerWordCloud():
    test_string_1 = """Lorem ipsum dolor sit amet, consetetur
    sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore
    et dolore magna aliquyam erat, sed diam voluptua."""

    test_string_2 = """At vero eos et accusam et justo duo dolores et
    ea rebum. Stet clita kasd gubergren, no sea takimata sanctus
    est. Lorem ipsum dolor sit amet."""

    # tokenize words
    sets = []
    for test_string in [test_string_1, test_string_2]:
        # get a word list
        words = test_string.split(' ')
        # remove non alphanumeric characters
        words = [''.join(ch for ch in word if ch.isalnum()) for word in words]
        # convert to all lower case
        words = [word.lower() for word in words]
        sets.append(set(words))

    fig, ax = plt.subplots()
    EulerWordCloud(sets, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_VennDiagram():
    fig, axes = plt.subplots(1, 2)
    # canonical
    VennDiagram([{0, 1}, {1, 2}], ax=axes[0])
    # all empty sets
    VennDiagram([{}, {}], ax=axes[1])
    return fig
