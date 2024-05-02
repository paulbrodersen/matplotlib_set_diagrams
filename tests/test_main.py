#!/usr/bin/env python
"""
Test _main.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from matplotlib_set_diagrams._main import (
    get_subset_ids,
    get_subsets,
    get_subset_sizes,
    blend_colors,
    rgba_to_grayscale,
    get_text_alignment,
    SetDiagram,
    EulerDiagramFromSubsetSizes,
    EulerDiagram,
    EulerWordCloud,
    VennDiagramFromSubsetSizes,
    VennDiagram,
    VennWordCloud,
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


def test_get_subset_sizes():
    actual = get_subset_sizes([{0, 1}, {1, 2}])
    desired = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 1,
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


@pytest.mark.mpl_image_compare
def test_SetDiagram():
    fig, ax = plt.subplots()
    SetDiagram([(0, 0), (1, 0)], [0.66, 0.66], ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_EulerDiagramFromSubsetSizes():
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15,10))
    axes = axes.ravel()

    axes[0].set_title("|A| = |B|")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    EulerDiagramFromSubsetSizes(subset_sizes, ax=axes[0])

    axes[1].set_title("|A| > |B|")
    subset_sizes = {
        (1, 0) : 2,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    EulerDiagramFromSubsetSizes(subset_sizes, ax=axes[1])

    axes[2].set_title(r"A $\supset$ B")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 0,
        (1, 1) : 0.5,
    }
    EulerDiagramFromSubsetSizes(subset_sizes, ax=axes[2])

    axes[3].set_title(r"A $\sqcup$ B")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0,
    }
    EulerDiagramFromSubsetSizes(subset_sizes, ax=axes[3])

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
    EulerDiagramFromSubsetSizes(subset_sizes, ax=axes[4])

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
    EulerDiagramFromSubsetSizes(subset_sizes, ax=axes[5])
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
    text_1 = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
    enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
    ut aliquip ex ea commodo consequat."""

    text_2 = """Duis aute irure dolor in reprehenderit in voluptate velit
    esse cillum dolore eu fugiat nulla pariatur. Lorem ipsum dolor sit
    amet."""

    # Tokenize words.
    # The procedure below is a poor-man's tokenization.
    # Consider using the Natural Language Toolkit (NLTK) instead:
    # import nltk; words = nltk.word_tokenize(text)
    sets = []
    for text in [text_1, text_2]:
        # get a word list
        words = text.split(' ')
        # remove non alphanumeric characters
        words = [''.join(ch for ch in word if ch.isalnum()) for word in words]
        # convert to all lower case
        words = [word.lower() for word in words]
        sets.append(set(words))

    fig, ax = plt.subplots()
    EulerWordCloud(sets, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_VennDiagramFromSubsetSizes():
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].set_title("|A| = |B|")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    VennDiagramFromSubsetSizes(subset_sizes, ax=axes[0])

    axes[1].set_title("|A| > |B|")
    subset_sizes = {
        (1, 0) : 2,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    VennDiagramFromSubsetSizes(subset_sizes, ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_VennDiagram():
    fig, axes = plt.subplots(1, 2)
    # canonical
    VennDiagram([{0, 1}, {1, 2}], ax=axes[0])
    # all empty sets
    VennDiagram([{}, {}], ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_VennWordCloud():
    text_1 = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
    enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
    ut aliquip ex ea commodo consequat."""

    text_2 = """Duis aute irure dolor in reprehenderit in voluptate velit
    esse cillum dolore eu fugiat nulla pariatur. Lorem ipsum dolor sit
    amet."""

    # Tokenize words.
    # The procedure below is a poor-man's tokenization.
    # Consider using the Natural Language Toolkit (NLTK) instead:
    # import nltk; words = nltk.word_tokenize(text)
    sets = []
    for text in [text_1, text_2]:
        # get a word list
        words = text.split(' ')
        # remove non alphanumeric characters
        words = [''.join(ch for ch in word if ch.isalnum()) for word in words]
        # convert to all lower case
        words = [word.lower() for word in words]
        sets.append(set(words))

    fig, ax = plt.subplots()
    VennWordCloud(sets, ax=ax)
    return fig
