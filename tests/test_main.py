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
    blend_colors,
    rgba_to_grayscale,
    get_text_alignment,
    SetDiagram,
    EulerDiagram,
    VennDiagram,
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


@pytest.mark.mpl_image_compare
def test_SetDiagram():
    fig, ax = plt.subplots()
    SetDiagram([(0, 0), (1, 0)], [0.66, 0.66], ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_EulerDiagram():
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.ravel()

    axes[0].set_title("|A| = |B|")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    EulerDiagram(subset_sizes, ax=axes[0])

    axes[1].set_title("|A| > |B|")
    subset_sizes = {
        (1, 0) : 2,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    EulerDiagram(subset_sizes, ax=axes[1])

    axes[2].set_title(r"A $\supset$ B")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 0,
        (1, 1) : 0.5,
    }
    EulerDiagram(subset_sizes, ax=axes[2])

    axes[3].set_title(r"A $\sqcup$ B")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0,
    }
    EulerDiagram(subset_sizes, ax=axes[3])

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
    EulerDiagram(subset_sizes, ax=axes[4])

    axes[5].set_title("|A| = |B| = |C| = |D|")
    subset_sizes = {
        (0, 0, 0, 1) : 1.0,
        (0, 0, 1, 0) : 1.0,
        (0, 0, 1, 1) : 0.5,
        (0, 1, 0, 0) : 1.0,
        (0, 1, 0, 1) : 0.5,
        (0, 1, 1, 0) : 0.5,
        (0, 1, 1, 1) : 0.25,
        (1, 0, 0, 0) : 1.0,
        (1, 0, 0, 1) : 0.5,
        (1, 0, 1, 0) : 0.5,
        (1, 0, 1, 1) : 0.25,
        (1, 1, 0, 0) : 0.5,
        (1, 1, 0, 1) : 0.25,
        (1, 1, 1, 0) : 0.25,
        (1, 1, 1, 1) : 0.125,
    }
    EulerDiagram(subset_sizes, ax=axes[5])
    return fig


@pytest.mark.mpl_image_compare
def test_EulerDiagram_from_sets():
    fig, axes = plt.subplots(1, 3)

    # canonical example
    sets = [
        {"Lorem", "ipsum", "dolor"},
        {"dolor", "sit", "amet"}
    ]
    EulerDiagram.from_sets(sets, ax=axes[0])

    # no intersection
    sets = [
        {"Lorem", "ipsum", "dolor"},
        {"sit", "amet"}
    ]
    EulerDiagram.from_sets(sets, ax=axes[1])

    # empty set
    sets = [
        {"Lorem", "ipsum", "dolor", "sit", "amet"},
        set()
    ]
    EulerDiagram.from_sets(sets, ax=axes[2])

    return fig


@pytest.mark.mpl_image_compare
def test_EulerDiagram_as_wordcloud():
    text_1 = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
    enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
    ut aliquip ex ea commodo consequat."""

    text_2 = """Duis aute irure dolor in reprehenderit in voluptate velit
    esse cillum dolore eu fugiat nulla pariatur. Lorem ipsum dolor sit
    amet."""

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
    EulerDiagram.as_wordcloud(sets, wordcloud_kwargs=dict(random_state=42), ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_VennDiagram():
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].set_title("|A| = |B|")
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    VennDiagram(subset_sizes, ax=axes[0])

    axes[1].set_title("|A| > |B|")
    subset_sizes = {
        (1, 0) : 2,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }
    VennDiagram(subset_sizes, ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_VennDiagram_from_sets():
    fig, axes = plt.subplots(1, 2)
    # canonical
    VennDiagram.from_sets([{0, 1}, {1, 2}], ax=axes[0])
    # all empty sets
    VennDiagram.from_sets([set(), set()], ax=axes[1])
    return fig


@pytest.mark.mpl_image_compare
def test_VennDiagram_as_wordcloud():
    text_1 = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
    enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
    ut aliquip ex ea commodo consequat."""

    text_2 = """Duis aute irure dolor in reprehenderit in voluptate velit
    esse cillum dolore eu fugiat nulla pariatur. Lorem ipsum dolor sit
    amet."""

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
    VennDiagram.as_wordcloud(sets, wordcloud_kwargs=dict(random_state=42), ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_cost_function_objective():
    subset_sizes = {
        (1, 0, 0) : 1,
        (0, 1, 0) : 1,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0.5,
        (1, 0, 1) : 0.5,
        (0, 1, 1) : 0.5,
        (1, 1, 1) : 0.25,
    }
    cost_function_objectives = [
        "simple",
        "squared",
        "logarithmic",
        "relative",
        "inverse",
    ]
    fig, axes = plt.subplots(1, len(cost_function_objectives), figsize=(15, 3))
    for cost_function_objective, ax in zip(cost_function_objectives, axes):
        EulerDiagram(subset_sizes, cost_function_objective=cost_function_objective, ax=ax)
        ax.set_title(cost_function_objective)
    fig.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_hide_empty_subsets():
    subset_sizes = {
        (1, 0, 0) : 1,
        (0, 1, 0) : 1,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0.5,
        (1, 0, 1) : 0.5,
        (0, 1, 1) : 0.5,
        (1, 1, 1) : 0,
    }
    fig, ax = plt.subplots()
    EulerDiagram(subset_sizes, cost_function_objective="simple", ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_three_way_euler_with_xor_set():
    # adapted from https://github.com/gecko984/supervenn
    set_1 = {1, 2}
    set_2 = {2, 3}
    set_3 = set_1 ^ set_2
    fig, ax = plt.subplots()
    EulerDiagram.from_sets([set_1, set_2, set_3], ax=ax)
    return fig
