import numpy as np

from itertools import product
from matplotlib.colors import to_rgba

# type hinting
from typing import Tuple
from numpy.typing import NDArray
from matplotlib.typing import ColorType


def get_subset_ids(total_sets : int) -> list[Tuple[bool]]:
    """Given the number of sets, generate unique subset IDs for all
    potentially non-empty sets.

    Parameters
    ----------
    total_sets : int
        The number of sets.

    Returns
    -------
    subset_ids : list[Tuple[bool]]
        Subsets IDs are tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.

    """
    assert total_sets > 1, "Subset intersections can only exist for collections of more than one set."
    return [subset_id for subset_id in list(product(*total_sets * [(False, True)])) if np.any(subset_id)]


def get_subsets(sets : list[set]) -> dict[Tuple[bool], set]:
    """Given a list of sets, create a dictionary mapping subsets to set items.

    Parameters
    ----------
    sets : list[set]
        The sets.

    Returns
    -------
    subsets : dict[Tuple[bool], set]
        A dictionary mapping subset IDs to subset items.
        Subsets IDs are tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.

    """
    subsets = dict()
    for subset_id in get_subset_ids(len(sets)):
        include_elements = set.intersection(*[sets[ii] for ii, include in enumerate(subset_id) if include])
        exclude_elements = set.union(*[sets[ii] for ii, include in enumerate(subset_id) if not include]) if not np.all(subset_id) else set()
        subsets[subset_id] = include_elements - exclude_elements
    return subsets


def blend_colors(colors : list[ColorType], gamma : float = 2.2) -> NDArray:
    # Adapted from: https://stackoverflow.com/a/29321264/2912349
    rgba = np.array([to_rgba(color) for color in colors])
    rgb = np.power(np.mean(np.power(rgba[:, :3], gamma), axis=0), 1/gamma)
    a = np.mean(rgba[:, -1])
    return np.array([*rgb, a])


def rgba_to_grayscale(r : float, g : float, b : float, a : float = 1) -> float:
    # Adapted from: https://stackoverflow.com/a/689547/2912349
    return (0.299 * r + 0.587 * g + 0.114 * b) * a


def get_text_alignment(dx : float, dy : float) -> Tuple[str, str]:
    """For given arrow (dx, dy), determine the text alignment for a
    label placed at the arrow head such that the text does not overlap
    the arrow.

    """
    angle = np.arctan2(dy, dx) # radians
    angle = angle / (2.0 * np.pi) * 360 % 360 # degrees

    if (45 <= angle < 135):
        horizontalalignment = 'center'
        verticalalignment = 'bottom'
    elif (135 <= angle < 225):
        horizontalalignment = 'right'
        verticalalignment = 'center'
    elif (225 <= angle < 315):
        horizontalalignment = 'center'
        verticalalignment = 'top'
    else:
        horizontalalignment = 'left'
        verticalalignment = 'center'

    return horizontalalignment, verticalalignment
