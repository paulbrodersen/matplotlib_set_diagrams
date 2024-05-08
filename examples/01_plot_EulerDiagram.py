#!/usr/bin/env python
"""
Euler diagrams
==============

Euler diagrams are area-proportional set diagrams, i.e. each area is proportional to the size of the corresponding subset.

The :code::`EulerDiagram` class supports initialisation from

  - a dictionary mapping subsets to their sizes, and
  - directly from a list of sets.

In the latter case, the subset sizes will be computed internally.
"""

import matplotlib.pyplot as plt

from matplotlib_set_diagrams import EulerDiagram

fig, (ax1, ax2) = plt.subplots(1, 2)

EulerDiagram(
    {
        (1, 0) : 3, # {"a", "b", "c"}
        (0, 1) : 1, # {"e"}
        (1, 1) : 1, # {"d"}
    },
    ax=ax1)

EulerDiagram.from_sets(
    [
        {"a", "b", "c", "d"},
        {"d", "e"},
    ],
    ax=ax2)

plt.show()
