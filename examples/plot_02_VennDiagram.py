#!/usr/bin/env python
"""
Venn diagrams
=============

Venn diagrams show all possible relationships of inclusion and exclusion between two or more sets.

The :py:class:`VennDiagram` class supports initialisation from

  - a dictionary mapping subsets to their sizes, and
  - directly from a list of sets.

In the latter case, the subset sizes will be computed internally.

Note that unlike in Euler diagrams, in Venn diagrams, the size of a
subset does not influence the size of the corresponding area in the
visualisation; only the subset labels are adjusted.

"""

import matplotlib.pyplot as plt

from matplotlib_set_diagrams import VennDiagram

fig, (ax1, ax2) = plt.subplots(1, 2)

VennDiagram(
    {
        (1, 0, 0) : 2, # {"a", "b"}
        (0, 1, 0) : 2, # {"e", "f"}
        (0, 0, 1) : 2, # {"h", "i"}
        (1, 1, 0) : 1, # {"c"}
        (1, 0, 1) : 1, # {"d"}
        (0, 1, 1) : 1, # {"g"}
        (1, 1, 1) : 0, # {}
    },
    ax=ax1)

VennDiagram.from_sets(
    [
        {"a", "b", "c", "d"},
        {"e", "f", "g", "c"},
        {"h", "i", "g", "d"},
    ],
    ax=ax2)

plt.show()
