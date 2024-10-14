#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Matplotlib Set Diagrams --- Venn and Euler diagrams in Python.

# Copyright (C) 2024 Paul Brodersen <paulbrodersen+matplotlib_set_diagrams@gmail.com>

# Author: Paul Brodersen <paulbrodersen+matplotlib_set_diagrams@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Matplotlib Set Diagrams
=======================

*Draw Euler diagrams and Venn diagrams with Matplotlib.*

[Euler](https://en.wikipedia.org/wiki/Euler_diagram) and [Venn](https://en.wikipedia.org/wiki/Venn_diagram) diagrams are used to visualise the relationships between sets. Both typically employ circles to represent sets, and areas where two circles overlap represent subsets common to both supersets.
Venn diagrams show all possible relationships of inclusion and exclusion between two or more sets.
In Euler diagrams, the area corresponding to each subset is scaled according to the size of the subset. If a subset doesn't exist, the corresponding area doesn't exist.

This library was inspired by [matplotlib-venn](https://github.com/konstantint/matplotlib-venn), but developed independently.

"""

__version__ = "0.1.0"
__author__ = "Paul Brodersen"
__email__ = "paulbrodersen+matplotlib_set_diagrams@gmail.com"


from ._diagram_classes import (
    EulerDiagram,
    VennDiagram,
)

from ._utils import (
    get_subset_ids,
    get_subsets,
)

__all__ = [
    "EulerDiagram",
    "VennDiagram",
    "get_subset_ids",
    "get_subsets",
]
