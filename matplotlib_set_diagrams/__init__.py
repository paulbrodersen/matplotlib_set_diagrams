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

A python library to plot [Venn
diagrams](https://en.wikipedia.org/wiki/Venn_diagram) and
area-proportional [Euler
diagrams](https://en.wikipedia.org/wiki/Euler_diagram). Euler diagrams
are similar to Venn diagrams, in as much as they visualize set
relationships. However, unlike Venn diagrams, which show all possible
relations between different sets, Euler diagrams show only relevant
relationships, i.e. non-empty subsets.

This library was inspired by
[matplotlib-venn](https://github.com/konstantint/matplotlib-venn), but
developed
[independently](https://github.com/konstantint/matplotlib-venn/issues/35).

"""

__version__ = "0.0.0"
__author__ = "Paul Brodersen"
__email__ = "paulbrodersen+matplotlib_set_diagrams@gmail.com"


from ._main import (
    EulerDiagramBase,
    EulerDiagram,
    EulerWordCloud,
)

__all__ = [
    "EulerDiagramBase",
    "EulerDiagram",
    "EulerWordCloud",
]
