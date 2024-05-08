#!/usr/bin/env python
"""
Layout optimisation
===================

Choosing the right Euler diagram cost function objective
--------------------------------------------------------

Sets are represented through overlapping circles, and the relative arrangement of these circles is determined through a minimisation procedure that attempts to match subset sizes to the corresponding areas formed by circle overlaps in the diagram.
However, it is not always possible to find a perfect solution. In these cases, the choice of cost function objective strongly determines which discrepancies between the subset sizes and the corresponding areas likely remain:

  - With the 'simple' cost function objective, the optimisation simply minimises the difference between the desired subset areas y and the current subset areas x (i.e. |x - y|). This is particularly useful when all subsets have similar sizes.
  - The 'squared' cost (i.e. (x - y)^2) penalises larger area discrepancies. Also particularly useful when all subsets have similar sizes.
  - The 'logarithmic' cost (i.e. |log(x + 1) - log(y + 1)|) scales strongly sublinearly with the size of the subset. This allows small subsets to affect the optimisation more strongly without assigning them the same weight as large subsets. This is useful when some subsets are much smaller than others.
  - The 'relative' cost (i.e. 1 - min(x/y, y/x)) assigns each subset equal weight. Small discrepancies in small subsets have the same impact as large discrepancies in large subsets.
  - The 'inverse' cost (i.e. |1 / (x + epsilon) - 1 / (y + epsilon)|) weighs small subsets stronger than large subsets. This is particularly useful when some theoretically possible subsets are absent. The epsilon parameter is arbitrarily set to 1% of the largest set size.

"""

import matplotlib.pyplot as plt

from matplotlib_set_diagrams import EulerDiagram

# 1. All subsets have the same size.
test_case_1 = {
    (1, 0, 0) : 1,
    (0, 1, 0) : 1,
    (0, 0, 1) : 1,
    (1, 1, 0) : 1,
    (1, 0, 1) : 1,
    (0, 1, 1) : 1,
    (1, 1, 1) : 1,
}

# 2. Equal set sizes, but differently sized subsets.
test_case_2 = {
    (1, 0, 0) : 1,
    (0, 1, 0) : 1,
    (0, 0, 1) : 1,
    (1, 1, 0) : 0.5,
    (1, 0, 1) : 0.5,
    (0, 1, 1) : 0.5,
    (1, 1, 1) : 0.25,
}

# 3. Some theoretically possible subsets are missing (easy optimisation problem).
test_case_3 = {
    (1, 0, 0) : 1,
    (0, 1, 0) : 1,
    (0, 0, 1) : 1,
    (1, 1, 0) : 0.5,
    (1, 0, 1) : 0,
    (0, 1, 1) : 0.25,
    (1, 1, 1) : 0,
}

# 4. Some theoretically possible subsets are missing (hard optimisation problem).
test_case_4 = {
    (1, 0, 0) : 1,
    (0, 1, 0) : 1,
    (0, 0, 1) : 1,
    (1, 1, 0) : 0.5,
    (1, 0, 1) : 0.5,
    (0, 1, 1) : 0.5,
    (1, 1, 1) : 0,
}

# 5. Some sets are fully contained within another.
test_case_5 = {
    (1, 0, 0) : 1,
    (0, 1, 0) : 0,
    (0, 0, 1) : 0,
    (1, 1, 0) : 0.5,
    (1, 0, 1) : 0,
    (0, 1, 1) : 0,
    (1, 1, 1) : 0.25,
}

# 6. Vastly different set sizes.
test_case_6 = {
    (1, 0, 0) : 2,
    (0, 1, 0) : 0.5,
    (0, 0, 1) : 0.1,
    (1, 1, 0) : 0.1,
    (1, 0, 1) : 0.1,
    (0, 1, 1) : 0.1,
    (1, 1, 1) : 0.1,
}

test_cases = [
    test_case_1,
    test_case_2,
    test_case_3,
    test_case_4,
    test_case_5,
    test_case_6,
]
cost_function_objectives = [
    "simple",
    "squared",
    "logarithmic",
    "relative",
    "inverse",
]

fig, axes = plt.subplots(6, 6, figsize=(15, 15))
fig.suptitle("Cost function objectives", fontweight="bold")

# plot desired subset sizes on the left
for ii, subset_sizes in enumerate(test_cases):
    EulerDiagram(subset_sizes, ax=axes[ii, 0])

    # draw the bounding box to highlight targets
    axes[ii, 0].axis("on")
    axes[ii, 0].set_xticks([])
    axes[ii, 0].set_yticks([])

axes[0, 0].set_title("Desired subset sizes")

# visualise difference between the given subset size and the actual area of the corresponding polygon
for ii, subset_sizes in enumerate(test_cases):
    for jj, cost_function_objective in enumerate(cost_function_objectives):
        diagram = EulerDiagram(subset_sizes, cost_function_objective=cost_function_objective, ax=axes[ii, jj+1])

        # determine actual subset areas
        subset_areas = dict()
        for subset_id, desired_size in subset_sizes.items():
            if subset_id in diagram.subset_geometries:
                subset_areas[subset_id] = diagram.subset_geometries[subset_id].area
            else: # subset area does not exist
                subset_areas[subset_id] = 0

        # amend subset labels to display difference between the given subset size and the actual area
        for subset_id in subset_sizes:
            desired = subset_sizes[subset_id]
            actual = subset_areas[subset_id]
            new_label = f"{desired-actual:.2f}"

            if subset_id in diagram.subset_label_artists:
                diagram.subset_label_artists[subset_id].set_text(new_label)

for jj, cost_function_objective in enumerate(cost_function_objectives):
    axes[0, jj+1].set_title(f"\"{cost_function_objective}\"")

fig.tight_layout()
fig.subplots_adjust(top=0.925)
plt.show()
