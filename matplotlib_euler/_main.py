import warnings
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint
from shapely import intersection_all, union_all
from shapely.geometry import Point
from shapely.ops import polylabel


def get_subset_sizes(sets):
    """Creates a dictionary mapping subsets to set items. The
    subset IDs are tuples of booleans, with each boolean
    indicating if the corresponding input set is a superset of the
    subset or not.

    """
    output = dict()
    for subset_id in list(product(*len(sets) * [(False, True)])):
        if np.any(subset_id):
            output[subset_id] = len(
                set.intersection(*[sets[ii] for ii, include in enumerate(subset_id) if include]) \
                - set.union(*[sets[ii] for ii, include in enumerate(subset_id) if not include])
            )
    return output


class EulerDiagram(object):

    def __init__(self, subset_sizes, verbose=False, ax=None):
        self.subset_sizes = subset_sizes
        self.set_sizes = self._get_set_sizes()
        self._radii = self._get_radii()
        self._origins = self._get_origins(verbose=verbose)
        self._performance = self._evaluate(verbose=verbose)
        self.plot(ax=ax)


    def _get_set_sizes(self):
        return np.sum([size * np.array(subset) for subset, size in subset_sizes.items()], axis=0)


    def _get_radii(self):
        return np.array([np.sqrt(size / np.pi) for size in self.set_sizes])


    def _initialize_origins(self):
        """The optimisation procedure uses gradient descent to find
        the circle arrangement that best matches the desired subset
        areas. If a subset area is zero, there is no gradient to
        follow. It is hence paraount that all subset areas exist at
        initialization.

        Here, we evenly space the circle origins around the center of
        the diagram, such that their circumferences touch. We then
        shift each circle origin towards that center, such that all
        circles overlap.

        """
        x0, y0 = 0, 0 # venn diagram center
        total_sets = len(self.set_sizes)
        angles = 2 * np.pi * np.linspace(0, 1 - 1/total_sets, total_sets)
        overlap = 0.5 * np.min(self._radii)
        distances = self._radii - overlap
        x = x0 + distances * np.cos(angles)
        y = y0 + distances * np.sin(angles)
        return np.c_[x, y]


    def _get_subset_geometries(self, origins):
        set_geometries = [Point(*origin).buffer(radius) for origin, radius in zip(origins, self._radii)]
        output = dict()
        for subset in self.subset_sizes:
            include = intersection_all([set_geometries[ii] for ii, include in enumerate(subset) if include])
            exclude = union_all([set_geometries[ii] for ii, include in enumerate(subset) if not include])
            output[subset] = include.difference(exclude)
        return output


    def _get_subset_areas(self, origins):
        return np.array([geometry.area for geometry in self._get_subset_geometries(origins).values()])


    def _get_origins(self, verbose):

        desired_areas = np.array(list(self.subset_sizes.values()))

        def cost_function(flattened_origins):
            origins = flattened_origins.reshape(-1, 2)
            subset_areas = self._get_subset_areas(origins)

            # # Option 1: absolute difference
            # # Probably not the best choice, as small areas are often practically ignored.
            # cost = np.abs(subset_areas - desired_areas)

            # # Option 2: relative difference
            # # This often results in the optimization routine failing.
            # minimum_area = 1e-2 * np.pi * np.max(self._radii)**2
            # cost = np.abs(subset_areas - desired_areas) / np.clip(desired_areas, minimum_area, None)

            # Option 3: absolute difference of log(area + 1)
            # This transformation is monotonic increasing but strongly compresses large numbers,
            # thus allowing small numbers to carry more weight in the optimization.
            # cost = np.abs(np.log(subset_areas + 1) - np.log(desired_areas + 1))

            # Option 4: absolute difference of 1 / area
            # This transformation gives smaller subsets more weight such that
            # small or non-existant subsets are represented accurately.
            minimum_area = 1e-2 * np.pi * np.max(self._radii)**2
            cost = np.abs(1 / (subset_areas + minimum_area) - 1 / (desired_areas + minimum_area))

            return np.sum(cost)

        # constraints:
        eps = np.min(self._radii) * 0.001
        lower_bounds = np.abs(self._radii[np.newaxis, :] - self._radii[:, np.newaxis]) - eps
        lower_bounds[lower_bounds < 0] = 0
        lower_bounds = squareform(lower_bounds)

        upper_bounds = self._radii[np.newaxis, :] + self._radii[:, np.newaxis] + eps
        upper_bounds -= np.diag(np.diag(upper_bounds)) # squareform requires zeros on diagonal
        upper_bounds = squareform(upper_bounds)

        def constraint_function(flattened_origins):
            origins = np.reshape(flattened_origins, (-1, 2))
            return pdist(origins)

        distance_between_origins = NonlinearConstraint(
            constraint_function, lb=lower_bounds, ub=upper_bounds)

        result = minimize(
            cost_function,
            self._initialize_origins().flatten(),
            method='SLSQP', # 'COBYLA',
            constraints=[distance_between_origins],
            options=dict(disp=verbose, eps=eps)
        )

        if not result.success:
            print("Warning: could not compute circle positions for given subsets.")
            print(f"scipy.optimize.minimize: {result.message}.")

        origins = result.x.reshape((-1, 2))

        return origins


    def _evaluate(self, verbose):
        desired_areas = np.array(list(self.subset_sizes.values()))
        displayed_areas = self._get_subset_areas(self._origins)
        performance = {
            "Subset" : list(self.subset_sizes.keys()),
            "Desired area" : desired_areas,
            "Displayed area" : displayed_areas,
        }
        performance["Absolute difference"] = np.abs(desired_areas - displayed_areas)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            performance["Relative difference"] = np.abs(desired_areas - displayed_areas) / desired_areas

        if verbose:
            self._pretty_print_performance(performance)

        return performance


    def _pretty_print_performance(self, performance):
        paddings = [len(key) for key in performance]
        paddings[0] = max(paddings[0], len(str(performance["Subset"][0])))
        print()
        print(" | ".join([f"{item:>{pad}}" for item, pad in zip(performance.keys(), paddings)]))
        for row in zip(*performance.values()):
            print(" | ".join([f"{item:>{pad}.2f}" if isinstance(item, float) else f"{str(item):>{pad}}" for item, pad in zip(row, paddings)]))


    def _get_subset_label_positions(self):
        "For each non-zero subset, find the point of inaccesibility."
        subset_geometries = self._get_subset_geometries(self._origins)
        output = dict()
        for subset, geometry in subset_geometries.items():
            if geometry.area > 0:
                poi = polylabel(geometry)
                output[subset] = (poi.x, poi.y)
        return output


    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for origin, radius in zip(self._origins, self._radii):
            ax.add_patch(plt.Circle(origin, radius, alpha=1/len(self.set_sizes)))

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")

        label_positions = self._get_subset_label_positions()
        for subset, (x, y) in label_positions.items():
            ax.text(x, y, self.subset_sizes[subset], va="center", ha="center")


if __name__ == "__main__":

    # canonical 2-way Euler diagram
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }

    # a is superset of b
    subset_sizes = {
        (1, 0) : 1,
        (0, 1) : 0,
        (1, 1) : 0.5,
    }

    # canonical 3-way Euler diagram
    subset_sizes = {
        (1, 0, 0) : 1,
        (0, 1, 0) : 1,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0.5,
        (1, 0, 1) : 0.5,
        (0, 1, 1) : 0.5,
        (1, 1, 1) : 0.25,
    }

    # different areas
    subset_sizes = {
        (1, 0, 0) : 3,
        (0, 1, 0) : 2,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0.5,
        (1, 0, 1) : 0.4,
        (0, 1, 1) : 0.3,
        (1, 1, 1) : 0.2,
    }

    # no intersections
    subset_sizes = {
        (1, 0, 0) : 1,
        (0, 1, 0) : 1,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0,
        (1, 0, 1) : 0,
        (0, 1, 1) : 0,
        (1, 1, 1) : 0,
    }

    # a is superset to b, b is superset to c
    subset_sizes = {
        (1, 0, 0) : 3,
        (0, 1, 0) : 0,
        (0, 0, 1) : 0,
        (1, 1, 0) : 2,
        (1, 0, 1) : 0,
        (0, 1, 1) : 0,
        (1, 1, 1) : 1,
    }

    # a intersects b, b intersect c
    subset_sizes = {
        (1, 0, 0) : 1,
        (0, 1, 0) : 1,
        (0, 0, 1) : 1,
        (1, 1, 0) : 0.5,
        (1, 0, 1) : 0,
        (0, 1, 1) : 0.5,
        (1, 1, 1) : 0,
    }

    # no common intersection between all three
    subset_sizes = {
        (1, 0, 0) : 2,
        (0, 1, 0) : 2,
        (0, 0, 1) : 2,
        (1, 1, 0) : 0.1,
        (1, 0, 1) : 0.1,
        (0, 1, 1) : 0.1,
        (1, 1, 1) : 0,
    }

    # # # issue #30 <- raises AttributeError: 'MultiPolygon' object has no attribute 'exterior'
    # # subset_sizes = {
    # #     (1, 0, 0) : 1,
    # #     (0, 1, 0) : 0,
    # #     (0, 0, 1) : 0,
    # #     (1, 1, 0) : 32,
    # #     (1, 0, 1) : 0,
    # #     (0, 1, 1) : 76,
    # #     (1, 1, 1) : 13,
    # # }

    # subset_sizes = {
    #     (1, 0, 0) : 1,
    #     (0, 1, 0) : 0,
    #     (0, 0, 1) : 0,
    #     (1, 1, 0) : 650,
    #     (1, 0, 1) : 0,
    #     (0, 1, 1) : 76,
    #     (1, 1, 1) : 13,
    # }

    # subset_sizes = {
    #     (1, 0, 0) : 7549417,
    #     (0, 1, 0) : 15685620,
    #     (0, 0, 1) : 26018311,
    #     (1, 1, 0) : 5128906,
    #     (1, 0, 1) : 301048,
    #     (0, 1, 1) : 6841264,
    #     (1, 1, 1) : 2762301,
    # }

    # # issue #34
    # subset_sizes = {
    #     (1, 0, 0) : 167,
    #     (0, 1, 0) : 7,
    #     (0, 0, 1) : 25,
    #     (1, 1, 0) : 41,
    #     (1, 0, 1) : 174,
    #     (0, 1, 1) : 171,
    #     (1, 1, 1) : 51,
    # }

    # # issue # 50
    # subset_sizes = {
    #     (1, 0, 0) : 1808,
    #     (0, 1, 0) : 1181,
    #     (0, 0, 1) : 1858,
    #     (1, 1, 0) : 4715,
    #     (1, 0, 1) : 3031,
    #     (0, 1, 1) : 26482,
    #     (1, 1, 1) : 65012,
    # }

    EulerDiagram(subset_sizes, verbose=True)
    plt.show()
