import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint
from shapely.geometry import Point
from shapely.ops import polylabel
from shapely import intersection_all, union_all


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


class VennDiagram(object):

    def __init__(self, subset_sizes, verbose=False, ax=None):
        self.subset_sizes = subset_sizes
        self.set_sizes = self._get_set_sizes()
        self._radii = self._get_radii()
        self._origins = self._get_origins(verbose=verbose)
        self.plot(ax=ax)


    def _get_set_sizes(self):
        return np.sum([size * np.array(subset) for subset, size in subset_sizes.items()], axis=0)


    def _get_radii(self):
        return np.array([np.sqrt(size / np.pi) for size in self.set_sizes])


    def _initialize_origins(self):
        """Initialize origins on a small circle around (0, 0)."""

        origin = np.zeros((2))
        radius = np.min(self._radii)
        total_sets = len(self.set_sizes)
        angles = 2 * np.pi * np.linspace(0, 1 - 1/total_sets, total_sets)

        def get_point_on_a_circle(origin, radius, angle):
            """Compute the (x, y) coordinate of a point at a specified angle
            on a circle given by its (x, y) origin and radius."""
            x0, y0 = origin
            x = x0 + radius * np.cos(angle)
            y = y0 + radius * np.sin(angle)
            return np.array([x, y])

        return np.array(
            [get_point_on_a_circle(origin, radius, angle) for angle in angles]
        )


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

        desired_areas = np.array([size for size in self.subset_sizes.values()])

        def cost_function(flattened_origins):
            origins = flattened_origins.reshape(-1, 2)
            subset_areas = self._get_subset_areas(origins)

            # # Option 1: absolute difference
            # # Probably not the best choice, as small areas are often practically ignored.
            # cost = np.abs(subset_areas - desired_areas)

            # # Option 2: relative difference
            # # This often results in the optimization routine failing.
            # minimum_area = 1e-2
            # desired_areas[desired_areas < minimum_area] = minimum_area
            # cost = np.abs(subset_areas - desired_areas) / desired_areas

            # Option 3: absolute difference of log(area + 1)
            # This transformation is monotonic increasing but strongly compresses large numbers,
            # thus allowing small numbers to carry more weight in the optimization.
            cost = np.abs(np.log(subset_areas + 1) - np.log(desired_areas + 1))

            return np.sum(cost)

        # constraints:
        eps = np.min(self._radii) * 0.01
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
            options=dict(disp=verbose)
        )

        if not result.success:
            print("Warning: could not compute circle positions for given subsets.")
            print(f"scipy.optimize.minimize: {result.message}.")

        origins = result.x.reshape((-1, 2))

        if verbose:
            import pandas as pd
            data = pd.DataFrame(dict(desired=desired_areas, actual=self._get_subset_areas(origins)))
            data["absolute difference"] = np.abs(data["desired"] - data["actual"])
            data["relative difference"] = data["absolute difference"] / data["desired"]
            with pd.option_context('display.float_format', '{:0.1f}'.format):
                print(data)

        return origins


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

    # canonical 2-way Venn diagram
    subset_sizes = {
        (1, 0) : 2,
        (0, 1) : 1,
        (1, 1) : 0.5,
    }

    # # canonical 3-way Venn diagram
    # subset_sizes = {
    #     (1, 0, 0) : 1,
    #     (0, 1, 0) : 1,
    #     (0, 0, 1) : 1,
    #     (1, 1, 0) : 0.5,
    #     (1, 0, 1) : 0.5,
    #     (0, 1, 1) : 0.5,
    #     (1, 1, 1) : 0.25,
    # }
    VennDiagram(subset_sizes, verbose=True)
    plt.show()
