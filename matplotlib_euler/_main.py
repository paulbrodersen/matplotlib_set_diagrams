import warnings
import string
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint
from shapely import intersection_all, union_all
from shapely.geometry import Point
from shapely.ops import polylabel
from matplotlib.colors import to_rgba, to_rgba_array


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


def blend_colors(colors, gamma=2.2):
    # Adapted from: https://stackoverflow.com/a/29321264/2912349
    rgba = to_rgba_array(colors)
    rgb = np.power(np.mean(np.power(rgba[:, :3], gamma), axis=0), 1/gamma)
    a = np.mean(rgba[:, -1])
    return np.array([*rgb, a])


def rgba_to_grayscale(r, g, b, a=1):
    # Adapted from: https://stackoverflow.com/a/689547/2912349
    return (0.299 * r + 0.587 * g + 0.114 * b) * a


def get_text_alignment(dx, dy):
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


class EulerDiagram(object):

    def __init__(self, subset_sizes, set_labels=None, set_colors=None, verbose=False, ax=None):
        self.subset_sizes = subset_sizes
        self.set_sizes = self._get_set_sizes()
        self.radii = self._get_radii()
        self.origins = self._get_origins(verbose=verbose)
        self._subset_geometries = self._get_subset_geometries(self.origins)
        self.performance = self._evaluate(verbose=verbose)
        self.ax = self._initialize_axis(ax=ax)
        self.subset_artists = self._draw_subsets(set_colors)
        self.subset_label_artists = self._draw_subset_labels()
        self.set_label_artists = self._draw_set_labels(set_labels)


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
        overlap = 0.5 * np.min(self.radii)
        distances = self.radii - overlap
        x = x0 + distances * np.cos(angles)
        y = y0 + distances * np.sin(angles)
        return np.c_[x, y]


    def _get_subset_geometries(self, origins):
        set_geometries = [Point(*origin).buffer(radius) for origin, radius in zip(origins, self.radii)]
        output = dict()
        for subset in self.subset_sizes:
            include = intersection_all([set_geometries[ii] for ii, include in enumerate(subset) if include])
            exclude = union_all([set_geometries[ii] for ii, include in enumerate(subset) if not include])
            output[subset] = include.difference(exclude)
        return output


    def _get_origins(self, verbose):

        desired_areas = np.array(list(self.subset_sizes.values()))

        def cost_function(flattened_origins):
            origins = flattened_origins.reshape(-1, 2)
            subset_areas = np.array([geometry.area for geometry in self._get_subset_geometries(origins).values()])

            # # Option 1: absolute difference
            # # Probably not the best choice, as small areas are often practically ignored.
            # cost = np.abs(subset_areas - desired_areas)

            # # Option 2: relative difference
            # # This often results in the optimization routine failing.
            # minimum_area = 1e-2 * np.pi * np.max(self.radii)**2
            # cost = np.abs(subset_areas - desired_areas) / np.clip(desired_areas, minimum_area, None)

            # Option 3: absolute difference of log(area + 1)
            # This transformation is monotonic increasing but strongly compresses large numbers,
            # thus allowing small numbers to carry more weight in the optimization.
            # cost = np.abs(np.log(subset_areas + 1) - np.log(desired_areas + 1))

            # Option 4: absolute difference of 1 / area
            # This transformation gives smaller subsets more weight such that
            # small or non-existant subsets are represented accurately.
            minimum_area = 1e-2 * np.pi * np.max(self.radii)**2
            cost = np.abs(1 / (subset_areas + minimum_area) - 1 / (desired_areas + minimum_area))

            return np.sum(cost)

        # constraints:
        eps = np.min(self.radii) * 0.001
        lower_bounds = np.abs(self.radii[np.newaxis, :] - self.radii[:, np.newaxis]) - eps
        lower_bounds[lower_bounds < 0] = 0
        lower_bounds = squareform(lower_bounds)

        upper_bounds = self.radii[np.newaxis, :] + self.radii[:, np.newaxis] + eps
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
        displayed_areas = np.array([geometry.area for geometry in self._subset_geometries.values()])
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


    def _initialize_axis(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.axis("off")
        return ax


    def _draw_subsets(self, set_colors=None):
        if not set_colors:
            set_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subset_artists = dict()
        for subset, geometry in self._subset_geometries.items():
            if geometry.area > 0:
                color = blend_colors([set_colors[ii] for ii, is_superset in enumerate(subset) if is_superset])
                artist = plt.Polygon(geometry.exterior.coords, color=color)
                self.ax.add_patch(artist)
                subset_artists[subset] = artist
        self.ax.autoscale_view()
        return subset_artists


    def _draw_subset_labels(self):
        subset_label_artists = dict()
        for subset, geometry in self._subset_geometries.items():
            if geometry.area > 0:
                poi = polylabel(geometry) # point of inaccesibility
                label = self.subset_sizes[subset]
                subset_color = to_rgba(self.subset_artists[subset].get_facecolor())
                color = "black" if rgba_to_grayscale(*subset_color) > 0.5 else "white"
                subset_label_artists[subset] = self.ax.text(
                    poi.x, poi.y, label,
                    color=color,
                    va="center", ha="center")
        return subset_label_artists


    def _draw_set_labels(self, set_labels, offset=0.1):
        """Place the set label on the side opposite to the centroid of all other sets."""
        if not set_labels:
            set_labels = string.ascii_uppercase[:len(self.set_sizes)]

        set_label_artists = []
        for ii, label in enumerate(set_labels):
            delta = self.origins[ii] - np.mean([origin for jj, origin in enumerate(self.origins) if ii != jj])
            x, y = self.origins[ii] + (1 + offset) * self.radii[ii] * delta / np.linalg.norm(delta)
            ha, va = get_text_alignment(delta)
            set_label_artists.append(self.ax.text(x, y, label, ha=ha, va=va))
        return set_label_artists


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
