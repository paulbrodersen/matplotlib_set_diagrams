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


class EulerDiagramBase(object):
    """Create an area-proportional Euler diagram visualising the relationships
    between two or more sets given the subset sizes.

    Sets are represented through overlapping circles, and the relative
    arrangement of these circles is determined through a minimisation
    procedure that attempts to match subset sizes to the corresponding
    areas formed by circle overlaps in the diagram. However, it is not
    always possible to find a perfect solution. In these cases, the
    choice of cost function objective strongly determines which
    discrepancies between the subset sizes and the corresponding areas
    likely remain:

    - With the 'simple' cost function objective, the optimisation simply
      minimizes the difference between the desired subset areas y and
      the current subset areas x (i.e. |x - y|). This is particularly
      useful when all subsets have similar sizes.
    - The 'squared' cost (i.e. (x - y)^2) penalises larger area discrepancies.
      Also particularly useful when all subsets have similar sizes.
    - The 'logarithmic' cost (i.e. |log(x + 1) - log(y + 1)|) scales strongly sublinearly
      with the size of the subset. This allows small subsets to affect the
      optimisation more strongly without assigning them the same weight as large subsets.
      This is useful when some subsets are much smaller than others.
    - The 'relative' cost (i.e. 1 - min(x/y, y/x)) assigns each subset equal weight.
    - The 'inverse' cost (i.e. |1 / (x + epsilon) - 1 / (y + epsilon)|)
      weighs small subsets stronger than large subsets. This is
      particularly useful when some theoretically possible subsets are
      absent. The epsilon parameter is arbitrarily set to 1% of the largest set size.

    Determining the best cost function objective for a given case can require
    some trial-and-error. If one or more subsets areas are not represented
    adequately, print the performance report (with :code:`verbose=True`).
    This will indicate for each subset the penalty on the remaining discrepancies,
    and thus facilitate choosing a more appropriate objective for the next iteration.

    Parameters
    ----------
    subset_sizes : dict[tuple[bool], float]
        A dictionary mapping each subset to its desired size.
        Subsets are represented by tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.
    subset_label_formatter : Optional[Callable]
        The formatter used to create subset labels based on the subset sizes.
        The function should accept an int or float and return a string.
    set_labels : list[str]
        A list of set labels.
        If none, defaults to the letters of the alphabet (capitalized).
    set_colors : Optional[list[Any]]
        A corresponding list of matplotlib colors.
        If none, defaults to the default matplotlib color cycle.
    cost_function_objective : str
        The cost function objective; one of:

        - 'simple'
        - 'squared'
        - 'logarithmic'
        - 'relative'
        - 'inverse'

    verbose : bool
        Print a report of the optimisation process.
    ax : matplotlib axis instance
        The axis to plot onto. If none, a new figure is instantiated.

    Attributes
    ----------
    subset_sizes : dict[tuple[bool], float]
        The dictionary mapping each subset to its desired size.
    set_sizes : list[int]
        The set sizes.
    radii : list[float]
        The radii of the corresponding circles.
    origins : list[float]
        The origins of the corresponding circles.
    performance : dict[str, list[float]]
        A table summarising the performance of the optimisation.
    subset_artists : dict[tuple[bool], matplotlib.patches.Polygon]
        The matplotlib Polygon patches representing each subset.
    subset_label_artists : dict[tuple[bool], matplotlib.text.Text]
        The matplotlib text objects used to label each subset.
    set_label_artists : list[matplotlib.text.Text]
        The matplotlib text objects used to label each set.

    """

    def __init__(
            self, subset_sizes,
            subset_label_formatter=lambda x : str(x),
            set_labels=None,
            set_colors=None,
            cost_function_objective="inverse",
            verbose=False,
            ax=None,
    ):
        self.subset_sizes = subset_sizes
        self.set_sizes = self._get_set_sizes()
        self.radii = self._get_radii()
        self.origins = self._get_origins(cost_function_objective, verbose=verbose)
        self._subset_geometries = self._get_subset_geometries(self.origins)
        self.performance = self._evaluate(verbose=verbose)
        self.ax = self._initialize_axis(ax=ax)
        self.subset_artists = self._draw_subsets(set_colors)
        self.subset_label_artists = self._draw_subset_labels(subset_label_formatter)
        self.set_label_artists = self._draw_set_labels(set_labels)


    def _get_set_sizes(self):
        """Compute the size of each set based on the sizes of its constituent sub-sets"""
        return np.sum([size * np.array(subset) for subset, size in self.subset_sizes.items()], axis=0)


    def _get_radii(self):
        """Map set sizes onto circle radii."""
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
        "Compute each subset polygon as a shapely geometry object."
        set_geometries = [Point(*origin).buffer(radius) for origin, radius in zip(origins, self.radii)]
        subset_geometries = dict()
        for subset in self.subset_sizes:
            include = intersection_all([set_geometries[ii] for ii, include in enumerate(subset) if include])
            exclude = union_all([set_geometries[ii] for ii, include in enumerate(subset) if not include])
            subset_geometries[subset] = include.difference(exclude)
        return subset_geometries


    def _get_origins(self, objective, verbose):
        """Optimize the placement of circle origins according to the
        given cost function objective.

        """
        desired_areas = np.array(list(self.subset_sizes.values()))

        def cost_function(flattened_origins):
            origins = flattened_origins.reshape(-1, 2)
            subset_areas = np.array([geometry.area for geometry in self._get_subset_geometries(origins).values()])

            if objective == "simple":
                cost = subset_areas - desired_areas
            elif objective == "squared":
                cost = (subset_areas - desired_areas)**2
            elif objective == "relative":
                cost = [1 - min(x/y, y/x) if x != y else 0. for x, y in zip(subset_areas, desired_areas)]
            elif objective == "logarithmic":
                cost = np.log(subset_areas + 1) - np.log(desired_areas + 1)
            elif objective == "inverse":
                eps = 1e-2 * np.pi * np.max(self.radii)**2
                cost = 1 / (subset_areas + eps) - 1 / (desired_areas + eps)
            else:
                msg = f"The provided cost function objective is not implemented: {objective}."
                msg += "\nAvailable objectives are: 'simple', 'squared', 'logarithmic', 'relative', and 'inverse'."
                raise NotImplementedError(msg)

            return np.sum(np.abs(cost))

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
            method='SLSQP',
            constraints=[distance_between_origins],
            options=dict(disp=verbose, eps=eps)
        )

        if not result.success:
            print("Warning: could not compute circle positions for given subsets.")
            print(f"scipy.optimize.minimize: {result.message}.")

        origins = result.x.reshape((-1, 2))

        return origins


    def _evaluate(self, verbose):
        """Collate the performance report."""
        desired_areas = np.array(list(self.subset_sizes.values()))
        subset_areas = np.array([geometry.area for geometry in self._subset_geometries.values()])
        performance = {
            "subset" : list(self.subset_sizes.keys()),
            "desired area" : desired_areas,
            "displayed area" : subset_areas,
        }

        def get_cost(objective):
            if objective == "simple":
                cost = subset_areas - desired_areas
            elif objective == "squared":
                cost = (subset_areas - desired_areas)**2
            elif objective == "relative":
                cost = [1 - min(x/y, y/x) if x != y else 0. for x, y in zip(subset_areas, desired_areas)]
            elif objective == "logarithmic":
                cost = np.log(subset_areas + 1) - np.log(desired_areas + 1)
            elif objective == "inverse":
                eps = 1e-2 * np.pi * np.max(self.radii)**2
                cost = 1 / (subset_areas + eps) - 1 / (desired_areas + eps)
            else:
                raise NotImplementedError
            return np.abs(cost)

        for objective in ["simple", "squared", "relative", "logarithmic", "inverse"]:
            performance[objective] = get_cost(objective)

        if verbose:
            self._pretty_print_performance(performance)

        return performance


    def _pretty_print_performance(self, performance):
        """Print the performance report."""
        paddings = [len(key) for key in performance]
        paddings[0] = max(len(str("subset")), len(str(performance["subset"][0]))) # the subset IDs are often longer than the string "subset"
        print()
        print(" | ".join([f"{item:>{pad}}" for item, pad in zip(performance.keys(), paddings)]))
        for row in zip(*performance.values()):
            print(" | ".join([f"{item:>{pad}.2f}" if isinstance(item, float) else f"{str(item):>{pad}}" for item, pad in zip(row, paddings)]))
        print()


    def _initialize_axis(self, ax=None):
        """Initialize the axis if none provided. Ensure that the
        aspect is equal such that circles are circles and not
        ellipses.

        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.axis("off")
        return ax


    def _get_subset_colors(self, set_colors=None):
        if not set_colors:
            set_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subset_colors = dict()
        for subset in self.subset_sizes:
            subset_colors[subset] = blend_colors([set_colors[ii] for ii, is_superset in enumerate(subset) if is_superset])
        return subset_colors


    def _draw_subsets(self):
        """Draw each subset as a separate polygon patch."""
        subset_artists = dict()
        for subset, geometry in self._subset_geometries.items():
            if geometry.area > 0:
                artist = plt.Polygon(geometry.exterior.coords, color=self.subset_colors[subset])
                self.ax.add_patch(artist)
                subset_artists[subset] = artist
        self.ax.autoscale_view()
        return subset_artists


    def _draw_subset_labels(self, formatter):
        """Map subset sizes to strings using the provided formatter
        and then place at them centred on the point of inaccesibility
        (POI) of the corresponding polygon.

        """
        subset_label_artists = dict()
        for subset, geometry in self._subset_geometries.items():
            if geometry.area > 0:
                poi = polylabel(geometry)
                label = formatter(self.subset_sizes[subset])
                subset_color = to_rgba(self.subset_artists[subset].get_facecolor())
                color = "black" if rgba_to_grayscale(*subset_color) > 0.5 else "white"
                subset_label_artists[subset] = self.ax.text(
                    poi.x, poi.y, label,
                    color=color,
                    va="center", ha="center")
        return subset_label_artists


    def _draw_set_labels(self, set_labels, offset=0.1):
        """Place the set label on the side opposite to the centroid of
        all other sets.

        """
        if not set_labels:
            set_labels = string.ascii_uppercase[:len(self.set_sizes)]

        set_label_artists = []
        for ii, label in enumerate(set_labels):
            delta = self.origins[ii] - np.mean([origin for jj, origin in enumerate(self.origins) if ii != jj])
            x, y = self.origins[ii] + (1 + offset) * self.radii[ii] * delta / np.linalg.norm(delta)
            ha, va = get_text_alignment(*delta)
            set_label_artists.append(self.ax.text(x, y, label, ha=ha, va=va))
        return set_label_artists


class EulerDiagram(EulerDiagramBase):
    """Create an area-proportional Euler diagram visualising the relationships
    between two or more sets.

    Sets are represented through overlapping circles, and the relative
    arrangement of these circles is determined through a minimisation
    procedure that attempts to match subset sizes to the corresponding
    areas formed by circle overlaps in the diagram. However, it is not
    always possible to find a perfect solution. In these cases, the
    choice of cost function objective strongly determines which
    discrepancies between the subset sizes and the corresponding areas
    likely remain:

    - With the 'simple' cost function objective, the optimisation simply
      minimizes the difference between the desired subset areas y and
      the current subset areas x (i.e. |x - y|). This is particularly
      useful when all subsets have similar sizes.
    - The 'squared' cost (i.e. (x - y)^2) penalises larger area discrepancies.
      Also particularly useful when all subsets have similar sizes.
    - The 'logarithmic' cost (i.e. |log(x + 1) - log(y + 1)|) scales strongly sublinearly
      with the size of the subset. This allows small subsets to affect the
      optimisation more strongly without assigning them the same weight as large subsets.
      This is useful when some subsets are much smaller than others.
    - The 'relative' cost (i.e. 1 - min(x/y, y/x)) assigns each subset equal weight.
    - The 'inverse' cost (i.e. |1 / (x + epsilon) - 1 / (y + epsilon)|)
      weighs small subsets stronger than large subsets. This is
      particularly useful when some theoretically possible subsets are
      absent. The epsilon parameter is arbitrarily set to 1% of the largest set size.

    Determining the best cost function objective for a given case can require
    some trial-and-error. If one or more subsets areas are not represented
    adequately, print the performance report (with :code:`verbose=True`).
    This will indicate for each subset the penalty on the remaining discrepancies,
    and thus facilitate choosing a more appropriate objective for the next iteration.

    Parameters
    ----------
    sets : list[set]
        The sets.
    subset_label_formatter : Optional[Callable]
        The formatter used to create subset labels based on the subset sizes.
        The function should accept an int or float and return a string.
    set_labels : list[str]
        A list of set labels.
        If none, defaults to the letters of the alphabet (capitalized).
    set_colors : Optional[list[Any]]
        A corresponding list of matplotlib colors.
        If none, defaults to the default matplotlib color cycle.
    cost_function_objective : str
        The cost function objective; one of:

        - 'simple'
        - 'squared'
        - 'logarithmic'
        - 'relative'
        - 'inverse'

    verbose : bool
        Print a report of the optimisation process.
    ax : matplotlib axis instance
        The axis to plot onto. If none, a new figure is instantiated.

    Attributes
    ----------
    subset_sizes : dict[tuple[bool, ...], float]
        The dictionary mapping each subset to its desired size.
        Subsets are represented by tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.
    set_sizes : list[int]
        The set sizes.
    radii : list[float]
        The radii of the corresponding circles.
    origins : list[float]
        The origins of the corresponding circles.
    performance : dict[str, list[float]]
        A table summarising the performance of the optimisation.
    subset_artists : dict[tuple[bool], matplotlib.patches.Polygon]
        The matplotlib Polygon patches representing each subset.
    subset_label_artists : dict[tuple[bool], matplotlib.text.Text]
        The matplotlib text objects used to label each subset.
    set_label_artists : list[matplotlib.text.Text]
        The matplotlib text objects used to label each set.

    """
    def __init__(self, sets, *args, **kwargs):
        self.sets = [set(item) for item in sets]
        subset_sizes = self._get_subset_sizes()
        super().__init__(subset_sizes, *args, **kwargs)


    def _get_subset_sizes(self):
        """Creates a dictionary mapping subsets to subset size. The
        subset IDs are tuples of booleans, with each boolean
        indicating if the corresponding input set is a superset of the
        subset or not.

        """
        subset_size = dict()
        for subset_id in list(product(*len(self.sets) * [(False, True)])):
            if np.any(subset_id):
                include_elements = set.intersection(*[self.sets[ii] for ii, include in enumerate(subset_id) if include])
                exclude_elements = set.union(*[self.sets[ii] for ii, include in enumerate(subset_id) if not include]) if not np.all(subset_id) else set()
                subset_size[subset_id] = len(include_elements - exclude_elements)
        return subset_size





    # subset_sizes = {
    # }

    # subset_sizes = {
    # }

    # subset_sizes = {
    # }
    plt.show()
