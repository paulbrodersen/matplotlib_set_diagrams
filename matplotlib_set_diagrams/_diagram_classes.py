import numpy as np
import matplotlib.pyplot as plt
import warnings

from string import ascii_uppercase
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint
from shapely import intersection_all, union_all
from shapely.geometry import Point
from shapely.ops import polylabel
from matplotlib.colors import to_rgba
from matplotlib.path import Path
from matplotlib.collections import PolyCollection
from wordcloud import WordCloud

from ._utils import (
    get_subset_ids,
    get_subsets,
    blend_colors,
    rgba_to_grayscale,
    get_text_alignment,
)

# type hinting
from typing import (
    Any,
    Tuple,
    Optional,
    Callable,
    Mapping,
    Union,
)
from numpy.typing import NDArray
from matplotlib.typing import ColorType
from matplotlib.image import AxesImage
from shapely import (
    Polygon as ShapelyPolygon,
    MultiPolygon as ShapelyMultiPolygon,
)


class SetDiagram:
    """Draw a diagram visualising the relationships between two or
    more sets using two or more overlapping circles.

    origins : NDArray
        The circle origins.
    radii : NDArray
        The circle radii.
    subset_labels : Optional[Mapping[Tuple[bool], str]]
        A dictionary mapping subsets to their labels or None. If None, no subset labels are created.
        Subsets are represented by tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.
    set_labels : Optional[list[str]]
        A list of set labels.
        If None, no subset labels are created.
    set_colors : Optional[list[ColorType]]
        A corresponding list of matplotlib colors.
        If none, defaults to the default matplotlib color cycle.
    ax : Optional[plt.Axes]
        The matplotlib axis instance to draw onto.
        If none provided, a new figure with a single axis is instantiated.

    Attributes
    ----------
    subset_geometries : dict[Tuple[bool], shapely.geometry.polygon.Polygon]
        The dictionary mapping each subset to its shapely geometry.
    subset_artists : dict[tuple[bool], plt.Polygon]
        The matplotlib Polygon patches representing each subset.
    subset_label_artists : dict[tuple[bool], plt.Text]
        The matplotlib text objects used to label each subset.
    set_label_artists : list[plt.Text]
        The matplotlib text objects used to label each set.
    ax : plt.Axes
        The matplotlib axis instance.

    """
    def __init__(
            self,
            origins       : NDArray,
            radii         : NDArray,
            subset_labels : Optional[Mapping[Tuple[bool], str]] = None,
            set_labels    : Optional[list[str]]                 = None,
            set_colors    : Optional[list]                      = None,
            ax            : Optional[plt.Axes]                  = None,
    ) -> None:

        subset_ids = get_subset_ids(len(origins))
        self.subset_geometries : ShapelyPolygon = \
            self._get_subset_geometries(subset_ids, origins, radii)
        self.subset_colors = self._get_subset_colors(subset_ids, set_colors)
        self.ax = self._initialize_axis(ax=ax)
        self.subset_artists = self._draw_subsets(
            self.subset_geometries, self.subset_colors, self.ax)

        if subset_labels:
            self.subset_label_artists = self._draw_subset_labels(
                subset_labels, self.subset_geometries, self.subset_colors, self.ax)

        if set_labels:
            self.set_label_artists = self._draw_set_labels(
                set_labels, origins, radii, self.ax)


    def _get_subset_geometries(
            self,
            subsets : list[Tuple[bool]],
            origins : NDArray,
            radii   : NDArray
    ) -> dict[Tuple[bool], ShapelyPolygon]:
        "Compute each subset polygon as a shapely geometry object."
        set_geometries = [Point(*origin).buffer(radius) for origin, radius in zip(origins, radii)]
        subset_geometries = dict()
        for subset in subsets:
            include = intersection_all([set_geometries[ii] for ii, include in enumerate(subset) if include])
            exclude = union_all([set_geometries[ii] for ii, include in enumerate(subset) if not include])
            subset_geometries[subset] = include.difference(exclude)
        return subset_geometries


    def _get_subset_colors(
            self,
            subsets    : list[Tuple[bool]],
            set_colors : Optional[list[ColorType]] = None,
    ) -> dict[Tuple[bool], NDArray]:
        """Determine the color of each subset patch based on the colors of the overlapping sets."""
        if set_colors is None:
            set_colors = plt.rcParamsDefault['axes.prop_cycle'].by_key()['color']
        subset_colors = dict()
        for subset in subsets:
            subset_colors[subset] = blend_colors([set_colors[ii] for ii, is_superset in enumerate(subset) if is_superset])
        return subset_colors


    def _initialize_axis(self, ax : Optional[plt.Axes] = None) -> plt.Axes:
        """Initialize the axis if none provided. Ensure that the
        aspect is equal such that circles are circles and not
        ellipses.

        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.axis("off")
        return ax


    def _draw_subsets(
            self,
            subset_geometries : Mapping[Tuple[bool], ShapelyPolygon],
            subset_colors     : Mapping[Tuple[bool], NDArray],
            ax                : plt.Axes,
    ) -> dict[Tuple[bool], Union[plt.Polygon, PolyCollection]]:
        """Draw each subset as a separate polygon patch."""
        subset_artists : dict[Tuple[bool], Union[plt.Polygon, PolyCollection]] = dict()
        for subset, geometry in subset_geometries.items():
            if geometry.area > 0:
                if isinstance(geometry, ShapelyPolygon):
                    polygon = plt.Polygon(geometry.exterior.coords, color=subset_colors[subset])
                    ax.add_patch(polygon)
                    subset_artists[subset] = polygon
                elif isinstance(geometry, ShapelyMultiPolygon):
                    polygon_collection = PolyCollection([geom.exterior.coords for geom in geometry.geoms], color=subset_colors[subset])
                    ax.add_collection(polygon_collection)
                    subset_artists[subset] = polygon_collection
                else:
                    raise TypeError(f"Shapely returned neither a Polygon or MultiPolygon but instead {type(geometry)} object!")
        ax.autoscale_view()
        return subset_artists


    def _draw_subset_labels(
            self,
            subset_labels     : Mapping[Tuple[bool], str],
            subset_geometries : Mapping[Tuple[bool], ShapelyPolygon],
            subset_colors     : Mapping[Tuple[bool], NDArray],
            ax                : plt.Axes,
    ) -> dict[Tuple[bool], plt.Text]:
        """Place subset labels centred on the point of inaccesibility
        (POI) of the corresponding polygon.
        """
        subset_label_artists = dict()
        for subset, label in subset_labels.items():
            geometry = subset_geometries[subset]
            if geometry.area > 0:
                if isinstance(geometry, ShapelyPolygon):
                    poi = polylabel(geometry)
                elif isinstance(geometry, ShapelyMultiPolygon):
                    # use largest sub-geometry
                    poi = polylabel(max(geometry.geoms, key=lambda x:x.area))
                else:
                    raise TypeError(f"Shapely returned neither a Polygon or MultiPolygon but instead {type(geometry)} object!")
                fontcolor = "black" if rgba_to_grayscale(*subset_colors[subset]) > 0.5 else "white"
                subset_label_artists[subset] = ax.text(
                    poi.x, poi.y, label,
                    color=fontcolor, va="center", ha="center"
                )
        return subset_label_artists


    def _draw_set_labels(
            self,
            set_labels : list[str],
            origins    : NDArray,
            radii      : NDArray,
            ax         : plt.Axes,
            offset     : float = 0.1,
    ) -> list[plt.Text]:
        """Place the set label on the side opposite to the centroid of all other sets."""
        set_label_artists = []
        for ii, label in enumerate(set_labels):
            delta = origins[ii] - np.mean([origin for jj, origin in enumerate(origins) if ii != jj], axis=0)
            x, y = origins[ii] + (1 + offset) * radii[ii] * delta / np.linalg.norm(delta)
            ha, va = get_text_alignment(*delta)
            set_label_artists.append(ax.text(x, y, label, ha=ha, va=va))
        return set_label_artists


class EulerDiagram(SetDiagram):
    """Create an area-proportional Euler diagram visualising the relationships
    between two or more sets given the subset sizes.

    Sets are represented through overlapping circles, and the relative
    arrangement of these circles is determined through a minimisation
    procedure that attempts to match subset sizes to the corresponding
    areas formed by circle overlaps in the diagram.

    Parameters
    ----------
    subset_sizes : Mapping[Tuple[bool], int | float]
        A dictionary mapping each subset to its desired size.
        Subsets are represented by tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.
    subset_labels : Optional[Mapping[Tuple[bool], str]]
        A dictionary mapping each subset to its desired label or None. If None,
        the subset_label_formatter is used create subset labels based on the subset sizes.
    subset_label_formatter : Callable[[Tuple[bool], int | float], str]
        The formatter used to create subset labels based on the subset sizes.
        The argument is ignored if subset_labels are not None.
    set_labels : Optional[list[str]]
        A list of set labels.
        If none, defaults to the letters of the alphabet (capitalized).
    set_colors : Optional[list[ColorType]]
        A corresponding list of matplotlib colors.
        If none, defaults to the default matplotlib color cycle.
    cost_function_objective : str
        The cost function objective; one of:

        - 'simple'      : :code:`|x - y|`
        - 'squared'     : :code:`(x - y)^2`
        - 'logarithmic' : :code:`|log(x + 1) - log(y + 1)|`
        - 'relative'    : :code:`1 - min(x/y, y/x)`
        - 'inverse'     : :code:`|1 / (x + epsilon) - 1 / (y + epsilon)|`

    verbose : bool
        Print a report of the optimisation process.
    ax : Optional[plt.Axes]
        The matplotlib axis instance to draw onto.
        If none provided, a new figure with a single axis is instantiated.

    Attributes
    ----------
    origins : NDArray
        The circle origins.
    radii : NDArray
        The circle radii.
    subset_geometries : dict[Tuple[bool], shapely.geometry.polygon.Polygon]
        The dictionary mapping each subset to its shapely geometry.
    subset_artists : dict[tuple[bool], plt.Polygon]
        The matplotlib Polygon patches representing each subset.
    subset_label_artists : dict[tuple[bool], plt.Text]
        The matplotlib text objects used to label each subset.
    set_label_artists : list[plt.Text]
        The matplotlib text objects used to label each set.
    ax : plt.Axes
        The matplotlib axis instance.

    """

    def __init__(
            self,
            subset_sizes            : Mapping[Tuple[bool], Union[int, float]],
            subset_labels           : Optional[Mapping[Tuple[bool], str]]       = None,
            subset_label_formatter  : Callable[[Tuple[bool], Union[int, float]], str] = lambda subset, size : str(size),
            set_labels              : Optional[list[str]]                       = None,
            set_colors              : Optional[list[ColorType]]                 = None,
            cost_function_objective : str                                       = "inverse",
            verbose                 : bool                                      = False,
            ax                      : Optional[plt.Axes]                        = None,
    ) -> None:

        self.origins, self.radii = self._get_layout(
            subset_sizes, cost_function_objective, verbose)

        if subset_labels is None:
            subset_labels = self._get_subset_labels(
                subset_sizes, subset_label_formatter)

        if set_labels is None:
            set_labels = self._get_set_labels(len(self.origins))

        super().__init__(
            self.origins, self.radii,
            subset_labels = subset_labels,
            set_labels    = set_labels,
            set_colors    = set_colors,
            ax            = ax,
        )

        self._hide_empty_subsets(subset_sizes)


    def _get_layout(
            self,
            subset_sizes : Mapping[Tuple[bool], Union[int, float]],
            cost_function_objective : str,
            verbose : bool
    ) -> Tuple[NDArray, NDArray]:
        origins, radii = self._initialize_layout(subset_sizes)
        origins, radii = self._optimize_layout(subset_sizes, origins, radii,
                                               cost_function_objective,
                                               verbose=verbose)
        return origins, radii


    def _initialize_layout(self, subset_sizes : Mapping[Tuple[bool], Union[int, float]]) -> Tuple[NDArray, NDArray]:
        set_sizes = self._get_set_sizes(subset_sizes)
        radii = self._initialize_radii(set_sizes)
        origins = self._initialize_origins(radii)
        return origins, radii


    def _get_set_sizes(self, subset_sizes : Mapping[Tuple[bool], Union[int, float]]) -> NDArray:
        """Compute the size of each set based on the sizes of its constituent sub-sets"""
        return np.sum([size * np.array(subset) for subset, size in subset_sizes.items()], axis=0)


    def _initialize_radii(self, areas : NDArray) -> NDArray:
        """Map set sizes onto circle radii."""
        return np.array([np.sqrt(area / np.pi) for area in areas])


    def _initialize_origins(self, radii : NDArray) -> NDArray:
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
        x0, y0 = 0, 0 # diagram center
        total_sets = len(radii)
        angles = 2 * np.pi * np.linspace(0, 1 - 1/total_sets, total_sets)
        angles += np.pi # place origin of first set on the left, not the right
        overlap = 0.5 * np.min(radii)
        distances = radii - overlap
        x = x0 + distances * np.cos(angles)
        y = y0 + distances * np.sin(angles)
        return np.c_[x, y]


    def _optimize_layout(
            self,
            subset_sizes : Mapping[Tuple[bool], Union[int, float]],
            origins      : NDArray,
            radii        : NDArray,
            objective    : str,
            verbose      : bool
    ) -> Tuple[NDArray, NDArray]:
        """Optimize the placement of circle origins according to the
        given cost function objective.

        """
        desired_areas = np.array(list(subset_sizes.values()))

        def cost_function(flattened_origins):
            origins = flattened_origins.reshape(-1, 2)
            subset_areas = np.array(
                [geometry.area for geometry in self._get_subset_geometries(subset_sizes.keys(), origins, radii).values()]
            )

            if objective == "simple":
                cost = subset_areas - desired_areas
            elif objective == "squared":
                cost = (subset_areas - desired_areas)**2
            elif objective == "relative":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
                    cost = [1 - min(x/y, y/x) if x != y else 0. for x, y in zip(subset_areas, desired_areas)]
            elif objective == "logarithmic":
                cost = np.log(subset_areas + 1) - np.log(desired_areas + 1)
            elif objective == "inverse":
                eps = 1e-2 * np.sum(desired_areas)
                cost = 1 / (subset_areas + eps) - 1 / (desired_areas + eps)
            else:
                msg = f"The provided cost function objective is not implemented: {objective}."
                msg += "\nAvailable objectives are: 'simple', 'squared', 'logarithmic', 'relative', and 'inverse'."
                raise ValueError(msg)

            return np.sum(np.abs(cost))

        # constraints:
        eps = np.min(radii) * 0.01
        lower_bounds = np.abs(radii[np.newaxis, :] - radii[:, np.newaxis]) - eps
        lower_bounds[lower_bounds < 0] = 0
        lower_bounds = squareform(lower_bounds)

        upper_bounds = radii[np.newaxis, :] + radii[:, np.newaxis] + eps
        upper_bounds -= np.diag(np.diag(upper_bounds)) # squareform requires zeros on diagonal
        upper_bounds = squareform(upper_bounds)

        def constraint_function(flattened_origins):
            origins = np.reshape(flattened_origins, (-1, 2))
            return pdist(origins)

        distance_between_origins = NonlinearConstraint(
            constraint_function, lb=lower_bounds, ub=upper_bounds)

        result = minimize(
            cost_function,
            origins.flatten(),
            method='SLSQP',
            constraints=[distance_between_origins],
            options=dict(disp=verbose, eps=eps)
        )

        if not result.success:
            feedback = "Could not optimise layout for the given subsets. Try a different cost function objective."
            warnings.warn(f"{result.message}. {feedback}")

        origins = result.x.reshape((-1, 2))

        return origins, radii


    def _get_set_labels(self, total_sets : int) -> list[str]:
        return [char for char in ascii_uppercase[:total_sets]]


    def _get_subset_labels(
            self,
            subset_sizes : Mapping[Tuple[bool], Union[int, float]],
            formatter    : Callable[[Tuple[bool], Union[int, float]], str],
    ) -> dict[Tuple[bool], str]:
        """Map subset sizes to strings using the provided formatter."""
        subset_labels = dict()
        for subset, size in subset_sizes.items():
            subset_labels[subset] = formatter(subset, size)
        return subset_labels


    def _hide_empty_subsets(self, subset_sizes : Mapping[Tuple[bool], Union[int, float]]) -> None:
        """If the layout routine assigned a non-zero area to a zero-size subset, hide it."""
        for subset, size in subset_sizes.items():
            if (size == 0) & (self.subset_geometries[subset].area > 0):
                self.subset_artists[subset].set_visible(False)
                self.subset_label_artists[subset].set_visible(False)


    @classmethod
    def from_sets(cls, sets, *args, **kwargs):
        """Instantiate the set diagram from a list of sets, rather than subset sizes.

        Parameters
        ----------
        sets : list[set]
            The sets.
        subset_labels : Optional[Mapping[Tuple[bool], str]]
            A dictionary mapping each subset to its desired label or None. If None,
            the subset_label_formatter is used create subset labels based on the subset sizes.
        subset_label_formatter : Callable[[Tuple[bool], int | float], str]
            The formatter used to create subset labels based on the subset sizes.
            The argument is ignored if subset_labels are not None.
        set_labels : Optional[list[str]]
            A list of set labels.
            If none, defaults to the letters of the alphabet (capitalized).
        set_colors : Optional[list[ColorType]]
            A corresponding list of matplotlib colors.
            If none, defaults to the default matplotlib color cycle.
        cost_function_objective : str
            The cost function objective; one of:

            - 'simple'      : :code:`|x - y|`
            - 'squared'     : :code:`(x - y)^2`
            - 'logarithmic' : :code:`|log(x + 1) - log(y + 1)|`
            - 'relative'    : :code:`1 - min(x/y, y/x)`
            - 'inverse'     : :code:`|1 / (x + epsilon) - 1 / (y + epsilon)|`

            Only applicable when instantiating an :code:`EulerDiagram`.
        verbose : bool
            Print a report of the optimisation process.
            Only applicable when instantiating an :code:`EulerDiagram`.
        ax : Optional[plt.Axes]
            The matplotlib axis instance to draw onto.
            If none provided, a new figure with a single axis is instantiated.

        Attributes
        ----------
        origins : NDArray
            The circle origins.
        radii : NDArray
            The circle radii.
        subset_geometries : dict[Tuple[bool], shapely.geometry.polygon.Polygon]
            The dictionary mapping each subset to its shapely geometry.
        subset_artists : dict[tuple[bool], plt.Polygon]
            The matplotlib Polygon patches representing each subset.
        subset_label_artists : dict[tuple[bool], plt.Text]
            The matplotlib text objects used to label each subset.
        set_label_artists : list[plt.Text]
            The matplotlib text objects used to label each set.
        ax : plt.Axes
            The matplotlib axis instance.

        """

        subsets = get_subsets(sets)
        subset_sizes = {subset_id : len(subset) for subset_id, subset in subsets.items()}
        class_instance = cls(subset_sizes, *args, **kwargs)
        return class_instance


    @classmethod
    def as_wordcloud(cls, sets, minimum_resolution=300, wordcloud_kwargs=dict(), *args, **kwargs):
        """Generate a set diagram with word clouds displaying the subset items.

        Parameters
        ----------
        sets : list[set]
            The sets.
        minimum_resolution : int
            The minimum extent of the wordcloud image in pixels (i.e. :code:`min(width, height)`).
            Larger images take significantly longer to generate.
        wordcloud_kwargs : dict[str, Any]
            Key word arguments passed through to wordcloud.WordCloud.
            Consult the wordcloud documentation [1]_ for a complete list.
            However, the following arguments are reserved:

            - :code:`mode = 'RGBA'`
            - :code:`background = None`
            - :code:`color_func = lambda *args, **kwargs : subset_color`

        subset_labels : Optional[Mapping[Tuple[bool], str]]
            A dictionary mapping each subset to its desired label or None. If None,
            the subset_label_formatter is used create subset labels based on the subset sizes.
        subset_label_formatter : Callable[[Tuple[bool], int | float], str]
            The formatter used to create subset labels based on the subset sizes.
            The argument is ignored if subset_labels are not None.
        set_labels : Optional[list[str]]
            A list of set labels.
            If none, defaults to the letters of the alphabet (capitalized).
        set_colors : Optional[list[ColorType]]
            A corresponding list of matplotlib colors.
            If none, defaults to the default matplotlib color cycle.
        cost_function_objective : str
            The cost function objective; one of:

            - 'simple'      : :code:`|x - y|`
            - 'squared'     : :code:`(x - y)^2`
            - 'logarithmic' : :code:`|log(x + 1) - log(y + 1)|`
            - 'relative'    : :code:`1 - min(x/y, y/x)`
            - 'inverse'     : :code:`|1 / (x + epsilon) - 1 / (y + epsilon)|`

            Only applicable when instantiating an :code:`EulerDiagram`.
        verbose : bool
            Print a report of the optimisation process.
            Only applicable when instantiating an :code:`EulerDiagram`.
        ax : Optional[plt.Axes]
            The matplotlib axis instance to draw onto.
            If none provided, a new figure with a single axis is instantiated.

        Attributes
        ----------
        origins : NDArray
            The circle origins.
        radii : NDArray
            The circle radii.
        subset_geometries : dict[Tuple[bool], shapely.geometry.polygon.Polygon]
            The dictionary mapping each subset to its shapely geometry.
        subset_artists : dict[tuple[bool], plt.Polygon]
            The matplotlib Polygon patches representing each subset.
        subset_label_artists : dict[tuple[bool], plt.Text]
            The matplotlib text objects used to label each subset.
        set_label_artists : list[plt.Text]
            The matplotlib text objects used to label each set.
        ax : plt.Axes
            The matplotlib axis instance.
        wordcloud : matplotlib.image.AxesImage
            The WordCloud image.

        References
        ----------
        .. [1] https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html

        """
        subsets = get_subsets(sets)
        subset_sizes = {subset_id : len(subset) for subset_id, subset in subsets.items()}
        class_instance = cls(subset_sizes, *args, **kwargs)
        class_instance._make_subsets_transparent()
        class_instance.wordcloud = class_instance._generate_wordcloud(
            subsets,
            subset_geometries  = class_instance.subset_geometries,
            minimum_resolution = minimum_resolution,
            wordcloud_kwargs   = wordcloud_kwargs,
            ax                 = class_instance.ax,
        )
        return class_instance


    def _make_subsets_transparent(self) -> None:
        """Make subset faces and subset labels transparent, as they
        would overlap with the word cloud text otherwise.
        """
        # We don't use artist.set_alpha(0), as this would also make the artist
        # edge transparent as well.
        for subset, artist in self.subset_artists.items():
            r, g, b, a = to_rgba(artist.get_facecolor()) # type: ignore
            artist.set_facecolor((r, g, b, 0.))

        for subset, label in self.subset_label_artists.items():
            label.set_visible(False)


    def _generate_wordcloud(
            self,
            subsets            : dict[Tuple[bool], set[str]],
            subset_geometries  : dict[Tuple[bool], ShapelyPolygon],
            minimum_resolution : int,
            wordcloud_kwargs   : dict[str, Any],
            ax                 : plt.Axes,
    ) -> AxesImage:

        subset_masks = self._get_subset_masks(
            subset_geometries, minimum_resolution, ax)

        subset_images = [
            self._generate_subset_wordcloud(
                subset           = subsets[subset_id],
                mask             = mask,
                rgba             = self.subset_colors[subset_id],
                wordcloud_kwargs = wordcloud_kwargs,
            ) for subset_id, mask in subset_masks.items()
        ]

        combined_image = np.sum(subset_images, axis=0)

        return self.ax.imshow(combined_image / 255, interpolation="bilinear", extent=ax.axis())


    def _get_subset_masks(self, subset_geometries, minimum_resolution, ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = xmax - xmin
        dy = ymax - ymin

        if dx < dy:
            width_in_pixel = minimum_resolution
            height_in_pixel = int(dy / dx * minimum_resolution)
        else:
            width_in_pixel = int(dx / dy * minimum_resolution)
            height_in_pixel = minimum_resolution

        X, Y = np.meshgrid(np.linspace(xmin, xmax, width_in_pixel),
                           np.linspace(ymin, ymax, height_in_pixel))
        XY = np.c_[X.ravel(), Y.ravel()]

        subset_masks = dict()
        for subset_id, geometry in subset_geometries.items():
            if geometry.area > 0:
                path = Path(geometry.exterior.coords)
                mask = path.contains_points(XY).reshape((height_in_pixel, width_in_pixel))
                mask = np.flipud(mask) # image origin is in the upper left
                subset_masks[subset_id] = mask
        return subset_masks


    def _generate_subset_wordcloud(
            self,
            subset           : set[str],
            mask             : NDArray,
            rgba             : NDArray,
            wordcloud_kwargs : dict[str, Any],
    ) -> NDArray:

        mask = 255 * np.invert(mask).astype(np.uint8) # black is filled by WordCloud
        rgba_as_tuple = tuple(int(255 * channel) for channel in rgba)

        wc = WordCloud(
            mask             = mask,
            mode             = "RGBA",
            background_color = None,
            color_func       = lambda *args, **kwargs : rgba_as_tuple,
            **wordcloud_kwargs
        )

        return wc.generate_from_frequencies(Counter(subset)).to_array()


class VennDiagram(EulerDiagram):
    """Create an area-equal Venn diagram visualising the relationships
    between two or more sets.

    Sets are represented through overlapping circles. The size of a
    subset is indicated by the label of the corresponding patch; the
    size of the patch, however, is not indicative of the size of the
    subset, such that even zero-size subsets can be represented.

    Parameters
    ----------
    subset_sizes : Mapping[Tuple[bool], int | float]
        The dictionary mapping each subset to its size.
        Subsets are represented by tuples of booleans using the inclusion/exclusion nomenclature, i.e.
        each entry in the tuple indicates if the corresponding set is a superset of the subset.
        For example, given the sets A, B, C, the subset (1, 1, 1) corresponds to the intersection of all three sets,
        whereas (1, 1, 0) is the subset formed by the difference between the intersection of A with B, and C.
    subset_labels : Optional[Mapping[Tuple[bool], str]]
        A dictionary mapping each subset to its desired label.
        If None, the subset_label_formatter is used create subset labels based on the subset sizes.
    subset_label_formatter : Callable[[Tuple[bool], int | float], str]
        The formatter used to create subset labels based on the subset sizes.
        The argument is ignored if subset_labels are not None.
    set_labels : Optional[list[str]]
        A list of set labels.
        If none, defaults to the letters of the alphabet (capitalized).
    set_colors : Optional[list[ColorType]]
        A corresponding list of matplotlib colors.
        If none, defaults to the default matplotlib color cycle.
    ax : Optional[plt.Axes]
        The matplotlib axis instance to draw onto.
        If none provided, a new figure with a single axis is instantiated.

    Attributes
    ----------
    subset_areas : Mapping[Tuple[bool], int | float]
        The dictionary mapping each subset to a desired area size.
    origins : NDArray
        The circle origins.
    radii : NDArray
        The circle radii.
    subset_geometries : dict[Tuple[bool], shapely.geometry.polygon.Polygon]
        The dictionary mapping each subset to its shapely geometry.
    subset_artists : dict[tuple[bool], plt.Polygon]
        The matplotlib Polygon patches representing each subset.
    subset_label_artists : dict[tuple[bool], plt.Text]
        The matplotlib text objects used to label each subset.
    set_label_artists : list[plt.Text]
        The matplotlib text objects used to label each set.
    ax : plt.Axes
        The matplotlib axis instance.

    """

    def __init__(
            self,
            subset_sizes            : Mapping[Tuple[bool], Union[int, float]],
            subset_labels           : Optional[Mapping[Tuple[bool], str]]       = None,
            subset_label_formatter  : Callable[[Tuple[bool], Union[int, float]], str] = lambda subset, size : str(size),
            set_labels              : Optional[list[str]]                       = None,
            set_colors              : Optional[list[ColorType]]                 = None,
            ax                      : Optional[plt.Axes]                        = None,
    ) -> None:

        if subset_labels is None:
            subset_labels = self._get_subset_labels(
                subset_sizes, subset_label_formatter)

        # Specify area of subset patches independently of actual subset size.
        self.subset_areas = self._get_subset_areas(list(subset_sizes.keys()))
        super().__init__(
            self.subset_areas,
            subset_labels           = subset_labels,
            set_labels              = set_labels,
            set_colors              = set_colors,
            cost_function_objective = "simple",
            verbose                 = False,
            ax                      = ax,
        )


    def _get_subset_areas(self, subsets : list[Tuple[bool]]) -> dict[Tuple[bool], float]:
        """Creates a dictionary mapping subsets to area sizes. The
        values are independent of subset size."""
        subset_size = dict()
        for subset_id in subsets:
            # # Option 1: all subsets are equal size
            # subset_size[subset_id] = 1
            # Option 2: intersections half in size with each superset
            subset_size[subset_id] = 1 / 2**(np.sum(subset_id) - 1)
        return subset_size
