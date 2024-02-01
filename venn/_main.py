import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint

from shapely.geometry import Point
from shapely.ops import polylabel

DEBUG = True


def get_radii(a, b, c, ab, bc, ac, abc):
    areas = np.array([
        a + ab + ac + abc,
        b + ab + bc + abc,
        c + ac + bc + abc,
    ])
    return np.sqrt(areas / np.pi)


def get_subset_areas(origins, radii):
    geometries = get_subset_geometries(origins, radii)
    return np.array([geometry.area for geometry in geometries])


def get_subset_geometries(origins, radii):
    a, b, c = [get_shapely_circle(origin, radius) for origin, radius in zip(origins, radii)]
    return [
        a.difference(b).difference(c), # A
        b.difference(a).difference(c), # B
        c.difference(a).difference(b), # C
        a.intersection(b).difference(c), # AB
        b.intersection(c).difference(a), # BC
        a.intersection(c).difference(b), # AC
        a.intersection(b).intersection(c), # ABC
    ]


def get_shapely_circle(origin, radius):
    return Point(*origin).buffer(radius)


def initialize_origins(radii):
    """Initialize origins on a small circle around (0, 0)."""
    origin = np.zeros((2))
    radius = np.min(radii)
    angles = np.pi * np.array([0, 2/3, 4/3])
    return np.array(
        [get_point_on_a_circle(origin, radius, angle) for angle in angles]
    )


def get_point_on_a_circle(origin, radius, angle):
    """Compute the (x, y) coordinate of a point at a specified angle
    on a circle given by its (x, y) origin and radius."""
    x0, y0 = origin
    x = x0 + radius * np.cos(angle)
    y = y0 + radius * np.sin(angle)
    return np.array([x, y])


def solve_venn3_circles(desired_areas):
    # [A, B, C, AB, BC, AC, ABC]
    desired_areas = np.array(desired_areas)

    radii = get_radii(*desired_areas)

    def cost_function(origins):
        current_areas = get_subset_areas(origins.reshape(-1, 2), radii)

        # # Option 1: absolute difference
        # # Probably not the best choice, as small areas are often practically ignored.
        # cost = np.abs(current_areas - desired_areas)

        # # Option 2: relative difference
        # # This often results in the optimization routine failing.
        # minimum_area = 1e-2
        # desired_areas[desired_areas < minimum_area] = minimum_area
        # cost = np.abs(current_areas - desired_areas) / desired_areas

        # Option 3: absolute difference of log(area + 1)
        # This transformation is monotonic increasing but strongly compresses large numbers,
        # thus allowing small numbers to carry more weight in the optimization.
        cost = np.abs(np.log(current_areas + 1) - np.log(desired_areas + 1))

        return np.sum(cost)

    # constraints:
    eps = np.min(radii) * 0.01
    lower_bounds = np.abs(radii[np.newaxis, :] - radii[:, np.newaxis]) - eps
    lower_bounds[lower_bounds < 0] = 0
    lower_bounds = squareform(lower_bounds)

    upper_bounds = radii[np.newaxis, :] + radii[:, np.newaxis] + eps
    upper_bounds -= np.diag(np.diag(upper_bounds)) # squareform requires zeros on diagonal
    upper_bounds = squareform(upper_bounds)

    def constraint_function(origins):
        origins = np.reshape(origins, (-1, 2))
        return pdist(origins)

    distance_between_origins = NonlinearConstraint(
        constraint_function, lb=lower_bounds, ub=upper_bounds)

    result = minimize(
        cost_function,
        initialize_origins(radii).flatten(),
        method='SLSQP',# 'COBYLA',
        constraints=[distance_between_origins],
        options=dict(disp=DEBUG)
    )

    if not result.success:
        print("Warning: could not compute circle positions for given subsets.")
        print(f"scipy.optimize.minimize: {result.message}.")

    origins = result.x.reshape((-1, 2))

    if DEBUG:
        import pandas as pd
        data = pd.DataFrame(dict(desired=desired_areas, actual=get_subset_areas(origins, radii)))
        data["absolute difference"] = np.abs(data["desired"] - data["actual"])
        data["relative difference"] = data["absolute difference"] / data["desired"]
        with pd.option_context('display.float_format', '{:0.1f}'.format):
            print(data)

    return origins, radii


def get_label_positions(origins, radii, labels):
    "For each non-zero subset, find the point of inaccesibility."
    geometries = get_subset_geometries(origins, radii)
    output = list()
    for ii, (label, geometry) in enumerate(zip(labels, geometries)):
        if geometry.area > 0:
            poi = polylabel(geometry)
            output.append((poi.x, poi.y, label))
    return output


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def test(A, B, C, AB, BC, AC, ABC, ax=None):
        origins, radii = solve_venn3_circles([A, B, C, AB, BC, AC, ABC])

        if ax is None:
            fig, ax = plt.subplots()

        for origin, radius in zip(origins, radii):
            ax.add_patch(plt.Circle(origin, radius, alpha=0.33))
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")
        # ax.set_title(f"A={A}, B={B}, C={C},\nAB={AB}, BC={BC}, AC={AC}, ABC={ABC}")

        label_positions = get_label_positions(origins, radii, [A, B, C, AB, BC, AC, ABC])
        for (x, y, label) in label_positions:
            ax.text(x, y, label, va="center", ha="center")

    # fig, axes = plt.subplots(2, 3, figsize=(15,10))
    # axes = axes.ravel()
    # test(1, 1, 1, 0.5, 0.5, 0.5, 0.25, axes[0]) # canonical example
    # test(3, 2, 1, 0.5, 0.4, 0.3, 0.2, axes[1]) # different areas
    # test(1, 1, 1, 0, 0, 0, 0, axes[2]) # no intersections; not bad, but could be handled better
    # test(1, 0, 0, 1, 0, 0, 1, axes[3]) # a is superset to b, b is superset to c
    # test(2, 2, 2, 0.5, 0, 0.5, 0, axes[4]) # no BC, and no ABC
    # test(2, 2, 2, 0.1, 0.1, 0.1, 0, axes[5]) # no ABC

    # # issue #30
    # test(1, 0, 0, 32, 0, 76, 13)
    # test(1, 0, 0, 650, 0, 76, 13)
    # test(7549417, 15685620, 26018311, 5128906, 301048, 6841264, 2762301) # works well with latest version

    # # issue #34
    # test(167, 7, 25, 41, 174, 171, 51)

    # issue # 50
    test(1808, 1181, 1858, 4715, 3031, 26482, 65012)
    plt.show()
