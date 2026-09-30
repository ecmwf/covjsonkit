import numpy as np
from polytope_feature.datacube.datacube_axis import IntDatacubeAxis
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

try:
    # Only available on polytope versions that support compacted unstructured
    # (ICON, Lambert LAM) results. Absent on released polytope, in which case
    # the merged-node fixtures/tests below are skipped rather than breaking
    # collection of the whole test suite.
    from polytope_feature.datacube.tensor_index_tree import MergedTensorIndexNode
except ImportError:
    MergedTensorIndexNode = None

# -- Shared constants for reforecast tests --

REFORECAST_METADATA_BASE = {
    "class": "ce",
    "date": "2024-03-01",
    "domain": "g",
    "expver": "4321",
    "levtype": "sfc",
    "step": 0,
    "stream": "efcl",
    "type": "sfo",
    "number": 0,
}

COMPOSITE_TWO_POINTS_XYZ = {
    "dataType": "tuple",
    "coordinates": ["x", "y", "z"],
    "values": [[48.0, 11.0, 0], [50.0, 12.0, 0]],
}


# -- Tree-building helpers --


def node(name, values):
    """Create a TensorIndexTree node with the given axis name and values."""
    ax = IntDatacubeAxis()
    ax.name = name
    return TensorIndexTree(axis=ax, values=tuple(values))


def chain(*nodes):
    """Link nodes sequentially via add_child(), return the root."""
    for a, b in zip(nodes, nodes[1:]):
        a.add_child(b)
    return nodes[0]


def tip(tree):
    """Walk to the deepest single-child descendant (the 'tip' of a linear chain)."""
    while tree.children:
        tree = tree.children[0]
    return tree


def make_leaf(lon, result):
    """Create a longitude leaf node with result data."""
    leaf = node("longitude", (lon,))
    leaf.result = [np.float64(r) for r in result]
    return leaf


def make_point(lat, lon, result):
    """Create a latitude->longitude(leaf) subtree for a single spatial point."""
    lat_n = node("latitude", (lat,))
    lat_n.add_child(make_leaf(lon, result))
    return lat_n


def make_merged_point(lat, lon, result):
    """Create a compacted MergedTensorIndexNode for a single spatial point.

    This is the unstructured-grid equivalent of :func:`make_point`: instead of a
    latitude node with a longitude-leaf child, the (lat, lon) pair is compacted
    into a single leaf node carrying its own ``result``.
    """
    if MergedTensorIndexNode is None:
        raise RuntimeError(
            "MergedTensorIndexNode is not available in this polytope build; " "merged-node tests should be skipped."
        )
    lat_ax = IntDatacubeAxis()
    lat_ax.name = "latitude"
    lon_ax = IntDatacubeAxis()
    lon_ax.name = "longitude"
    merged = MergedTensorIndexNode(axes=(lat_ax, lon_ax), values=(lat, lon))
    merged.result = [np.float64(r) for r in result]
    return merged


def forecast_tree(points, param="167", step=(0,), date=np.datetime64("2025-01-01T00:00:00"), point_factory=make_point):
    """Build a standard forecast TensorIndexTree with the given spatial points.

    Args:
        points: list of (lat, lon, result_list) tuples, passed to point_factory().
        param: MARS parameter code.
        step: tuple of step values.
        date: forecast date.
        point_factory: callable(lat, lon, result) -> node. Use make_merged_point
            to build a compacted unstructured (MergedTensorIndexNode) tree.
    """
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (date,)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("sfc",)),
        node("param", (param,)),
        node("step", step),
        node("stream", ("oper",)),
        node("type", ("fc",)),
    )
    parent = tip(tree)
    for lat, lon, result in points:
        parent.add_child(point_factory(lat, lon, result))
    return tree


def month_tree(points, param="167", years=(2020, 2021), months=(1,), point_factory=make_point):
    """Build a monthly-mean tree (year/month axes) for the walk_tree_month path.

    Args:
        points: list of (lat, lon, result_list) tuples, passed to point_factory().
        param: MARS parameter code.
        years: tuple of year values.
        months: tuple of month values.
        point_factory: callable(lat, lon, result) -> node. Use make_merged_point
            to build a compacted unstructured (MergedTensorIndexNode) tree.
    """
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("levtype", ("sfc",)),
        node("param", (param,)),
        node("year", years),
        node("month", months),
    )
    parent = tip(tree)
    for lat, lon, result in points:
        parent.add_child(point_factory(lat, lon, result))
    return tree


def reforecast_branch(hdate, points, param="167", step=(0,), point_factory=make_point):
    """Build a reforecast branch rooted at an hdate node.

    Attaches spatial points at the leaf. Caller is responsible for
    grafting this onto a root tree via ``tip(root).add_child(branch)``.
    """
    branch = chain(
        node("hdate", (hdate,)),
        node("domain", ("g",)),
        node("expver", ("4321",)),
        node("levtype", ("sfc",)),
        node("param", (param,)),
        node("step", step),
        node("stream", ("efcl",)),
        node("type", ("sfo",)),
    )
    parent = tip(branch)
    for lat, lon, result in points:
        parent.add_child(point_factory(lat, lon, result))
    return branch


def reforecast_tree(branches, date=np.datetime64("2024-03-01")):
    """Build a reforecast tree with class=ce root, attaching pre-built hdate branches."""
    tree = chain(
        TensorIndexTree(),
        node("class", ("ce",)),
        node("date", (date,)),
    )
    root = tip(tree)
    for b in branches:
        root.add_child(b)
    return tree


try:
    # Only available on polytope versions that return array-backed bulk lat/lon leaves.
    from polytope_feature.datacube.tensor_index_tree import (
        BulkGridTensorIndexNode,
        BulkMergedTensorIndexNode,
    )
except ImportError:
    BulkGridTensorIndexNode = None
    BulkMergedTensorIndexNode = None


def _latlon_axes():
    lat_ax = IntDatacubeAxis()
    lat_ax.name = "latitude"
    lon_ax = IntDatacubeAxis()
    lon_ax.name = "longitude"
    return lat_ax, lon_ax


def _set_bulk_result(bulk, point_results):
    """Store per-point result lists as one array of all points per combination, as polytope does."""
    n_combos = len(point_results[0])
    bulk.result = [np.array([res[c] for res in point_results], dtype=np.float64) for c in range(n_combos)]


def bulkify(tree):
    """Replace, under every node, the compacted ``MergedTensorIndexNode`` children by one bulk leaf.

    Turns a tree built with :func:`make_merged_point` into the layout polytope returns
    for unstructured grids.
    """
    children = list(tree.children)
    if children and all(isinstance(c, MergedTensorIndexNode) for c in children):
        coordinates = [c.values for c in children]
        bulk = BulkMergedTensorIndexNode(_latlon_axes(), coordinates, list(range(len(children))))
        _set_bulk_result(bulk, [list(c.result) for c in children])
        for c in children:
            tree.children.remove(c)
        tree.add_child(bulk)
        return tree
    for c in children:
        bulkify(c)
    return tree


def gridify(tree):
    """Replace, under every node, the ``latitude -> longitude`` leaf layers by one bulk grid leaf.

    Turns a legacy tree (eg. built with :func:`make_point` or :func:`make_row`) into the
    layout polytope returns for structured grids with ``bulk_grid_leaves``.
    """
    children = list(tree.children)
    if children and all(getattr(c, "axis", None) is not None and c.axis.name == "latitude" for c in children):
        lat_values, lon_rows, point_results = [], [], []
        for lat_node in children:
            row = []
            for lon_leaf in lat_node.children:
                n = len(lon_leaf.values)
                n_combos = len(lon_leaf.result) // n
                for j, lon in enumerate(lon_leaf.values):
                    row.append(lon)
                    point_results.append([lon_leaf.result[c * n + j] for c in range(n_combos)])
            lat_values.append(lat_node.values[0])
            lon_rows.append(row)
        grid = BulkGridTensorIndexNode(_latlon_axes(), lat_values, lon_rows)
        _set_bulk_result(grid, point_results)
        for c in children:
            tree.children.remove(c)
        tree.add_child(grid)
        return tree
    for c in children:
        gridify(c)
    return tree


def make_row(lat, lons, point_results):
    """Create a legacy latitude->longitude(leaf) subtree holding several compressed longitudes.

    ``point_results[j]`` holds the per-combination values of longitude ``j``; the leaf's
    result is laid out combination-major, as polytope assigns it.
    """
    lat_n = node("latitude", (lat,))
    leaf = node("longitude", tuple(lons))
    n_combos = len(point_results[0])
    leaf.result = [np.float64(point_results[j][c]) for c in range(n_combos) for j in range(len(lons))]
    lat_n.add_child(leaf)
    return lat_n
