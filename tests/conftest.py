import numpy as np
from polytope_feature.datacube.datacube_axis import IntDatacubeAxis
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

# -- Shared constants for reforecast tests --

REFORECAST_METADATA_BASE = {
    "class": "ce",
    "date": np.datetime64("2024-03-01"),
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


def forecast_tree(points, param="167", step=(0,), date=np.datetime64("2025-01-01T00:00:00")):
    """Build a standard forecast TensorIndexTree with the given spatial points.

    Args:
        points: list of (lat, lon, result_list) tuples, passed to make_point().
        param: MARS parameter code.
        step: tuple of step values.
        date: forecast date.
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
        parent.add_child(make_point(lat, lon, result))
    return tree


def reforecast_branch(hdate, points, param="167", step=(0,)):
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
        parent.add_child(make_point(lat, lon, result))
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
