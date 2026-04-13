import numpy as np
from polytope_feature.datacube.datacube_axis import IntDatacubeAxis
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree


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
    """Walk to the deepest single-child descendant."""
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
