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
            "MergedTensorIndexNode is not available in this polytope build; merged-node tests should be skipped."
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


def reforecast_separate_datetime_tree(
    points,
    hdates,
    times,
    date=np.datetime64("2024-03-01"),
    step=(0,),
    param="167",
    point_factory=make_point,
):
    """Build a separate-datetime reforecast tree (class=ce).

    ``date``, ``hdate`` and ``time`` are all present as independent axes, mirroring
    the EFAS layout where the reforecast reference datetime is ``hdate + time``.

    Args:
        points: list of (lat, lon, result_list) tuples. Each ``result_list`` must
            be laid out as the flattened product over (hdate, step, time) in that
            order (hdate-major), matching the compressed leaf ordering.
        hdates: tuple of ``np.datetime64`` hdate values.
        times: tuple of ``np.timedelta64``/``timedelta`` time-of-day offsets.
        step: tuple of step values.
    """
    branch = chain(
        node("hdate", hdates),
        node("domain", ("g",)),
        node("expver", ("4321",)),
        node("levtype", ("sfc",)),
        node("param", (param,)),
        node("step", step),
        node("stream", ("efcl",)),
        node("time", times),
        node("type", ("sfo",)),
    )
    parent = tip(branch)
    for lat, lon, result in points:
        parent.add_child(point_factory(lat, lon, result))

    tree = chain(
        TensorIndexTree(),
        node("class", ("ce",)),
        node("date", (date,)),
    )
    tip(tree).add_child(branch)
    return tree


def forecast_separate_datetime_tree(
    points,
    times,
    date=np.datetime64("2026-05-01"),
    step=(6, 24),
    param="240023",
    point_factory=make_point,
):
    """Build a separate-datetime *forecast* tree (class=ce, stream=efas).

    Unlike :func:`reforecast_separate_datetime_tree` there is **no** ``hdate``
    axis: ``date`` + ``time`` is the forecast *run* reference and ``step`` is the
    lead time. Each ``(date, time)`` run should become its own coverage with the
    steps forming its ``t``-axis.

    Args:
        points: list of (lat, lon, result_list) tuples. Each ``result_list`` must
            be laid out as the flattened product over (date, step, time) in that
            order, matching the compressed leaf ordering (date is single-valued
            here, so effectively (step, time)).
        times: tuple of ``np.timedelta64``/``timedelta`` time-of-day offsets.
        step: tuple of step values.
    """
    branch = chain(
        node("date", (date,)),
        node("domain", ("g",)),
        node("expver", ("8888",)),
        node("levtype", ("sfc",)),
        node("model", ("lisflood",)),
        node("origin", ("ecmf",)),
        node("param", (param,)),
        node("step", step),
        node("stream", ("efas",)),
        node("time", times),
        node("type", ("cf",)),
    )
    parent = tip(branch)
    for lat, lon, result in points:
        parent.add_child(point_factory(lat, lon, result))

    tree = chain(
        TensorIndexTree(),
        node("class", ("ce",)),
    )
    tip(tree).add_child(branch)
    return tree


def reforecast_separate_datetime_vertical_tree(
    points,
    hdates,
    times,
    levels,
    date=np.datetime64("2024-03-01"),
    step=(0,),
    param="130",
    point_factory=make_point,
):
    """Build a separate-datetime reforecast tree (class=ce) with a ``levelist`` axis.

    Mirrors :func:`reforecast_separate_datetime_tree` but inserts a multi-valued
    ``levelist`` axis, for exercising the VerticalProfile reforecast path.

    Each ``result_list`` must be laid out as the flattened product over
    (hdate, levelist, step, time) in that order, matching the compressed leaf
    ordering (top-down axis order in the tree).
    """
    branch = chain(
        node("hdate", hdates),
        node("domain", ("g",)),
        node("expver", ("4321",)),
        node("levtype", ("pl",)),
        node("param", (param,)),
        node("levelist", levels),
        node("step", step),
        node("stream", ("efcl",)),
        node("time", times),
        node("type", ("sfo",)),
    )
    parent = tip(branch)
    for lat, lon, result in points:
        parent.add_child(point_factory(lat, lon, result))

    tree = chain(
        TensorIndexTree(),
        node("class", ("ce",)),
        node("date", (date,)),
    )
    tip(tree).add_child(branch)
    return tree
