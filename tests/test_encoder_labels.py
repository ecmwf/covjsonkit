"""Tests for point labels carried as polytope tags.

polytope-mars tags every requested point as ``(index, label)``: ``index`` is its
position in the request and ``label`` the user's label, or ``None``. Point-like
encoders emit one coverage per tag, in request order, so two requested points
that snap to the same grid point are both returned. ``mars:metadata.label`` is set
only when a label was requested. Trees without tags give the same output as before.
"""

import logging

import numpy as np
import pytest
from conftest import (
    MergedTensorIndexNode,
    chain,
    forecast_tree,
    make_leaf,
    make_merged_point,
    make_point,
    month_tree,
    node,
    reforecast_separate_datetime_tree,
    reforecast_separate_datetime_vertical_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit
from covjsonkit.encoder.encoder import expand_points_by_tags, node_tags, parse_tag


def tagged(factory, tags_by_point):
    """Wrap a conftest point factory so each point's leaf carries the given tags.

    The latitude node also gets the tags, as polytope stamps them on both.
    """

    def build(lat, lon, result):
        point = factory(lat, lon, result)
        tags = set(tags_by_point.get((lat, lon), ()))
        point.tags.update(tags)
        for child in point.children:
            child.tags.update(tags)
        return point

    return build


def encode(kind, tree, method="from_polytope"):
    return getattr(Covjsonkit().encode("CoverageCollection", kind), method)(tree)


def labels(covjson):
    return [c["mars:metadata"].get("label") for c in covjson["coverages"]]


def points_of(covjson):
    return [
        (c["domain"]["axes"]["latitude"]["values"][0], c["domain"]["axes"]["longitude"]["values"][0])
        for c in covjson["coverages"]
    ]


def values_of(covjson):
    return [list(c["ranges"].values())[0]["values"] for c in covjson["coverages"]]


# Two grid points; results are per step (0, 6)
P1 = (48.0, 11.0, [1.0, 2.0])
P2 = (50.0, 12.0, [3.0, 4.0])


# -- Helpers --


class TestTagHelpers:
    def test_parse_tag(self):
        assert parse_tag((0, "Lisbon")) == (0, "Lisbon")
        assert parse_tag((3, None)) == (3, None)
        # Anything that is not (int, label) is a bare label without an index
        assert parse_tag("Lisbon") == (None, "Lisbon")
        assert parse_tag((True, "x")) == (None, (True, "x"))

    def test_node_tags_sorted_by_index(self):
        leaf = make_leaf(11.0, [1.0])
        leaf.tags.update({(2, "C"), (0, "A")})
        assert node_tags(leaf) == [(0, "A"), (2, "C")]
        assert node_tags(make_leaf(11.0, [1.0])) == []

    def test_expand_orders_by_index_and_duplicates(self):
        tags = [[(2, "C")], [(0, "A"), (1, "B")]]
        assert expand_points_by_tags(tags, 2) == [(1, "A"), (1, "B"), (0, "C")]

    def test_expand_without_tags_is_tree_order(self):
        assert expand_points_by_tags(None, 3) == [(0, None), (1, None), (2, None)]
        assert expand_points_by_tags([[], [], []], 3) == [(0, None), (1, None), (2, None)]

    def test_expand_untagged_after_tagged(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert expand_points_by_tags([[], [(0, "A")]], 2) == [(1, "A"), (0, None)]
        assert "carry no tag" in caplog.text

    def test_expand_mismatched_length_ignores_tags(self):
        assert expand_points_by_tags([[(0, "A")]], 2) == [(0, None), (1, None)]


# -- TimeSeries (PointSeries) --


class TestTimeSeriesLabels:
    def test_labels_in_request_order(self):
        # Tree order is P1, P2 but the request listed P2 first
        tree = forecast_tree(
            [P1, P2],
            step=(0, 6),
            point_factory=tagged(make_point, {P1[:2]: [(1, "Munich")], P2[:2]: [(0, "Bonn")]}),
        )
        covjson = encode("pointseries", tree)
        assert labels(covjson) == ["Bonn", "Munich"]
        assert points_of(covjson) == [(50.0, 12.0), (48.0, 11.0)]
        assert values_of(covjson) == [[3.0, 4.0], [1.0, 2.0]]

    def test_merged_points_duplicated(self):
        # Two requested points snapped to the same grid point
        tree = forecast_tree(
            [P1], step=(0, 6), point_factory=tagged(make_point, {P1[:2]: [(0, "StationA"), (1, "StationB")]})
        )
        covjson = encode("pointseries", tree)
        assert labels(covjson) == ["StationA", "StationB"]
        a, b = covjson["coverages"]
        assert a["domain"] == b["domain"]
        assert a["ranges"] == b["ranges"]

    def test_unlabelled_same_output_as_labelled(self):
        def build(label_a, label_b):
            return forecast_tree(
                [P1, P2],
                step=(0, 6),
                point_factory=tagged(make_point, {P1[:2]: [(0, label_a), (1, label_b)], P2[:2]: [(2, None)]}),
            )

        labelled = encode("pointseries", build("A", "B"))
        unlabelled = encode("pointseries", build(None, None))
        assert labels(labelled) == ["A", "B", None]
        assert labels(unlabelled) == [None, None, None]
        assert all("label" not in c["mars:metadata"] for c in unlabelled["coverages"])
        for a, b in zip(labelled["coverages"], unlabelled["coverages"]):
            a_meta = {k: v for k, v in a["mars:metadata"].items() if k != "label"}
            assert a_meta == b["mars:metadata"]
            assert a["domain"] == b["domain"]
            assert a["ranges"] == b["ranges"]

    def test_untagged_tree_unchanged(self):
        plain = encode("pointseries", forecast_tree([P1, P2], step=(0, 6)))
        assert labels(plain) == [None, None]
        assert points_of(plain) == [(48.0, 11.0), (50.0, 12.0)]

    def test_tags_taken_from_leaf_not_latitude_row(self):
        # Two grid points in the same latitude row: the latitude node carries both
        # tags, each longitude leaf only its own.
        lat = node("latitude", (48.0,))
        lat.tags.update({(0, "West"), (1, "East")})
        west = make_leaf(11.0, [1.0, 2.0])
        west.tags.add((0, "West"))
        east = make_leaf(12.0, [3.0, 4.0])
        east.tags.add((1, "East"))
        lat.add_child(west)
        lat.add_child(east)
        tree = forecast_tree([], step=(0, 6))
        tip(tree).add_child(lat)

        covjson = encode("pointseries", tree)
        assert labels(covjson) == ["West", "East"]
        assert points_of(covjson) == [(48.0, 11.0), (48.0, 12.0)]

    def test_multiple_dates(self):
        dates = (np.datetime64("2025-01-01T00:00:00"), np.datetime64("2025-01-02T00:00:00"))
        tree = chain(TensorIndexTree(), node("class", ("od",)))
        root = tip(tree)
        factory = tagged(make_point, {P1[:2]: [(1, "B")], P2[:2]: [(0, "A")]})
        for date in dates:
            branch = chain(
                node("date", (date,)),
                node("domain", ("g",)),
                node("expver", ("0001",)),
                node("levtype", ("sfc",)),
                node("param", ("167",)),
                node("step", (0, 6)),
                node("stream", ("oper",)),
                node("type", ("fc",)),
            )
            parent = tip(branch)
            parent.add_child(factory(*P1))
            parent.add_child(factory(*P2))
            root.add_child(branch)

        covjson = encode("pointseries", tree)
        # Point-major in request order, then date
        assert labels(covjson) == ["A", "A", "B", "B"]
        assert [c["mars:metadata"]["Forecast date"] for c in covjson["coverages"]] == [
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
        ] * 2

    def test_month(self):
        tree = month_tree(
            [(48.0, 11.0, [1.0, 2.0])],
            years=(2020, 2021),
            point_factory=tagged(make_point, {(48.0, 11.0): [(0, "A"), (1, "B")]}),
        )
        covjson = encode("pointseries", tree, "from_polytope_month")
        assert labels(covjson) == ["A", "B"]
        assert values_of(covjson) == [[1.0, 2.0], [1.0, 2.0]]

    def test_reforecast_separate_datetime(self):
        hdates = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
        times = (np.timedelta64(0, "h"),)
        tree = reforecast_separate_datetime_tree(
            [(48.0, 11.0, [1.0, 2.0]), (50.0, 12.0, [3.0, 4.0])],
            hdates,
            times,
            point_factory=tagged(make_point, {(48.0, 11.0): [(1, "B"), (2, "C")], (50.0, 12.0): [(0, "A")]}),
        )
        covjson = encode("pointseries", tree, "from_polytope_reforecast")
        assert labels(covjson) == ["A", "B", "C"]
        assert values_of(covjson) == [[3.0, 4.0], [1.0, 2.0], [1.0, 2.0]]

    @pytest.mark.skipif(MergedTensorIndexNode is None, reason="polytope without MergedTensorIndexNode")
    def test_merged_node_leaves(self):
        tree = forecast_tree(
            [P1, P2],
            step=(0, 6),
            point_factory=tagged(make_merged_point, {P1[:2]: [(1, "B")], P2[:2]: [(0, "A")]}),
        )
        covjson = encode("pointseries", tree)
        assert labels(covjson) == ["A", "B"]
        assert points_of(covjson) == [(50.0, 12.0), (48.0, 11.0)]


def step_tree(factory):
    """Tree with a separate ``time`` axis, for the ``from_polytope_step`` path."""
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("sfc",)),
        node("param", ("167",)),
        node("step", (0,)),
        node("stream", ("oper",)),
        node("time", (np.timedelta64(0, "h"), np.timedelta64(6, "h"))),
        node("type", ("fc",)),
    )
    parent = tip(tree)
    parent.add_child(factory(*P1))
    parent.add_child(factory(*P2))
    return tree


@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_step_path_labels(kind):
    covjson = encode(
        kind, step_tree(tagged(make_point, {P1[:2]: [(1, "B"), (2, "C")], P2[:2]: [(0, "A")]})), "from_polytope_step"
    )
    assert labels(covjson) == ["A", "B", "C"]
    assert values_of(covjson) == [[3.0, 4.0], [1.0, 2.0], [1.0, 2.0]]

    plain = encode(kind, step_tree(make_point), "from_polytope_step")
    assert labels(plain) == [None, None]
    assert values_of(plain) == [[1.0, 2.0], [3.0, 4.0]]


# -- Position --


class TestPositionLabels:
    def test_labels_and_duplicates(self):
        tree = forecast_tree(
            [P1, P2],
            step=(0, 6),
            point_factory=tagged(make_point, {P1[:2]: [(1, "B"), (2, "C")], P2[:2]: [(0, "A")]}),
        )
        covjson = encode("position", tree)
        assert labels(covjson) == ["A", "B", "C"]
        assert points_of(covjson) == [(50.0, 12.0), (48.0, 11.0), (48.0, 11.0)]
        assert values_of(covjson)[1] == values_of(covjson)[2]

    def test_untagged_unchanged(self):
        covjson = encode("position", forecast_tree([P1, P2], step=(0, 6)))
        assert labels(covjson) == [None, None]
        assert points_of(covjson) == [(48.0, 11.0), (50.0, 12.0)]

    def test_reforecast_separate_datetime(self):
        hdates = (np.datetime64("2025-07-14T00:00:00"),)
        times = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))
        tree = reforecast_separate_datetime_tree(
            [(48.0, 11.0, [1.0, 2.0]), (50.0, 12.0, [3.0, 4.0])],
            hdates,
            times,
            point_factory=tagged(make_point, {(48.0, 11.0): [(1, "B")], (50.0, 12.0): [(0, "A")]}),
        )
        covjson = encode("position", tree, "from_polytope_reforecast")
        # One coverage per (point, reference); request order first
        assert labels(covjson) == ["A", "A", "B", "B"]
        assert all("__tags__" not in c["mars:metadata"] for c in covjson["coverages"])


# -- VerticalProfile --


def vp_tree(points, factory):
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("pl",)),
        node("param", ("130",)),
        node("step", (0,)),
        node("stream", ("oper",)),
        node("type", ("an",)),
        node("levelist", (1000, 850)),
    )
    parent = tip(tree)
    for lat, lon, result in points:
        parent.add_child(factory(lat, lon, result))
    return tree


class TestVerticalProfileLabels:
    def test_labels_and_duplicates(self):
        tree = vp_tree(
            [(48.0, 11.0, [290.0, 280.0]), (50.0, 12.0, [291.0, 281.0])],
            tagged(make_point, {(48.0, 11.0): [(1, "B"), (2, "C")], (50.0, 12.0): [(0, "A")]}),
        )
        covjson = encode("verticalprofile", tree)
        assert labels(covjson) == ["A", "B", "C"]
        assert values_of(covjson) == [[291.0, 281.0], [290.0, 280.0], [290.0, 280.0]]

    def test_untagged_unchanged(self):
        tree = vp_tree([(48.0, 11.0, [290.0, 280.0]), (50.0, 12.0, [291.0, 281.0])], make_point)
        covjson = encode("verticalprofile", tree)
        assert labels(covjson) == [None, None]
        assert values_of(covjson) == [[290.0, 280.0], [291.0, 281.0]]

    def test_reforecast_separate_datetime(self):
        hdates = (np.datetime64("2025-07-14T00:00:00"),)
        times = (np.timedelta64(0, "h"),)
        tree = reforecast_separate_datetime_vertical_tree(
            [(48.0, 11.0, [1.0, 2.0])],
            hdates,
            times,
            (500, 850),
            point_factory=tagged(make_point, {(48.0, 11.0): [(0, "A"), (1, "B")]}),
        )
        covjson = encode("verticalprofile", tree, "from_polytope_reforecast")
        assert labels(covjson) == ["A", "B"]
        assert values_of(covjson) == [[1.0, 2.0], [1.0, 2.0]]
