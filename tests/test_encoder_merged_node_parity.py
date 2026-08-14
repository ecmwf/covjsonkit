"""Parity tests for compacted unstructured-grid results.

Polytope emits ``MergedTensorIndexNode`` leaves for unstructured grids (ICON,
Lambert LAM), compacting each (latitude, longitude) point and its result into a
single leaf instead of a latitude -> longitude-leaf subtree. These tests assert
that covjsonkit produces byte-identical CoverageJSON whether the source tree
uses the legacy layout (``make_point``) or the compacted layout
(``make_merged_point``), exercising all tree walkers.
"""

import numpy as np
import pytest
from conftest import (
    MergedTensorIndexNode,
    chain,
    forecast_tree,
    make_merged_point,
    make_point,
    month_tree,
    node,
    reforecast_branch,
    reforecast_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

# These tests require a polytope build with compacted unstructured-grid support.
# On released polytope (no MergedTensorIndexNode) they are skipped entirely.
pytestmark = pytest.mark.skipif(
    MergedTensorIndexNode is None,
    reason="polytope build lacks MergedTensorIndexNode (compacted unstructured support)",
)

TWO_POINTS = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]


def _encode(tree, feature="BoundingBox", reforecast=False):
    api = Covjsonkit().encode("CoverageCollection", feature)
    if reforecast:
        return api.from_polytope_reforecast(tree)
    return api.from_polytope(tree)


def _encode_month(tree, feature="BoundingBox"):
    return Covjsonkit().encode("CoverageCollection", feature).from_polytope_month(tree)


class TestMergedNodeParity:
    def test_walk_tree_single_date_single_step(self):
        legacy = _encode(forecast_tree(TWO_POINTS, point_factory=make_point))
        merged = _encode(forecast_tree(TWO_POINTS, point_factory=make_merged_point))
        assert merged == legacy

    def test_walk_tree_multi_step(self):
        points = [(48.0, 11.0, [264.9, 270.1]), (50.0, 12.0, [265.1, 271.3])]
        legacy = _encode(forecast_tree(points, step=(0, 6), point_factory=make_point))
        merged = _encode(forecast_tree(points, step=(0, 6), point_factory=make_merged_point))
        assert merged == legacy

    def test_walk_tree_two_dates_two_steps(self):
        def build(point_factory):
            tree = chain(TensorIndexTree(), node("class", ("od",)))
            cls = tip(tree)
            for date_val, vals in [
                (np.datetime64("2025-01-01T00:00:00"), [[264.9, 270.1], [265.1, 271.3]]),
                (np.datetime64("2025-01-02T00:00:00"), [[266.0, 272.0], [267.0, 273.0]]),
            ]:
                branch = chain(
                    node("date", (date_val,)),
                    node("domain", ("g",)),
                    node("expver", ("0001",)),
                    node("levtype", ("sfc",)),
                    node("param", ("167",)),
                    node("step", (0, 6)),
                    node("stream", ("oper",)),
                    node("type", ("fc",)),
                )
                fc = tip(branch)
                fc.add_child(point_factory(48.0, 11.0, vals[0]))
                fc.add_child(point_factory(50.0, 12.0, vals[1]))
                cls.add_child(branch)
            return tree

        assert _encode(build(make_merged_point)) == _encode(build(make_point))

    def test_reforecast_walker(self):
        def build(point_factory):
            return reforecast_tree(
                [
                    reforecast_branch(np.datetime64("2025-07-14T06:00:00"), TWO_POINTS, point_factory=point_factory),
                    reforecast_branch(
                        np.datetime64("2025-07-15T06:00:00"),
                        [(48.0, 11.0, [266.0]), (50.0, 12.0, [267.0])],
                        point_factory=point_factory,
                    ),
                ]
            )

        legacy = _encode(build(make_point), reforecast=True)
        merged = _encode(build(make_merged_point), reforecast=True)
        assert merged == legacy

    def test_walk_tree_month(self):
        # Monthly-mean (year/month axes) path: two years x one month, two points.
        points = [(48.0, 11.0, [264.9, 265.9]), (50.0, 12.0, [266.1, 267.1])]
        legacy = _encode_month(month_tree(points, point_factory=make_point))
        merged = _encode_month(month_tree(points, point_factory=make_merged_point))
        assert len(legacy["coverages"]) == 2
        assert merged == legacy
