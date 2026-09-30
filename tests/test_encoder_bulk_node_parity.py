"""Parity tests for array-backed bulk lat/lon results.

Polytope returns all the (latitude, longitude) points under one path as a single
array-backed leaf: a ``BulkMergedTensorIndexNode`` for unstructured grids, or a
``BulkGridTensorIndexNode`` (which keeps the latitude rows and their compressed
longitudes) for structured grids. These tests assert that covjsonkit produces
byte-identical CoverageJSON whether the source tree uses the legacy layout or a
bulk leaf, exercising all tree walkers.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import (
    BulkMergedTensorIndexNode,
    bulkify,
    chain,
    forecast_tree,
    gridify,
    make_merged_point,
    make_point,
    make_row,
    month_tree,
    node,
    reforecast_branch,
    reforecast_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

pytestmark = pytest.mark.skipif(
    BulkMergedTensorIndexNode is None,
    reason="polytope build lacks BulkMergedTensorIndexNode (array-backed lat/lon leaves)",
)

TWO_POINTS = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]
MULTI_STEP_POINTS = [(48.0, 11.0, [264.9, 270.1]), (50.0, 12.0, [265.1, 271.3]), (50.0, 13.0, [266.0, 272.0])]

FEATURES = ["BoundingBox", "Grid", "Frame", "Circle", "Shapefile", "Polygon", "Path", "Position"]


def _encode(tree, feature="BoundingBox", reforecast=False):
    api = Covjsonkit().encode("CoverageCollection", feature)
    if reforecast:
        return api.from_polytope_reforecast(tree)
    return api.from_polytope(tree)


def _encode_month(tree, feature="BoundingBox"):
    return Covjsonkit().encode("CoverageCollection", feature).from_polytope_month(tree)


def _encode_step(tree, feature="PointSeries"):
    return Covjsonkit().encode("CoverageCollection", feature).from_polytope_step(tree)


class TestBulkNodeParity:
    @pytest.mark.parametrize("feature", FEATURES)
    def test_walk_tree_single_step(self, feature):
        legacy = _encode(forecast_tree(TWO_POINTS, point_factory=make_point), feature)
        bulk = _encode(bulkify(forecast_tree(TWO_POINTS, point_factory=make_merged_point)), feature)
        grid = _encode(gridify(forecast_tree(TWO_POINTS, point_factory=make_point)), feature)
        assert bulk == legacy
        assert grid == legacy

    def test_walk_tree_multi_step(self):
        legacy = _encode(forecast_tree(MULTI_STEP_POINTS, step=(0, 6), point_factory=make_point))
        bulk = _encode(bulkify(forecast_tree(MULTI_STEP_POINTS, step=(0, 6), point_factory=make_merged_point)))
        grid = _encode(gridify(forecast_tree(MULTI_STEP_POINTS, step=(0, 6), point_factory=make_point)))
        assert bulk == legacy
        assert grid == legacy

    def test_grid_row_with_compressed_longitudes(self):
        # One latitude row holding two compressed longitudes, as the hullslicer returns them
        def build():
            tree = forecast_tree([], step=(0, 6))
            fc = tip(tree)
            fc.add_child(make_row(48.0, (11.0, 11.5), [[264.9, 270.1], [264.5, 270.5]]))
            fc.add_child(make_row(50.0, (12.0,), [[265.1, 271.3]]))
            return tree

        legacy = _encode(build())
        grid = _encode(gridify(build()))
        assert len(legacy["coverages"]) == 2
        assert grid == legacy

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

        legacy = _encode(build(make_point))
        assert _encode(bulkify(build(make_merged_point))) == legacy
        assert _encode(gridify(build(make_point))) == legacy

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
        assert _encode(bulkify(build(make_merged_point)), reforecast=True) == legacy
        assert _encode(gridify(build(make_point)), reforecast=True) == legacy

    def test_walk_tree_month(self):
        points = [(48.0, 11.0, [264.9, 265.9]), (50.0, 12.0, [266.1, 267.1])]
        legacy = _encode_month(month_tree(points, point_factory=make_point))
        bulk = _encode_month(bulkify(month_tree(points, point_factory=make_merged_point)))
        grid = _encode_month(gridify(month_tree(points, point_factory=make_point)))
        assert len(legacy["coverages"]) == 2
        assert bulk == legacy
        assert grid == legacy

    def test_walk_tree_step(self):
        # Time series (step walker): date -> time axes, two points over two times
        def build(point_factory):
            tree = chain(
                TensorIndexTree(),
                node("class", ("od",)),
                node("date", (np.datetime64("2025-01-01T00:00:00"),)),
                node("time", (pd.Timedelta(hours=0), pd.Timedelta(hours=6))),
                node("domain", ("g",)),
                node("expver", ("0001",)),
                node("levtype", ("sfc",)),
                node("param", ("167",)),
                node("stream", ("oper",)),
                node("type", ("fc",)),
            )
            fc = tip(tree)
            fc.add_child(point_factory(48.0, 11.0, [264.9, 270.1]))
            fc.add_child(point_factory(50.0, 12.0, [265.1, 271.3]))
            return tree

        legacy = _encode_step(build(make_point))
        assert len(legacy["coverages"]) == 2
        assert _encode_step(bulkify(build(make_merged_point))) == legacy
        assert _encode_step(gridify(build(make_point))) == legacy
