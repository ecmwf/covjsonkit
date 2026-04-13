import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


class TestPositionFromPolytope:
    """Tests for Position (PointSeries) encoder's from_polytope method."""

    def _build_position_tree(self, points, param="167", steps=(0, 6)):
        """Build a Position tree.

        points: list of (lat, lon, result_list) tuples.
        result_list has one value per step.
        """
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", (param,)),
            node("step", steps),
            node("stream", ("oper",)),
            node("type", ("fc",)),
        )
        parent = tip(tree)
        for lat, lon, result in points:
            parent.add_child(make_point(lat, lon, result))
        return tree

    def test_single_point_two_steps(self):
        """1 point, 2 steps → 1 coverage with t=[step0, step6]."""
        points = [(48.0, 11.0, [264.9, 263.8])]
        tree = self._build_position_tree(points)
        encoder = Covjsonkit().encode("CoverageCollection", "Position")
        covjson = encoder.from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "PointSeries"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        # Domain axes
        axes = cov["domain"]["axes"]
        assert axes["latitude"]["values"] == [48.0]
        assert axes["longitude"]["values"] == [11.0]
        assert axes["levelist"]["values"] == [0]
        assert axes["t"]["values"] == [
            "2025-01-01T00:00:00Z",
            "2025-01-01T06:00:00Z",
        ]

        # Range
        assert "2t" in cov["ranges"]
        rng = cov["ranges"]["2t"]
        assert rng["type"] == "NdArray"
        assert rng["dataType"] == "float"
        assert rng["shape"] == [2]
        assert rng["values"] == [264.9, 263.8]

        # Metadata: "step" key should be deleted by from_polytope
        mm = cov["mars:metadata"]
        assert "step" not in mm
        assert mm["number"] == 0
        assert mm["Forecast date"] == "2025-01-01T00:00:00Z"

    def test_two_points_two_steps(self):
        """2 points, 2 steps → 2 coverages (one per point)."""
        points = [
            (48.0, 11.0, [264.9, 263.8]),
            (50.0, 13.0, [265.1, 264.2]),
        ]
        tree = self._build_position_tree(points)
        encoder = Covjsonkit().encode("CoverageCollection", "Position")
        covjson = encoder.from_polytope(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [48.0]
        assert cov0["domain"]["axes"]["longitude"]["values"] == [11.0]
        assert cov0["ranges"]["2t"]["values"] == [264.9, 263.8]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [50.0]
        assert cov1["domain"]["axes"]["longitude"]["values"] == [13.0]
        assert cov1["ranges"]["2t"]["values"] == [265.1, 264.2]

    def test_single_step(self):
        """Single step → t has just 1 value."""
        points = [(48.0, 11.0, [264.9])]
        tree = self._build_position_tree(points, steps=(0,))
        encoder = Covjsonkit().encode("CoverageCollection", "Position")
        covjson = encoder.from_polytope(tree)

        cov = covjson["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-01T00:00:00Z"]
        assert cov["ranges"]["2t"]["values"] == [264.9]
        assert cov["ranges"]["2t"]["shape"] == [1]

    def test_referencing(self):
        """Check the CRS referencing block."""
        points = [(48.0, 11.0, [264.9])]
        tree = self._build_position_tree(points, steps=(0,))
        encoder = Covjsonkit().encode("CoverageCollection", "Position")
        covjson = encoder.from_polytope(tree)

        ref = covjson["referencing"][0]
        assert ref["coordinates"] == ["latitude", "longitude", "levelist"]
        assert ref["system"]["type"] == "GeographicCRS"

    def test_parameters_block(self):
        """Top-level parameters dict should have param 167 = '2t'."""
        points = [(48.0, 11.0, [264.9])]
        tree = self._build_position_tree(points, steps=(0,))
        encoder = Covjsonkit().encode("CoverageCollection", "Position")
        covjson = encoder.from_polytope(tree)

        assert "2t" in covjson["parameters"]
        p = covjson["parameters"]["2t"]
        assert p["type"] == "Parameter"


class TestPositionFromPolytopeReforecast:
    """Tests for Position encoder's from_polytope_reforecast method."""

    def test_reforecast_single_hdate_two_points(self):
        """1 hdate, 2 points → 2 coverages (1 per point)."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("4321",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [264.9]))
        fc.add_child(make_point(50.0, 12.0, [265.1]))

        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "PointSeries"
        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        axes0 = cov0["domain"]["axes"]
        assert axes0["latitude"]["values"] == [48.0]
        assert axes0["longitude"]["values"] == [11.0]
        # t = hdate(06:00) + step(0) = 06:00
        assert axes0["t"]["values"] == ["2025-07-14T06:00:00Z"]

        cov1 = covjson["coverages"][1]
        axes1 = cov1["domain"]["axes"]
        assert axes1["latitude"]["values"] == [50.0]
        assert axes1["longitude"]["values"] == [12.0]
        assert axes1["t"]["values"] == ["2025-07-14T06:00:00Z"]

    def test_reforecast_two_hdates_two_points(self):
        """2 hdates × 2 points → 4 coverages."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
        )
        date_node = tip(tree)

        # hdate 1
        hdate1 = chain(
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("4321",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
        )
        fc1 = tip(hdate1)
        fc1.add_child(make_point(48.0, 11.0, [264.9]))
        fc1.add_child(make_point(50.0, 12.0, [265.1]))

        # hdate 2
        hdate2 = chain(
            node("hdate", (np.datetime64("2025-07-15T06:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("4321",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
        )
        fc2 = tip(hdate2)
        fc2.add_child(make_point(48.0, 11.0, [266.0]))
        fc2.add_child(make_point(50.0, 12.0, [267.0]))

        date_node.add_child(hdate1)
        date_node.add_child(hdate2)

        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 4
