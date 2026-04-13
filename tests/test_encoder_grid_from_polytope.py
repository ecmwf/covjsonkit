import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


class TestGridFromPolytope:
    """Tests for Grid encoder's from_polytope method."""

    def _build_grid_tree(self, grid_points, param="167", step=0):
        """Build a grid tree.

        grid_points: list of (lat, lon, result_value) triples.
        Each is a separate lat→lon subtree under the common metadata chain.
        """
        step_tuple = step if isinstance(step, tuple) else (step,)

        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", (param,)),
            node("step", step_tuple),
            node("stream", ("oper",)),
            node("type", ("an",)),
        )
        parent = tip(tree)
        for lat, lon, vals in grid_points:
            if not isinstance(vals, list):
                vals = [vals]
            parent.add_child(make_point(lat, lon, vals))

        return tree

    def test_2x2_grid(self):
        """2×2 grid: 2 latitudes, 2 longitudes, param 167 (2t), step 0."""
        grid_points = [
            (48.0, 11.0, [264.9]),
            (48.0, 12.0, [265.1]),
            (50.0, 11.0, [266.3]),
            (50.0, 12.0, [267.5]),
        ]
        tree = self._build_grid_tree(grid_points)
        encoder = Covjsonkit().encode("CoverageCollection", "Grid")
        covjson = encoder.from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Grid"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        # Domain axes
        axes = cov["domain"]["axes"]
        assert axes["t"]["values"] == [0]
        assert axes["levelist"]["values"] == [0]
        assert axes["latitude"]["values"] == [48.0, 50.0]
        assert axes["longitude"]["values"] == [11.0, 12.0]

        # Range
        assert "2t" in cov["ranges"]
        rng = cov["ranges"]["2t"]
        assert rng["type"] == "NdArray"
        assert rng["dataType"] == "float"
        assert rng["axisNames"] == ["t", "levelist", "latitude", "longitude"]
        assert rng["shape"] == [1, 1, 2, 2]
        assert rng["values"] == [264.9, 265.1, 266.3, 267.5]

    def test_1x1_grid(self):
        """1×1 grid: single point."""
        grid_points = [(48.0, 11.0, [264.9])]
        tree = self._build_grid_tree(grid_points)
        encoder = Covjsonkit().encode("CoverageCollection", "Grid")
        covjson = encoder.from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]
        assert cov["domain"]["axes"]["latitude"]["values"] == [48.0]
        assert cov["domain"]["axes"]["longitude"]["values"] == [11.0]
        assert cov["ranges"]["2t"]["shape"] == [1, 1, 1, 1]
        assert cov["ranges"]["2t"]["values"] == [264.9]

    def test_metadata(self):
        """mars:metadata should be populated correctly."""
        grid_points = [(48.0, 11.0, [264.9])]
        tree = self._build_grid_tree(grid_points)
        encoder = Covjsonkit().encode("CoverageCollection", "Grid")
        covjson = encoder.from_polytope(tree)

        mm = covjson["coverages"][0]["mars:metadata"]
        assert mm["class"] == "od"
        assert mm["Forecast date"] == "2025-01-01T00:00:00Z"
        assert mm["number"] == 0

    def test_referencing(self):
        """Check the CRS referencing block."""
        grid_points = [(48.0, 11.0, [264.9])]
        tree = self._build_grid_tree(grid_points)
        encoder = Covjsonkit().encode("CoverageCollection", "Grid")
        covjson = encoder.from_polytope(tree)

        ref = covjson["referencing"][0]
        assert ref["coordinates"] == ["latitude", "longitude", "levelist"]
        assert ref["system"]["type"] == "GeographicCRS"

    def test_parameters_block(self):
        """Top-level parameters dict should have param 167 = '2t'."""
        grid_points = [(48.0, 11.0, [264.9])]
        tree = self._build_grid_tree(grid_points)
        encoder = Covjsonkit().encode("CoverageCollection", "Grid")
        covjson = encoder.from_polytope(tree)

        assert "2t" in covjson["parameters"]
        p = covjson["parameters"]["2t"]
        assert p["type"] == "Parameter"


class TestGridFromPolytopeReforecast:
    """Tests for Grid encoder's from_polytope_reforecast method."""

    def test_reforecast_single_hdate_2x2_grid(self):
        """Single hdate with 2×2 grid → 1 Grid coverage."""
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
        fc.add_child(make_point(48.0, 12.0, [265.1]))
        fc.add_child(make_point(50.0, 11.0, [266.3]))
        fc.add_child(make_point(50.0, 12.0, [267.5]))
        covjson = Covjsonkit().encode("CoverageCollection", "Grid").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Grid"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        axes = cov["domain"]["axes"]
        assert len(axes["latitude"]["values"]) == 2
        assert len(axes["longitude"]["values"]) == 2
        assert axes["t"]["values"] == [0]

        rng = cov["ranges"]["2t"]
        assert rng["shape"] == [1, 1, 2, 2]
        assert rng["values"] == [264.9, 265.1, 266.3, 267.5]

    def test_reforecast_two_hdates_2x2_grid(self):
        """Two hdates each with 2×2 grid → 2 Grid coverages."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
        )
        date_node = tip(tree)

        for hdate_val in [np.datetime64("2025-07-14T06:00:00"), np.datetime64("2025-07-15T06:00:00")]:
            branch = chain(
                node("hdate", (hdate_val,)),
                node("domain", ("g",)),
                node("expver", ("4321",)),
                node("levtype", ("sfc",)),
                node("param", ("167",)),
                node("step", (0,)),
                node("stream", ("efcl",)),
                node("type", ("sfo",)),
            )
            fc = tip(branch)
            fc.add_child(make_point(48.0, 11.0, [264.9]))
            fc.add_child(make_point(48.0, 12.0, [265.1]))
            fc.add_child(make_point(50.0, 11.0, [266.3]))
            fc.add_child(make_point(50.0, 12.0, [267.5]))
            date_node.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "Grid").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2
