import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


class TestPathFromPolytope:
    """Tests for Path (Trajectory) encoder's from_polytope method."""

    def _build_path_tree(self, points, param="167", step=0):
        """Build a path tree with given points.

        points: list of (lat, lon, result_value) tuples
        Each point is a separate lat→lon subtree.
        The step value(s) become the 't' in the composite tuple.
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
            node("type", ("fc",)),
        )
        parent = tip(tree)
        for lat, lon, vals in points:
            if not isinstance(vals, list):
                vals = [vals]
            parent.add_child(make_point(lat, lon, vals))

        return tree

    def test_two_points_single_step(self):
        """2 points along a path, single step → 1 Trajectory coverage."""
        points = [
            (48.0, 11.0, [264.9]),
            (49.0, 12.0, [265.1]),
        ]
        tree = self._build_path_tree(points)
        encoder = Covjsonkit().encode("CoverageCollection", "Path")
        covjson = encoder.from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Trajectory"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        # Domain: composite tuples [step, lat, lon, level]
        comp = cov["domain"]["axes"]["composite"]
        assert comp["dataType"] == "tuple"
        assert comp["coordinates"] == ["t", "x", "y", "z"]
        assert len(comp["values"]) == 2
        assert comp["values"][0] == [0, 48.0, 11.0, 0]
        assert comp["values"][1] == [0, 49.0, 12.0, 0]

        # Range
        assert "2t" in cov["ranges"]
        rng = cov["ranges"]["2t"]
        assert rng["type"] == "NdArray"
        assert rng["dataType"] == "float"
        assert rng["shape"] == [2]
        assert rng["values"] == [264.9, 265.1]

        # Metadata: levelist should be removed
        mm = cov["mars:metadata"]
        assert "levelist" not in mm
        assert mm["number"] == 0
        assert mm["Forecast date"] == "2025-01-01T00:00:00Z"

    def test_three_points_along_path(self):
        """3 points along a path → composite has 3 tuples."""
        points = [
            (48.0, 11.0, [264.9]),
            (49.0, 12.0, [265.1]),
            (50.0, 13.0, [266.3]),
        ]
        tree = self._build_path_tree(points)
        encoder = Covjsonkit().encode("CoverageCollection", "Path")
        covjson = encoder.from_polytope(tree)

        cov = covjson["coverages"][0]
        assert len(cov["domain"]["axes"]["composite"]["values"]) == 3
        assert cov["ranges"]["2t"]["shape"] == [3]
        assert cov["ranges"]["2t"]["values"] == [264.9, 265.1, 266.3]

    def test_referencing(self):
        """Check the CRS referencing block uses [t, x, y, z]."""
        points = [(48.0, 11.0, [264.9])]
        tree = self._build_path_tree(points)
        encoder = Covjsonkit().encode("CoverageCollection", "Path")
        covjson = encoder.from_polytope(tree)

        ref = covjson["referencing"][0]
        assert ref["coordinates"] == ["t", "x", "y", "z"]
        assert ref["system"]["type"] == "GeographicCRS"

    def test_parameters_block(self):
        """Top-level parameters dict should have param 167 = '2t'."""
        points = [(48.0, 11.0, [264.9])]
        tree = self._build_path_tree(points)
        encoder = Covjsonkit().encode("CoverageCollection", "Path")
        covjson = encoder.from_polytope(tree)

        assert "2t" in covjson["parameters"]
        p = covjson["parameters"]["2t"]
        assert p["type"] == "Parameter"


class TestPathFromPolytopeReforecast:
    """Tests for Path (Trajectory) encoder's from_polytope_reforecast method."""

    def test_reforecast_single_hdate_two_points(self):
        """Single hdate with 2 path points → 1 Trajectory coverage."""
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
        covjson = Covjsonkit().encode("CoverageCollection", "Path").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Trajectory"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        comp = cov["domain"]["axes"]["composite"]
        assert comp["dataType"] == "tuple"
        assert comp["coordinates"] == ["t", "x", "y", "z"]
        assert comp["values"] == [[0, 48.0, 11.0, 0], [0, 50.0, 12.0, 0]]

        assert cov["ranges"]["2t"]["values"] == [264.9, 265.1]

        mm = cov["mars:metadata"]
        assert "Forecast date" in mm
        assert "2025-07-14" in mm["Forecast date"]

    def test_reforecast_two_hdates_two_points(self):
        """Two hdates each with 2 path points → 2 Trajectory coverages."""
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
            fc.add_child(make_point(50.0, 12.0, [265.1]))
            date_node.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "Path").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2
