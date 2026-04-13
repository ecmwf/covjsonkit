import numpy as np
from conftest import chain, make_leaf, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


class TestVerticalProfileFromPolytope:
    """Tests for VerticalProfile encoder's from_polytope method."""

    def _build_vp_tree(self, param="130", levels_values=None, step=0, lat=48.0, lon=11.0):
        """Build a vertical-profile tree.

        The tree has one levelist node whose values tuple contains ALL requested
        pressure levels.  The leaf result array is ordered
        [level0_val, level1_val, ...].
        """
        if levels_values is None:
            levels_values = {1000: 290.1, 850: 280.2, 500: 250.3}

        levels = tuple(levels_values.keys())
        result = [levels_values[lv] for lv in levels]

        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("pl",)),
            node("param", (param,)),
            node("step", (step,)),
            node("stream", ("oper",)),
            node("type", ("an",)),
            node("levelist", levels),
            make_point(lat, lon, result),
        )
        return tree

    def test_single_point_three_levels(self):
        """1 point, 3 pressure levels, param 130 (t), step 0."""
        tree = self._build_vp_tree()
        encoder = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson = encoder.from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "VerticalProfile"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        # Domain axes
        axes = cov["domain"]["axes"]
        assert axes["latitude"]["values"] == [48.0]
        assert axes["longitude"]["values"] == [11.0]
        assert axes["levelist"]["values"] == [1000, 850, 500]
        assert axes["t"]["values"] == ["2025-01-01T00:00:00Z"]

        # Range
        assert "t" in cov["ranges"]
        rng = cov["ranges"]["t"]
        assert rng["type"] == "NdArray"
        assert rng["dataType"] == "float"
        assert rng["axisNames"] == ["levelist"]
        assert rng["shape"] == [3]
        assert rng["values"] == [290.1, 280.2, 250.3]

        # Metadata
        mm = cov["mars:metadata"]
        assert mm["class"] == "od"
        assert mm["Forecast date"] == "2025-01-01T00:00:00Z"
        assert mm["number"] == 0
        assert mm["step"] == 0

    def test_two_points_two_levels(self):
        """2 spatial points, 2 levels → 2 coverages (one per point)."""
        levels = (1000, 500)
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-06-15T12:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("pl",)),
            node("param", ("130",)),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("an",)),
            node("levelist", levels),
        )
        lev_node = tip(tree)
        # Point 1: lat=48, lon=11
        lev_node.add_child(make_point(48.0, 11.0, [290.1, 250.3]))
        # Point 2: lat=50, lon=13
        lev_node.add_child(make_point(50.0, 13.0, [288.5, 248.7]))

        encoder = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson = encoder.from_polytope(tree)

        assert len(covjson["coverages"]) == 2

        # First coverage = first point
        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [48.0]
        assert cov0["domain"]["axes"]["longitude"]["values"] == [11.0]
        assert cov0["domain"]["axes"]["levelist"]["values"] == [1000, 500]
        assert cov0["ranges"]["t"]["values"] == [290.1, 250.3]
        assert cov0["ranges"]["t"]["shape"] == [2]

        # Second coverage = second point
        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [50.0]
        assert cov1["domain"]["axes"]["longitude"]["values"] == [13.0]
        assert cov1["ranges"]["t"]["values"] == [288.5, 248.7]

    def test_single_level(self):
        """Edge case: only 1 pressure level."""
        tree = self._build_vp_tree(levels_values={500: 250.3})
        encoder = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson = encoder.from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]
        assert cov["domain"]["axes"]["levelist"]["values"] == [500]
        assert cov["ranges"]["t"]["values"] == [250.3]
        assert cov["ranges"]["t"]["shape"] == [1]

    def test_step_offset(self):
        """Step=6 should shift the t coordinate by 6 hours."""
        tree = self._build_vp_tree(step=6, levels_values={1000: 290.0})
        encoder = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson = encoder.from_polytope(tree)

        cov = covjson["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-01T06:00:00Z"]
        assert cov["mars:metadata"]["step"] == 6

    def test_referencing(self):
        """Check the CRS referencing block."""
        tree = self._build_vp_tree(levels_values={1000: 290.0})
        encoder = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson = encoder.from_polytope(tree)

        ref = covjson["referencing"][0]
        assert ref["coordinates"] == ["latitude", "longitude", "levelist"]
        assert ref["system"]["type"] == "GeographicCRS"

    def test_parameters_block(self):
        """The top-level parameters dict should contain param 130 = 't'."""
        tree = self._build_vp_tree(levels_values={1000: 290.0})
        encoder = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson = encoder.from_polytope(tree)

        assert "t" in covjson["parameters"]
        p = covjson["parameters"]["t"]
        assert p["type"] == "Parameter"
        assert "Temperature" in p["observedProperty"]["label"]["en"]


class TestVerticalProfileFromPolytopeReforecast:
    """Tests for VerticalProfile encoder's from_polytope_reforecast method."""

    def test_reforecast_single_hdate_three_levels(self):
        """1 hdate, 3 pressure levels, 1 point → 1 coverage."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("4321",)),
            node("levtype", ("pl",)),
            node("param", ("130",)),
            node("step", (6,)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
            node("levelist", (1000, 850, 500)),
            node("latitude", (48.0,)),
            make_leaf(11.0, [290.1, 280.2, 250.3]),
        )
        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "VerticalProfile"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        axes = cov["domain"]["axes"]
        assert axes["levelist"]["values"] == [1000, 850, 500]
        assert axes["latitude"]["values"] == [48.0]
        assert axes["longitude"]["values"] == [11.0]
        # t = hdate(06:00) + step(6h) = 12:00
        assert axes["t"]["values"] == ["2025-07-14T12:00:00Z"]

        assert "t" in cov["ranges"]
        rng = cov["ranges"]["t"]
        assert rng["type"] == "NdArray"
        assert rng["dataType"] == "float"
        assert rng["shape"] == [3]
        assert rng["values"] == [290.1, 280.2, 250.3]

    def test_reforecast_two_hdates_three_levels(self):
        """2 hdates, 3 levels, 1 point → 2 coverages (one per hdate)."""
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
            node("levtype", ("pl",)),
            node("param", ("130",)),
            node("step", (6,)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
            node("levelist", (1000, 850, 500)),
            node("latitude", (48.0,)),
            make_leaf(11.0, [290.1, 280.2, 250.3]),
        )

        # hdate 2
        hdate2 = chain(
            node("hdate", (np.datetime64("2025-07-15T06:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("4321",)),
            node("levtype", ("pl",)),
            node("param", ("130",)),
            node("step", (6,)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
            node("levelist", (1000, 850, 500)),
            node("latitude", (48.0,)),
            make_leaf(11.0, [291.0, 281.0, 251.0]),
        )

        date_node.add_child(hdate1)
        date_node.add_child(hdate2)

        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["t"]["values"] == [290.1, 280.2, 250.3]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-15T12:00:00Z"]
        assert cov1["ranges"]["t"]["values"] == [291.0, 281.0, 251.0]
