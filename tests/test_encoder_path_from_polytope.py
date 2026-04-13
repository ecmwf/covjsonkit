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
        covjson = Covjsonkit().encode("CoverageCollection", "Path").from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Trajectory"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "composite": {
                "dataType": "tuple",
                "coordinates": ["t", "x", "y", "z"],
                "values": [[0, 48.0, 11.0, 0], [0, 49.0, 12.0, 0]],
            }
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["2t"],
                "values": [264.9, 265.1],
            }
        }

        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "step": 0,
            "stream": "oper",
            "type": "fc",
            "number": 0,
        }

        # Collection-level referencing
        assert covjson["referencing"] == [
            {
                "coordinates": ["t", "x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        ]

        # Collection-level parameters
        assert "2t" in covjson["parameters"]
        assert covjson["parameters"]["2t"]["type"] == "Parameter"
        assert covjson["parameters"]["2t"]["observedProperty"]["id"] == "2t"


class TestPathFromPolytopeReforecast:
    """Tests for Path (Trajectory) encoder's from_polytope_reforecast method."""

    SHARED_METADATA = {
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

    EXPECTED_AXES = {
        "composite": {
            "dataType": "tuple",
            "coordinates": ["t", "x", "y", "z"],
            "values": [[0, 48.0, 11.0, 0], [0, 50.0, 12.0, 0]],
        }
    }

    EXPECTED_RANGES = {
        "2t": {
            "type": "NdArray",
            "dataType": "float",
            "shape": [2],
            "axisNames": ["2t"],
            "values": [264.9, 265.1],
        }
    }

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

        assert cov["domain"]["axes"] == self.EXPECTED_AXES
        assert cov["ranges"] == self.EXPECTED_RANGES
        assert cov["mars:metadata"] == {
            **self.SHARED_METADATA,
            "Forecast date": "2025-07-14T06:00:00Z",
        }

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

        expected = [
            "2025-07-14T06:00:00Z",
            "2025-07-15T06:00:00Z",
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, fc_date in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == self.EXPECTED_AXES
            assert cov["ranges"] == self.EXPECTED_RANGES
            assert cov["mars:metadata"] == {**self.SHARED_METADATA, "Forecast date": fc_date}
