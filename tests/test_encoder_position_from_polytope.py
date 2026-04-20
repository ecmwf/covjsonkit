import numpy as np
from conftest import forecast_tree, reforecast_branch, reforecast_tree

from covjsonkit.api import Covjsonkit


class TestPositionFromPolytope:
    """Tests for Position (PointSeries) encoder's from_polytope method."""

    def test_single_point_two_steps(self):
        """1 point, 2 steps -> 1 coverage with t=[step0, step6]."""
        tree = forecast_tree([(48.0, 11.0, [264.9, 263.8])], step=(0, 6))
        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "PointSeries"

        # Referencing
        ref = covjson["referencing"][0]
        assert ref["coordinates"] == ["latitude", "longitude", "levelist"]
        assert ref["system"]["type"] == "GeographicCRS"

        # Parameters
        assert "2t" in covjson["parameters"]
        assert covjson["parameters"]["2t"]["type"] == "Parameter"

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]},
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["2t"],
                "values": [264.9, 263.8],
            }
        }

        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "number": 0,
        }

    def test_two_points_two_steps(self):
        """2 points, 2 steps -> 2 coverages (one per point)."""
        tree = forecast_tree(
            [(48.0, 11.0, [264.9, 263.8]), (50.0, 13.0, [265.1, 264.2])],
            step=(0, 6),
        )
        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope(tree)

        shared_metadata = {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "number": 0,
        }

        expected = [
            (48.0, 11.0, [264.9, 263.8]),
            (50.0, 13.0, [265.1, 264.2]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, lon, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "latitude": {"values": [lat]},
                "longitude": {"values": [lon]},
                "levelist": {"values": [0]},
                "t": {"values": ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]},
            }
            assert cov["ranges"]["2t"]["values"] == vals
            assert cov["mars:metadata"] == shared_metadata

    def test_single_step(self):
        """Edge case: 1 step -> shape [1], single t-value."""
        tree = forecast_tree([(48.0, 11.0, [264.9])], step=(0,))
        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-01-01T00:00:00Z"]},
        }
        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [1],
                "axisNames": ["2t"],
                "values": [264.9],
            }
        }


class TestPositionFromPolytopeReforecast:
    """Tests for Position encoder's from_polytope_reforecast method."""

    def test_reforecast_single_hdate_two_points(self):
        """1 hdate, 2 points -> 2 coverages (1 per point)."""
        points = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), points),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope_reforecast(tree)

        shared_metadata = {
            "class": "ce",
            "date": "2024-03-01",
            "Forecast date": "2025-07-14T06:00:00Z",
            "domain": "g",
            "expver": "4321",
            "levtype": "sfc",
            "stream": "efcl",
            "type": "sfo",
            "number": 0,
        }

        expected = [
            (48.0, 11.0, [264.9]),
            (50.0, 12.0, [265.1]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, lon, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "latitude": {"values": [lat]},
                "longitude": {"values": [lon]},
                "levelist": {"values": [0]},
                "t": {"values": ["2025-07-14T06:00:00Z"]},
            }
            assert cov["ranges"]["2t"]["values"] == vals
            assert cov["mars:metadata"] == shared_metadata

    def test_reforecast_two_hdates_two_points(self):
        """2 hdates x 2 points -> 4 coverages."""
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]),
                reforecast_branch(np.datetime64("2025-07-15T06:00:00"), [(48.0, 11.0, [266.0]), (50.0, 12.0, [267.0])]),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope_reforecast(tree)

        shared_metadata = {
            "class": "ce",
            "date": "2024-03-01",
            "domain": "g",
            "expver": "4321",
            "levtype": "sfc",
            "stream": "efcl",
            "type": "sfo",
            "number": 0,
        }

        expected = [
            (48.0, 11.0, ["2025-07-14T06:00:00Z"], [264.9], "2025-07-14T06:00:00Z"),
            (48.0, 11.0, ["2025-07-15T06:00:00Z"], [266.0], "2025-07-15T06:00:00Z"),
            (50.0, 12.0, ["2025-07-14T06:00:00Z"], [265.1], "2025-07-14T06:00:00Z"),
            (50.0, 12.0, ["2025-07-15T06:00:00Z"], [267.0], "2025-07-15T06:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, lon, t, vals, fc_date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "latitude": {"values": [lat]},
                "longitude": {"values": [lon]},
                "levelist": {"values": [0]},
                "t": {"values": t},
            }
            assert cov["ranges"]["2t"]["values"] == vals
            assert cov["mars:metadata"] == {**shared_metadata, "Forecast date": fc_date}
