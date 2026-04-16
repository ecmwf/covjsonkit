import numpy as np
from conftest import (
    REFORECAST_METADATA_BASE,
    forecast_tree,
    reforecast_branch,
    reforecast_tree,
)

from covjsonkit.api import Covjsonkit

THREE_POINTS_COMPOSITE = {
    "dataType": "tuple",
    "coordinates": ["latitude", "longitude", "levelist"],
    "values": [
        [48.0, 11.0, 0],
        [49.0, 11.5, 0],
        [50.0, 12.0, 0],
    ],
}

THREE_POINTS = [(48.0, 11.0, [264.9]), (49.0, 11.5, [265.5]), (50.0, 12.0, [266.1])]


class TestCircleFromPolytope:
    def test_single_date_single_step_three_points(self):
        tree = forecast_tree(THREE_POINTS)
        covjson = Covjsonkit().encode("CoverageCollection", "Circle").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": ["2025-01-01T00:00:00Z"]},
            "composite": THREE_POINTS_COMPOSITE,
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["2t"],
                "values": [264.9, 265.5, 266.1],
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


class TestCircleFromPolytopeReforecast:
    def test_reforecast_single_hdate_three_points(self):
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), THREE_POINTS),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Circle").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": ["2025-07-14T06:00:00Z"]},
            "composite": THREE_POINTS_COMPOSITE,
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["2t"],
                "values": [264.9, 265.5, 266.1],
            }
        }

        assert cov["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **REFORECAST_METADATA_BASE}

    def test_reforecast_two_hdates_three_points(self):
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), THREE_POINTS),
                reforecast_branch(
                    np.datetime64("2025-07-15T06:00:00"),
                    [(48.0, 11.0, [270.0]), (49.0, 11.5, [271.0]), (50.0, 12.0, [272.0])],
                ),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Circle").from_polytope_reforecast(tree)

        expected = [
            (["2025-07-14T06:00:00Z"], [264.9, 265.5, 266.1], "2025-07-14T06:00:00Z"),
            (["2025-07-15T06:00:00Z"], [270.0, 271.0, 272.0], "2025-07-15T06:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (t_vals, range_vals, fc_date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "t": {"values": t_vals},
                "composite": THREE_POINTS_COMPOSITE,
            }
            assert cov["ranges"]["2t"]["values"] == range_vals
            assert cov["mars:metadata"] == {"Forecast date": fc_date, **REFORECAST_METADATA_BASE}
