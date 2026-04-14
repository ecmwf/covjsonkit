import numpy as np
from conftest import (
    COMPOSITE_TWO_POINTS_XYZ,
    REFORECAST_METADATA_BASE,
    forecast_tree,
    reforecast_branch,
    reforecast_tree,
)

from covjsonkit.api import Covjsonkit


class TestShapefileFromPolytope:
    def test_single_date_single_step_two_points(self):
        tree = forecast_tree([(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])])
        covjson = Covjsonkit().encode("CoverageCollection", "Shapefile").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": ["2025-01-01T00:00:00Z"]},
            "composite": COMPOSITE_TWO_POINTS_XYZ,
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


class TestShapefileFromPolytopeReforecast:
    def test_reforecast_single_hdate_two_points(self):
        points = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), points),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Shapefile").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": ["2025-07-14T06:00:00Z"]},
            "composite": COMPOSITE_TWO_POINTS_XYZ,
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

        assert cov["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **REFORECAST_METADATA_BASE}

    def test_reforecast_two_hdates_two_points(self):
        points = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), points),
                reforecast_branch(np.datetime64("2025-07-15T06:00:00"), [(48.0, 11.0, [266.0]), (50.0, 12.0, [267.0])]),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Shapefile").from_polytope_reforecast(tree)

        expected = [
            (["2025-07-14T06:00:00Z"], [264.9, 265.1], "2025-07-14T06:00:00Z"),
            (["2025-07-15T06:00:00Z"], [266.0, 267.0], "2025-07-15T06:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (t_vals, range_vals, fc_date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "t": {"values": t_vals},
                "composite": COMPOSITE_TWO_POINTS_XYZ,
            }
            assert cov["ranges"]["2t"]["values"] == range_vals
            assert cov["mars:metadata"] == {"Forecast date": fc_date, **REFORECAST_METADATA_BASE}
