import numpy as np
from conftest import (
    COMPOSITE_TWO_POINTS,
    REFORECAST_METADATA_BASE,
    efas_tree,
    forecast_tree,
    reforecast_branch,
    reforecast_tree,
)

from covjsonkit.api import Covjsonkit


class TestFrameFromPolytope:
    def test_single_date_single_step_two_points(self):
        tree = forecast_tree([(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])])
        covjson = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": ["2025-01-01T00:00:00Z"]},
            "composite": COMPOSITE_TWO_POINTS,
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["composite"],
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


class TestFrameFromPolytopeReforecast:
    def test_reforecast_single_hdate_two_points(self):
        points = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), points),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": ["2025-07-14T06:00:00Z"]},
            "composite": COMPOSITE_TWO_POINTS,
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["composite"],
                "values": [264.9, 265.1],
            }
        }

        assert cov["mars:metadata"] == REFORECAST_METADATA_BASE

    def test_reforecast_two_hdates_two_points(self):
        points = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), points),
                reforecast_branch(np.datetime64("2025-07-15T06:00:00"), [(48.0, 11.0, [266.0]), (50.0, 12.0, [267.0])]),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope_reforecast(tree)

        expected = [
            ("2025-07-14T06:00:00Z", [264.9, 265.1]),
            ("2025-07-15T06:00:00Z", [266.0, 267.0]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (fc_date, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "t": {"values": [fc_date]},
                "composite": COMPOSITE_TWO_POINTS,
            }
            assert cov["ranges"]["2t"]["values"] == vals
            assert cov["mars:metadata"] == REFORECAST_METADATA_BASE


class TestFrameEfasAnoffset:
    """anoffset valid-time correction for class=ce, stream=efas (regular forecast)."""

    def test_efas_anoffset_shifts_valid_time(self):
        # 00:00 + step 6h - anoffset 6h = 00:00
        tree = efas_tree([(48.0, 11.0, [12.5])], anoffset=6)
        cov = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope(tree)["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2026-01-01T00:00:00Z"]

    def test_efas_anoffset_absent_leaves_valid_time_unchanged(self):
        # No anoffset -> 00:00 + step 6h = 06:00
        tree = efas_tree([(48.0, 11.0, [12.5])])
        cov = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope(tree)["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2026-01-01T06:00:00Z"]

    def test_anoffset_ignored_for_non_efas_stream(self):
        # anoffset present but stream != efas -> valid-time not shifted (06:00)
        tree = efas_tree([(48.0, 11.0, [12.5])], anoffset=6, cls="od", stream="oper")
        cov = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope(tree)["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2026-01-01T06:00:00Z"]
