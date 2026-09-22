import numpy as np
from conftest import (
    REFORECAST_METADATA_BASE,
    efas_tree,
    forecast_tree,
    reforecast_branch,
    reforecast_tree,
)

from covjsonkit.api import Covjsonkit

TWO_POINTS = [(48.0, 11.0, [264.9]), (50.0, 12.0, [265.1])]


class TestPathFromPolytope:
    """Tests for Path (Trajectory) encoder's from_polytope method."""

    def test_two_points_single_step(self):
        """2 points along a path, single step -> 1 Trajectory coverage."""
        points = [(48.0, 11.0, [264.9]), (49.0, 12.0, [265.1])]
        tree = forecast_tree(points)
        covjson = Covjsonkit().encode("CoverageCollection", "Path").from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Trajectory"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "composite": {
                "dataType": "tuple",
                "coordinates": ["t", "x", "y"],
                "values": [
                    ["2025-01-01T00:00:00Z", 11.0, 48.0],
                    ["2025-01-01T00:00:00Z", 12.0, 49.0],
                ],
            }
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

        # Collection-level referencing (split GeographicCRS / TemporalRS)
        assert covjson["referencing"] == [
            {
                "coordinates": ["x", "y"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            },
            {
                "coordinates": ["t"],
                "system": {"type": "TemporalRS", "calendar": "Gregorian"},
            },
        ]

        # Collection-level parameters
        assert "2t" in covjson["parameters"]
        assert covjson["parameters"]["2t"]["type"] == "Parameter"
        assert covjson["parameters"]["2t"]["observedProperty"]["id"] == "2t"


class TestPathFromPolytopeReforecast:
    """Tests for Path (Trajectory) encoder's from_polytope_reforecast method."""

    @staticmethod
    def axes_for(t_val):
        return {
            "composite": {
                "dataType": "tuple",
                "coordinates": ["t", "x", "y"],
                "values": [[t_val, 11.0, 48.0], [t_val, 12.0, 50.0]],
            }
        }

    EXPECTED_RANGES = {
        "2t": {
            "type": "NdArray",
            "dataType": "float",
            "shape": [2],
            "axisNames": ["composite"],
            "values": [264.9, 265.1],
        }
    }

    def test_reforecast_single_hdate_two_points(self):
        """Single hdate with 2 path points -> 1 Trajectory coverage."""
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), TWO_POINTS),
            ]
        )
        covjson = Covjsonkit().encode("CoverageCollection", "Path").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Trajectory"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == self.axes_for("2025-07-14T06:00:00Z")
        assert cov["ranges"] == self.EXPECTED_RANGES
        assert cov["mars:metadata"] == REFORECAST_METADATA_BASE

    def test_reforecast_two_hdates_two_points(self):
        """Two hdates each with 2 path points -> 2 Trajectory coverages."""
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), TWO_POINTS),
                reforecast_branch(np.datetime64("2025-07-15T06:00:00"), TWO_POINTS),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Path").from_polytope_reforecast(tree)

        expected = [
            "2025-07-14T06:00:00Z",
            "2025-07-15T06:00:00Z",
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, fc_date in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == self.axes_for(fc_date)
            assert cov["ranges"] == self.EXPECTED_RANGES
            assert cov["mars:metadata"] == REFORECAST_METADATA_BASE


class TestPathEfasAnoffset:
    """anoffset valid-time correction for class=ce, stream=efas (regular forecast).

    Path embeds the valid-time as the first element of each composite tuple.
    """

    def _t_values(self, cov):
        return [v[0] for v in cov["domain"]["axes"]["composite"]["values"]]

    def test_efas_anoffset_shifts_valid_time(self):
        # 00:00 + step 6h - anoffset 6h = 00:00
        tree = efas_tree([(48.0, 11.0, [12.5])], anoffset=6)
        cov = Covjsonkit().encode("CoverageCollection", "Path").from_polytope(tree)["coverages"][0]
        assert all(t == "2026-01-01T00:00:00Z" for t in self._t_values(cov))

    def test_efas_anoffset_absent_leaves_valid_time_unchanged(self):
        # No anoffset -> 00:00 + step 6h = 06:00
        tree = efas_tree([(48.0, 11.0, [12.5])])
        cov = Covjsonkit().encode("CoverageCollection", "Path").from_polytope(tree)["coverages"][0]
        assert all(t == "2026-01-01T06:00:00Z" for t in self._t_values(cov))

    def test_anoffset_ignored_for_non_efas_stream(self):
        # anoffset present but stream != efas -> valid-time not shifted (06:00)
        tree = efas_tree([(48.0, 11.0, [12.5])], anoffset=6, cls="od", stream="oper")
        cov = Covjsonkit().encode("CoverageCollection", "Path").from_polytope(tree)["coverages"][0]
        assert all(t == "2026-01-01T06:00:00Z" for t in self._t_values(cov))
