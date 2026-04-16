import numpy as np
from conftest import (
    REFORECAST_METADATA_BASE,
    chain,
    make_point,
    node,
    reforecast_branch,
    reforecast_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

GRID_2X2_AXES = {
    "t": {"values": [0]},
    "latitude": {"values": [48.0, 50.0]},
    "longitude": {"values": [11.0, 12.0]},
    "levelist": {"values": [0]},
}

GRID_2X2_RANGES = {
    "2t": {
        "type": "NdArray",
        "dataType": "float",
        "shape": [1, 1, 2, 2],
        "axisNames": ["t", "levelist", "latitude", "longitude"],
        "values": [264.9, 265.1, 266.3, 267.5],
    }
}

GRID_2X2_POINTS = [
    (48.0, 11.0, [264.9]),
    (48.0, 12.0, [265.1]),
    (50.0, 11.0, [266.3]),
    (50.0, 12.0, [267.5]),
]


class TestGridFromPolytope:
    """Tests for Grid encoder's from_polytope method."""

    def test_2x2_grid(self):
        """2x2 grid: 2 latitudes, 2 longitudes, param 167 (2t), step 0."""
        # Grid uses type="an" (analysis), not type="fc" (forecast),
        # so we build the tree inline instead of using forecast_tree().
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("an",)),
        )
        parent = tip(tree)
        for lat, lon, vals in GRID_2X2_POINTS:
            parent.add_child(make_point(lat, lon, vals))

        covjson = Covjsonkit().encode("CoverageCollection", "Grid").from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Grid"

        # Collection-level referencing
        assert covjson["referencing"] == [
            {
                "coordinates": ["latitude", "longitude", "levelist"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        ]

        # Collection-level parameters
        assert "2t" in covjson["parameters"]
        assert covjson["parameters"]["2t"]["type"] == "Parameter"
        assert covjson["parameters"]["2t"]["observedProperty"] == {
            "id": "2t",
            "label": {"en": "2 metre temperature"},
        }

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == GRID_2X2_AXES
        assert cov["ranges"] == GRID_2X2_RANGES
        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "step": 0,
            "stream": "oper",
            "type": "an",
            "number": 0,
        }

    def test_1x1_grid(self):
        """Edge case: single-point grid -> shape [1,1,1,1]."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("an",)),
            make_point(48.0, 11.0, [264.9]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Grid").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "t": {"values": [0]},
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [0]},
        }
        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [1, 1, 1, 1],
                "axisNames": ["t", "levelist", "latitude", "longitude"],
                "values": [264.9],
            }
        }


class TestGridFromPolytopeReforecast:
    """Tests for Grid encoder's from_polytope_reforecast method."""

    def test_reforecast_single_hdate_2x2_grid(self):
        """Single hdate with 2x2 grid -> 1 Grid coverage."""
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), GRID_2X2_POINTS),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Grid").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "Grid"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == GRID_2X2_AXES
        assert cov["ranges"] == GRID_2X2_RANGES
        assert cov["mars:metadata"] == {
            **REFORECAST_METADATA_BASE,
            "Forecast date": "2025-07-14T06:00:00Z",
        }

    def test_reforecast_two_hdates_2x2_grid(self):
        """Two hdates each with 2x2 grid -> 2 Grid coverages."""
        tree = reforecast_tree(
            [
                reforecast_branch(np.datetime64("2025-07-14T06:00:00"), GRID_2X2_POINTS),
                reforecast_branch(np.datetime64("2025-07-15T06:00:00"), GRID_2X2_POINTS),
            ]
        )

        covjson = Covjsonkit().encode("CoverageCollection", "Grid").from_polytope_reforecast(tree)

        expected = [
            "2025-07-14T06:00:00Z",
            "2025-07-15T06:00:00Z",
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, fc_date in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == GRID_2X2_AXES
            assert cov["ranges"] == GRID_2X2_RANGES
            assert cov["mars:metadata"] == {
                **REFORECAST_METADATA_BASE,
                "Forecast date": fc_date,
            }
