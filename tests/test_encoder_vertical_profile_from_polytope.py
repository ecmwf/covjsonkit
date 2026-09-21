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
        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "VerticalProfile"

        # Referencing (folded from removed test_referencing)
        ref = covjson["referencing"][0]
        assert ref["coordinates"] == ["latitude", "longitude", "levelist"]
        assert ref["system"]["type"] == "GeographicCRS"

        # Parameters (folded from removed test_parameters_block)
        assert "t" in covjson["parameters"]
        assert covjson["parameters"]["t"]["type"] == "Parameter"
        assert "Temperature" in covjson["parameters"]["t"]["observedProperty"]["label"]["en"]

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [1000, 850, 500]},
            "t": {"values": ["2025-01-01T00:00:00Z"]},
        }

        assert cov["ranges"] == {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["levelist"],
                "values": [290.1, 280.2, 250.3],
            }
        }

        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "pl",
            "step": 0,
            "stream": "oper",
            "type": "an",
            "levelist": 1000,
            "number": 0,
        }

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
        lev_node.add_child(make_point(48.0, 11.0, [290.1, 250.3]))
        lev_node.add_child(make_point(50.0, 13.0, [288.5, 248.7]))

        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope(tree)

        shared_metadata = {
            "class": "od",
            "Forecast date": "2025-06-15T12:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "pl",
            "step": 0,
            "stream": "oper",
            "type": "an",
            "levelist": 1000,
            "number": 0,
        }

        expected = [
            (48.0, 11.0, [290.1, 250.3]),
            (50.0, 13.0, [288.5, 248.7]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, lon, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "latitude": {"values": [lat]},
                "longitude": {"values": [lon]},
                "levelist": {"values": [1000, 500]},
                "t": {"values": ["2025-06-15T12:00:00Z"]},
            }
            assert cov["ranges"] == {
                "t": {
                    "type": "NdArray",
                    "dataType": "float",
                    "shape": [2],
                    "axisNames": ["levelist"],
                    "values": vals,
                }
            }
            assert cov["mars:metadata"] == shared_metadata

    def test_step_offset(self):
        """Step=6 should shift the t coordinate by 6 hours."""
        tree = self._build_vp_tree(step=6, levels_values={1000: 290.0})
        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [1000]},
            "t": {"values": ["2025-01-01T06:00:00Z"]},
        }

        assert cov["ranges"] == {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [1],
                "axisNames": ["levelist"],
                "values": [290.0],
            }
        }

        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "pl",
            "step": 6,
            "stream": "oper",
            "type": "an",
            "levelist": 1000,
            "number": 0,
        }


class TestVerticalProfileFromPolytopeReforecast:
    """Tests for VerticalProfile encoder's from_polytope_reforecast method."""

    REFORECAST_SUFFIX = [
        ("domain", ("g",)),
        ("expver", ("4321",)),
        ("levtype", ("pl",)),
        ("param", ("130",)),
        ("step", (6,)),
        ("stream", ("efcl",)),
        ("type", ("sfo",)),
    ]

    EXPECTED_REFORECAST_METADATA = {
        "class": "ce",
        "date": "2024-03-01",
        "domain": "g",
        "expver": "4321",
        "levtype": "pl",
        "step": 6,
        "stream": "efcl",
        "type": "sfo",
        "levelist": 1000,
        "number": 0,
    }

    def test_reforecast_single_hdate_three_levels(self):
        """1 hdate, 3 pressure levels, 1 point → 1 coverage."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            *[node(n, v) for n, v in self.REFORECAST_SUFFIX],
            node("levelist", (1000, 850, 500)),
            node("latitude", (48.0,)),
            make_leaf(11.0, [290.1, 280.2, 250.3]),
        )
        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "VerticalProfile"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [1000, 850, 500]},
            "t": {"values": ["2025-07-14T12:00:00Z"]},
        }

        assert cov["ranges"] == {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["levelist"],
                "values": [290.1, 280.2, 250.3],
            }
        }

        assert cov["mars:metadata"] == {
            "Forecast date": "2025-07-14T06:00:00Z",
            **self.EXPECTED_REFORECAST_METADATA,
        }

    def test_reforecast_two_hdates_three_levels(self):
        """2 hdates, 3 levels, 1 point → 2 coverages (one per hdate)."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
        )
        date_node = tip(tree)

        for hdate_val, vals in [
            (np.datetime64("2025-07-14T06:00:00"), [290.1, 280.2, 250.3]),
            (np.datetime64("2025-07-15T06:00:00"), [291.0, 281.0, 251.0]),
        ]:
            branch = chain(
                node("hdate", (hdate_val,)),
                *[node(n, v) for n, v in self.REFORECAST_SUFFIX],
                node("levelist", (1000, 850, 500)),
                node("latitude", (48.0,)),
                make_leaf(11.0, vals),
            )
            date_node.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope_reforecast(tree)

        expected = [
            ("2025-07-14T12:00:00Z", [290.1, 280.2, 250.3], "2025-07-14T06:00:00Z"),
            ("2025-07-15T12:00:00Z", [291.0, 281.0, 251.0], "2025-07-15T06:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (t, vals, fc_date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "latitude": {"values": [48.0]},
                "longitude": {"values": [11.0]},
                "levelist": {"values": [1000, 850, 500]},
                "t": {"values": [t]},
            }
            assert cov["ranges"] == {
                "t": {
                    "type": "NdArray",
                    "dataType": "float",
                    "shape": [3],
                    "axisNames": ["levelist"],
                    "values": vals,
                }
            }
            assert cov["mars:metadata"] == {
                "Forecast date": fc_date,
                **self.EXPECTED_REFORECAST_METADATA,
            }
