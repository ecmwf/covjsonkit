import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

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

EXPECTED_REFORECAST_METADATA = {
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


class TestCircleFromPolytope:
    def test_single_date_single_step_three_points(self):
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
            node("type", ("fc",)),
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [264.9]))
        fc.add_child(make_point(49.0, 11.5, [265.5]))
        fc.add_child(make_point(50.0, 12.0, [266.1]))

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
        fc.add_child(make_point(49.0, 11.5, [265.5]))
        fc.add_child(make_point(50.0, 12.0, [266.1]))

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

        assert cov["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_REFORECAST_METADATA}

    def test_reforecast_two_hdates_three_points(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
        )
        date_node = tip(tree)

        for hdate_val, vals in [
            (np.datetime64("2025-07-14T06:00:00"), [[264.9], [265.5], [266.1]]),
            (np.datetime64("2025-07-15T06:00:00"), [[270.0], [271.0], [272.0]]),
        ]:
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
            t = tip(branch)
            t.add_child(make_point(48.0, 11.0, vals[0]))
            t.add_child(make_point(49.0, 11.5, vals[1]))
            t.add_child(make_point(50.0, 12.0, vals[2]))
            date_node.add_child(branch)

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
            assert cov["mars:metadata"] == {"Forecast date": fc_date, **EXPECTED_REFORECAST_METADATA}
