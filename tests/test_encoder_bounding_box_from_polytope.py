import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

COMPOSITE_TWO_POINTS = {
    "dataType": "tuple",
    "coordinates": ["latitude", "longitude", "levelist"],
    "values": [[48.0, 11.0, 0], [50.0, 12.0, 0]],
}

EXPECTED_REFORECAST_METADATA = {
    "class": "ce",
    "date": np.datetime64("2024-03-01"),
    "domain": "g",
    "expver": "4321",
    "levtype": "sfc",
    "stream": "efcl",
    "type": "sfo",
    "number": 0,
}


class TestBoundingBoxFromPolytope:
    def test_single_date_single_step_two_points(self):
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
        fc.add_child(make_point(50.0, 12.0, [265.1]))

        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(tree)

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

    def test_two_dates_two_steps_two_points(self):
        # 2 dates × 2 steps = 4 coverages
        tree = chain(TensorIndexTree(), node("class", ("od",)))
        cls = tip(tree)

        for date_val, vals in [
            (np.datetime64("2025-01-01T00:00:00"), [[264.9, 270.1], [265.1, 271.3]]),
            (np.datetime64("2025-01-02T00:00:00"), [[266.0, 272.0], [267.0, 273.0]]),
        ]:
            branch = chain(
                node("date", (date_val,)),
                node("domain", ("g",)),
                node("expver", ("0001",)),
                node("levtype", ("sfc",)),
                node("param", ("167",)),
                node("step", (0, 6)),
                node("stream", ("oper",)),
                node("type", ("fc",)),
            )
            fc = tip(branch)
            fc.add_child(make_point(48.0, 11.0, vals[0]))
            fc.add_child(make_point(50.0, 12.0, vals[1]))
            cls.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(tree)

        shared_metadata = {
            "class": "od",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "number": 0,
        }

        expected = [
            ("2025-01-01T00:00:00Z", 0, [264.9, 265.1]),
            ("2025-01-01T00:00:00Z", 6, [270.1, 271.3]),
            ("2025-01-02T00:00:00Z", 0, [266.0, 267.0]),
            ("2025-01-02T00:00:00Z", 6, [272.0, 273.0]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (date, step, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "t": {"values": [date]},
                "composite": COMPOSITE_TWO_POINTS,
            }
            assert cov["ranges"]["2t"]["values"] == vals
            assert cov["mars:metadata"] == {**shared_metadata, "Forecast date": date, "step": step}


class TestBoundingBoxFromPolytopeReforecast:
    def test_reforecast_single_hdate_two_points(self):
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

        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope_reforecast(tree)

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
                "axisNames": ["2t"],
                "values": [264.9, 265.1],
            }
        }

        assert cov["mars:metadata"] == {
            **EXPECTED_REFORECAST_METADATA,
            "Forecast date": "2025-07-14T06:00:00Z",
            "step": 0,
        }

    def test_reforecast_two_hdates_two_points(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
        )
        date_node = tip(tree)

        for hdate_val, vals in [
            (np.datetime64("2025-07-14T06:00:00"), [[264.9], [265.1]]),
            (np.datetime64("2025-07-15T06:00:00"), [[266.0], [267.0]]),
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
            t.add_child(make_point(50.0, 12.0, vals[1]))
            date_node.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope_reforecast(tree)

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
            assert cov["mars:metadata"] == {
                **EXPECTED_REFORECAST_METADATA,
                "Forecast date": fc_date,
                "step": 0,
            }

    def test_reforecast_single_hdate_two_steps_two_points(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("4321",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0, 6)),
            node("stream", ("efcl",)),
            node("type", ("sfo",)),
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [264.9, 270.1]))
        fc.add_child(make_point(50.0, 12.0, [265.1, 271.3]))

        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope_reforecast(tree)

        expected = [
            (0, [264.9, 265.1]),
            (6, [270.1, 271.3]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (step, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "t": {"values": ["2025-07-14T06:00:00Z"]},
                "composite": COMPOSITE_TWO_POINTS,
            }
            assert cov["ranges"]["2t"]["values"] == vals
            assert cov["mars:metadata"] == {
                **EXPECTED_REFORECAST_METADATA,
                "Forecast date": "2025-07-14T06:00:00Z",
                "step": step,
            }
