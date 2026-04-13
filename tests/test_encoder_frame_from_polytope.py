import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


class TestFrameFromPolytope:
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

        covjson = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "MultiPoint"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        # Domain — Frame uses ["x", "y", "z"] coordinates
        composite = cov["domain"]["axes"]["composite"]
        assert composite["dataType"] == "tuple"
        assert composite["coordinates"] == ["x", "y", "z"]
        assert composite["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-01T00:00:00Z"]

        # Ranges
        r = cov["ranges"]["2t"]
        assert r["type"] == "NdArray"
        assert r["dataType"] == "float"
        assert r["shape"] == [2]
        assert r["axisNames"] == ["2t"]
        assert r["values"] == [264.9, 265.1]

        # Metadata
        mm = cov["mars:metadata"]
        assert mm["class"] == "od"
        assert mm["Forecast date"] == "2025-01-01T00:00:00Z"
        assert mm["stream"] == "oper"
        assert mm["type"] == "fc"
        assert mm["number"] == 0
        assert mm["step"] == 0


class TestFrameFromPolytopeReforecast:
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

        covjson = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope_reforecast(tree)

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "MultiPoint"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        composite = cov["domain"]["axes"]["composite"]
        assert composite["dataType"] == "tuple"
        assert composite["coordinates"] == ["x", "y", "z"]
        assert composite["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T06:00:00Z"]

        assert cov["ranges"]["2t"]["values"] == [264.9, 265.1]

        mm = cov["mars:metadata"]
        assert mm["Forecast date"] == "2025-07-14T06:00:00Z"

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

        covjson = Covjsonkit().encode("CoverageCollection", "Frame").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T06:00:00Z"]
        assert cov0["domain"]["axes"]["composite"]["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov0["ranges"]["2t"]["values"] == [264.9, 265.1]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-15T06:00:00Z"]
        assert cov1["domain"]["axes"]["composite"]["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov1["ranges"]["2t"]["values"] == [266.0, 267.0]
