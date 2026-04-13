import numpy as np
from conftest import chain, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


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

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "MultiPoint"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        # Domain
        composite = cov["domain"]["axes"]["composite"]
        assert composite["dataType"] == "tuple"
        assert composite["coordinates"] == ["latitude", "longitude", "levelist"]
        assert composite["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-01T00:00:00Z"]

        # Ranges
        assert "2t" in cov["ranges"]
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
        assert mm["domain"] == "g"
        assert mm["expver"] == "0001"
        assert mm["levtype"] == "sfc"
        assert mm["stream"] == "oper"
        assert mm["type"] == "fc"
        assert mm["number"] == 0
        assert mm["step"] == 0

    def test_two_dates_two_steps_two_points(self):
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

        # 2 dates × 2 steps = 4 coverages
        assert len(covjson["coverages"]) == 4

        # Collect coverages keyed by (date, step)
        by_key = {}
        for cov in covjson["coverages"]:
            mm = cov["mars:metadata"]
            by_key[(mm["Forecast date"], mm["step"])] = cov

        # Date 1, step 0
        cov = by_key[("2025-01-01T00:00:00Z", 0)]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-01T00:00:00Z"]
        assert cov["domain"]["axes"]["composite"]["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov["ranges"]["2t"]["values"] == [264.9, 265.1]

        # Date 1, step 6
        cov = by_key[("2025-01-01T00:00:00Z", 6)]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-01T00:00:00Z"]
        assert cov["ranges"]["2t"]["values"] == [270.1, 271.3]

        # Date 2, step 0
        cov = by_key[("2025-01-02T00:00:00Z", 0)]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-02T00:00:00Z"]
        assert cov["ranges"]["2t"]["values"] == [266.0, 267.0]

        # Date 2, step 6
        cov = by_key[("2025-01-02T00:00:00Z", 6)]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-01-02T00:00:00Z"]
        assert cov["ranges"]["2t"]["values"] == [272.0, 273.0]


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

        assert covjson["type"] == "CoverageCollection"
        assert covjson["domainType"] == "MultiPoint"
        assert len(covjson["coverages"]) == 1

        cov = covjson["coverages"][0]
        assert cov["type"] == "Coverage"

        composite = cov["domain"]["axes"]["composite"]
        assert composite["dataType"] == "tuple"
        assert composite["coordinates"] == ["latitude", "longitude", "levelist"]
        assert composite["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T06:00:00Z"]

        mm = cov["mars:metadata"]
        assert mm["Forecast date"] == "2025-07-14T06:00:00Z"

        assert cov["ranges"]["2t"]["values"] == [264.9, 265.1]

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

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T06:00:00Z"]
        assert cov0["domain"]["axes"]["composite"]["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov0["ranges"]["2t"]["values"] == [264.9, 265.1]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-15T06:00:00Z"]
        assert cov1["domain"]["axes"]["composite"]["values"] == [[48.0, 11.0, 0], [50.0, 12.0, 0]]
        assert cov1["ranges"]["2t"]["values"] == [266.0, 267.0]

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

        assert len(covjson["coverages"]) == 2

        by_step = {}
        for cov in covjson["coverages"]:
            by_step[cov["mars:metadata"]["step"]] = cov

        cov0 = by_step[0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T06:00:00Z"]
        assert cov0["ranges"]["2t"]["values"] == [264.9, 265.1]
        assert cov0["mars:metadata"]["Forecast date"] == "2025-07-14T06:00:00Z"

        cov6 = by_step[6]
        assert cov6["domain"]["axes"]["t"]["values"] == ["2025-07-14T06:00:00Z"]
        assert cov6["ranges"]["2t"]["values"] == [270.1, 271.3]
        assert cov6["mars:metadata"]["Forecast date"] == "2025-07-14T06:00:00Z"
