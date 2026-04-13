import numpy as np
from conftest import chain, make_leaf, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

# Axis ordering for hdate reforecast (between hdate and latitude in the tree)
HDATE_SUFFIX = [
    ("domain", ("g",)),
    ("expver", ("4321",)),
    ("levtype", ("sfc",)),
    ("model", ("lisflood",)),
    ("origin", ("ecmf",)),
    ("param", ("240023",)),
    ("step", (6,)),
    ("stream", ("efcl",)),
    ("type", ("sfo",)),
]


def hdate_branch(hdate, lat, lon, result):
    """hdate → [HDATE_SUFFIX axes] → lat → lon(leaf). Single-point branch."""
    return chain(
        node("hdate", (hdate,)),
        *[node(n, v) for n, v in HDATE_SUFFIX],
        make_point(lat, lon, result),
    )


EXPECTED_HDATE_METADATA = {
    "class": "ce",
    "date": np.datetime64("2024-03-01"),
    "domain": "g",
    "expver": "4321",
    "levtype": "sfc",
    "model": "lisflood",
    "origin": "ecmf",
    "stream": "efcl",
    "type": "sfo",
    "number": 0,
    "levelist": 0,
}


class TestTimeseriesFromPolytope:
    def test_standard_forecast_single_point(self):
        # od/oper/fc/sfc, 1 point, param 167 (2t), steps 0 and 6
        # axis ordering from a real polytope pprint
        leaf = make_leaf(11.0, [264.931, 263.831])

        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0, 6)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
            node("latitude", (48.0,)),
            leaf,
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]},
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["2t"],
                "values": [264.931, 263.831],
            }
        }

        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "number": 0,
            "levelist": 0,
        }

    def test_standard_forecast_multiple_coverages(self):
        # ce/efas/fc/sfc flood forecast: class=ce, stream=efas, type=fc, param=240023 (dis06)
        # 2 dates (time 00:00 and 12:00 pre-merged), 2 steps (6, 30), 2 points
        # → 4 coverages (2 dates × 2 points), each with 2 t-values from steps
        tree = chain(TensorIndexTree(), node("class", ("ce",)))
        cls = tip(tree)

        for date_val, point_vals in [
            (np.datetime64("2026-01-01T00:00:00"), [[12.5, 19.3], [8.7, 14.1]]),
            (np.datetime64("2026-01-01T12:00:00"), [[15.8, 22.6], [10.2, 16.9]]),
        ]:
            branch = chain(
                node("date", (date_val,)),
                node("domain", ("g",)),
                node("expver", ("0001",)),
                node("levtype", ("sfc",)),
                node("model", ("lisflood",)),
                node("origin", ("ecmf",)),
                node("param", ("240023",)),
                node("step", (6, 30)),
                node("stream", ("efas",)),
                node("type", ("fc",)),
            )
            fc = tip(branch)
            fc.add_child(make_point(51.5, 6.5, point_vals[0]))
            fc.add_child(make_point(52.0, 7.0, point_vals[1]))
            cls.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 4

        # point 1, date=2026-01-01T00:00
        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov0["domain"]["axes"]["longitude"]["values"] == [6.5]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2026-01-01T06:00:00Z", "2026-01-02T06:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [12.5, 19.3]

        # point 1, date=2026-01-01T12:00
        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov1["domain"]["axes"]["longitude"]["values"] == [6.5]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2026-01-01T18:00:00Z", "2026-01-02T18:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [15.8, 22.6]

        # point 2, date=2026-01-01T00:00
        cov2 = covjson["coverages"][2]
        assert cov2["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov2["domain"]["axes"]["longitude"]["values"] == [7.0]
        assert cov2["domain"]["axes"]["t"]["values"] == ["2026-01-01T06:00:00Z", "2026-01-02T06:00:00Z"]
        assert cov2["ranges"]["dis06"]["values"] == [8.7, 14.1]

        # point 2, date=2026-01-01T12:00
        cov3 = covjson["coverages"][3]
        assert cov3["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov3["domain"]["axes"]["longitude"]["values"] == [7.0]
        assert cov3["domain"]["axes"]["t"]["values"] == ["2026-01-01T18:00:00Z", "2026-01-02T18:00:00Z"]
        assert cov3["ranges"]["dis06"]["values"] == [10.2, 16.9]

        for cov in covjson["coverages"]:
            assert cov["mars:metadata"]["class"] == "ce"
            assert cov["mars:metadata"]["stream"] == "efas"
            assert cov["mars:metadata"]["type"] == "fc"
            assert cov["mars:metadata"]["model"] == "lisflood"


class TestTimeseriesFromPolytopeReforecast:
    def test_single_point(self):
        # 1 hdate (with time pre-merged by polytope-mars), 1 point
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        # t = hdate(2025-07-14T06:00) + step(6h) = 2025-07-14T12:00:00Z
        assert cov["domain"]["axes"] == {
            "latitude": {"values": [51.5]},
            "longitude": {"values": [6.5]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-07-14T12:00:00Z"]},
        }

        assert cov["ranges"] == {
            "dis06": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [1],
                "axisNames": ["dis06"],
                "values": [42.17],
            }
        }

        assert cov["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_multiple_times(self):
        # 2 hdate values (pre-merged times), 1 point → 2 coverages (one per hdate)
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(hdate_branch(np.datetime64("2025-07-14T12:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17]
        assert cov0["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T18:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [55.30]
        assert cov1["mars:metadata"] == {"Forecast date": "2025-07-14T12:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_multiple_hdates(self):
        # 2 hdates × 1 time (pre-merged), 1 point → 2 coverages (one per hdate)
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(hdate_branch(np.datetime64("2025-07-15T06:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17]
        assert cov0["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-15T12:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [55.30]
        assert cov1["mars:metadata"] == {"Forecast date": "2025-07-15T06:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_multiple_times_and_hdates(self):
        # 2 hdates × 2 times (pre-merged into 4 hdate values), 1 point → 4 coverages
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(hdate_branch(np.datetime64("2025-07-14T12:00:00"), 51.5, 6.5, [55.30]))
        date.add_child(hdate_branch(np.datetime64("2025-07-15T06:00:00"), 51.5, 6.5, [61.44]))
        date.add_child(hdate_branch(np.datetime64("2025-07-15T12:00:00"), 51.5, 6.5, [73.82]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 4

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17]
        assert cov0["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T18:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [55.30]
        assert cov1["mars:metadata"] == {"Forecast date": "2025-07-14T12:00:00Z", **EXPECTED_HDATE_METADATA}

        cov2 = covjson["coverages"][2]
        assert cov2["domain"]["axes"]["t"]["values"] == ["2025-07-15T12:00:00Z"]
        assert cov2["ranges"]["dis06"]["values"] == [61.44]
        assert cov2["mars:metadata"] == {"Forecast date": "2025-07-15T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov3 = covjson["coverages"][3]
        assert cov3["domain"]["axes"]["t"]["values"] == ["2025-07-15T18:00:00Z"]
        assert cov3["ranges"]["dis06"]["values"] == [73.82]
        assert cov3["mars:metadata"] == {"Forecast date": "2025-07-15T12:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_two_points(self):
        # 1 hdate (pre-merged with time), 2 points → 2 coverages (one per point)
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            *[node(n, v) for n, v in HDATE_SUFFIX],
        )
        sfo = tip(tree)
        sfo.add_child(make_point(51.5, 6.5, [42.17]))
        sfo.add_child(make_point(52.0, 7.0, [38.91]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov0["domain"]["axes"]["longitude"]["values"] == [6.5]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov1["domain"]["axes"]["longitude"]["values"] == [7.0]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [38.91]

        assert cov0["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}
        assert cov1["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_two_points_two_times(self):
        # 2 hdate values (= 1 hdate × 2 times pre-merged), 2 points → 4 coverages (point × hdate)
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)

        for hdate_val, vals in [
            (np.datetime64("2025-07-14T06:00:00"), [42.17, 38.91]),
            (np.datetime64("2025-07-14T12:00:00"), [55.30, 49.62]),
        ]:
            branch = chain(
                node("hdate", (hdate_val,)),
                *[node(n, v) for n, v in HDATE_SUFFIX],
            )
            sfo = tip(branch)
            sfo.add_child(make_point(51.5, 6.5, [vals[0]]))
            sfo.add_child(make_point(52.0, 7.0, [vals[1]]))
            date.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 4

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17]
        assert cov0["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T18:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [55.30]
        assert cov1["mars:metadata"] == {"Forecast date": "2025-07-14T12:00:00Z", **EXPECTED_HDATE_METADATA}

        cov2 = covjson["coverages"][2]
        assert cov2["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov2["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov2["ranges"]["dis06"]["values"] == [38.91]
        assert cov2["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov3 = covjson["coverages"][3]
        assert cov3["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov3["domain"]["axes"]["t"]["values"] == ["2025-07-14T18:00:00Z"]
        assert cov3["ranges"]["dis06"]["values"] == [49.62]
        assert cov3["mars:metadata"] == {"Forecast date": "2025-07-14T12:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_multiple_steps(self):
        """Single hdate, two steps (6h, 12h), single point."""
        suffix = [
            ("domain", ("g",)),
            ("expver", ("4321",)),
            ("levtype", ("sfc",)),
            ("model", ("lisflood",)),
            ("origin", ("ecmf",)),
            ("param", ("240023",)),
            ("step", (6, 12)),
            ("stream", ("efcl",)),
            ("type", ("sfo",)),
        ]

        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            *[node(n, v) for n, v in suffix],
            make_point(51.5, 6.5, [42.17, 55.30]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"]["t"]["values"] == [
            "2025-07-14T12:00:00Z",
            "2025-07-14T18:00:00Z",
        ]
        assert cov["ranges"]["dis06"]["values"] == [42.17, 55.30]
        assert cov["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_multiple_params(self):
        """Two parameters in a single hdate subtree."""
        suffix = [
            ("domain", ("g",)),
            ("expver", ("4321",)),
            ("levtype", ("sfc",)),
            ("model", ("lisflood",)),
            ("origin", ("ecmf",)),
            ("param", ("240023", "231002")),
            ("step", (6,)),
            ("stream", ("efcl",)),
            ("type", ("sfo",)),
        ]

        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            *[node(n, v) for n, v in suffix],
            make_point(51.5, 6.5, [42.17, 99.5]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov["ranges"]["dis06"]["values"] == [42.17]
        assert cov["ranges"]["rowe"]["values"] == [99.5]
        assert cov["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

    def test_multiple_hdates_and_steps(self):
        """4 hdates × 2 steps (6, 12) × 1 point → 4 coverages, each with 2 t-values."""
        suffix = [
            ("domain", ("g",)),
            ("expver", ("4321",)),
            ("levtype", ("sfc",)),
            ("model", ("lisflood",)),
            ("origin", ("ecmf",)),
            ("param", ("240023",)),
            ("step", (6, 12)),
            ("stream", ("efcl",)),
            ("type", ("sfo",)),
        ]

        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        for hdate_val, vals in [
            (np.datetime64("2025-07-14T06:00:00"), [10.0, 20.0]),
            (np.datetime64("2025-07-14T12:00:00"), [30.0, 40.0]),
            (np.datetime64("2025-07-15T06:00:00"), [50.0, 60.0]),
            (np.datetime64("2025-07-15T12:00:00"), [70.0, 80.0]),
        ]:
            branch = chain(
                node("hdate", (hdate_val,)),
                *[node(n, v) for n, v in suffix],
                make_point(51.5, 6.5, vals),
            )
            date.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 4

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [10.0, 20.0]
        assert cov0["mars:metadata"] == {"Forecast date": "2025-07-14T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T18:00:00Z", "2025-07-15T00:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [30.0, 40.0]
        assert cov1["mars:metadata"] == {"Forecast date": "2025-07-14T12:00:00Z", **EXPECTED_HDATE_METADATA}

        cov2 = covjson["coverages"][2]
        assert cov2["domain"]["axes"]["t"]["values"] == ["2025-07-15T12:00:00Z", "2025-07-15T18:00:00Z"]
        assert cov2["ranges"]["dis06"]["values"] == [50.0, 60.0]
        assert cov2["mars:metadata"] == {"Forecast date": "2025-07-15T06:00:00Z", **EXPECTED_HDATE_METADATA}

        cov3 = covjson["coverages"][3]
        assert cov3["domain"]["axes"]["t"]["values"] == ["2025-07-15T18:00:00Z", "2025-07-16T00:00:00Z"]
        assert cov3["ranges"]["dis06"]["values"] == [70.0, 80.0]
        assert cov3["mars:metadata"] == {"Forecast date": "2025-07-15T12:00:00Z", **EXPECTED_HDATE_METADATA}
