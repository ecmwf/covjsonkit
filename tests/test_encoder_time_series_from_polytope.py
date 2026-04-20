import json

import numpy as np
import orjson
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
    "date": "2024-03-01",
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
        # ce/efas/fc/sfc flood forecast: 2 dates × 2 steps × 2 points → 4 coverages
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

        shared_metadata = {
            "class": "ce",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "model": "lisflood",
            "origin": "ecmf",
            "stream": "efas",
            "type": "fc",
            "number": 0,
            "levelist": 0,
        }

        expected = [
            (51.5, 6.5, ["2026-01-01T06:00:00Z", "2026-01-02T06:00:00Z"], [12.5, 19.3], "2026-01-01T00:00:00Z"),
            (51.5, 6.5, ["2026-01-01T18:00:00Z", "2026-01-02T18:00:00Z"], [15.8, 22.6], "2026-01-01T12:00:00Z"),
            (52.0, 7.0, ["2026-01-01T06:00:00Z", "2026-01-02T06:00:00Z"], [8.7, 14.1], "2026-01-01T00:00:00Z"),
            (52.0, 7.0, ["2026-01-01T18:00:00Z", "2026-01-02T18:00:00Z"], [10.2, 16.9], "2026-01-01T12:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, lon, t, vals, date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "latitude": {"values": [lat]},
                "longitude": {"values": [lon]},
                "levelist": {"values": [0]},
                "t": {"values": t},
            }
            assert cov["ranges"]["dis06"]["values"] == vals
            assert cov["mars:metadata"] == {**shared_metadata, "Forecast date": date}

    def test_multiple_params(self):
        # 1 date, 2 params (167 = 2t, 168 = 2d), 1 step, 1 point → 1 coverage with both params
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167", "168")),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
            make_point(48.0, 11.0, [264.9, 250.1]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-01-01T00:00:00Z"]},
        }
        assert cov["ranges"] == {
            "2t": {"type": "NdArray", "dataType": "float", "shape": [1], "axisNames": ["2t"], "values": [264.9]},
            "2d": {"type": "NdArray", "dataType": "float", "shape": [1], "axisNames": ["2d"], "values": [250.1]},
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
        # 2 hdate values (pre-merged times from same day), 1 point → 2 coverages
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(hdate_branch(np.datetime64("2025-07-14T12:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        expected = [
            (["2025-07-14T12:00:00Z"], [42.17], "2025-07-14T06:00:00Z"),
            (["2025-07-14T18:00:00Z"], [55.30], "2025-07-14T12:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (t, vals, fc_date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"]["t"]["values"] == t
            assert cov["ranges"]["dis06"]["values"] == vals
            assert cov["mars:metadata"] == {"Forecast date": fc_date, **EXPECTED_HDATE_METADATA}

    def test_reforecast_covjson_is_json_serialisable(self):
        """CoverageJSON produced by from_polytope_reforecast must be serialisable
        with both stdlib json and orjson (i.e. no numpy.datetime64 left in the
        mars:metadata dict — regression test for the date axis leak)."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        # stdlib json.dumps must not raise TypeError
        serialised = json.dumps(covjson)
        assert serialised  # non-empty string

        # orjson.dumps must also succeed
        orjson.dumps(covjson)

        # The 'date' value in mars:metadata must be a plain string, not numpy.datetime64
        for cov in covjson["coverages"]:
            date_val = cov["mars:metadata"].get("date")
            assert not isinstance(
                date_val, np.datetime64
            ), f"mars:metadata['date'] is still a numpy.datetime64: {date_val!r}"

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

        expected = [
            (["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"], [10.0, 20.0], "2025-07-14T06:00:00Z"),
            (["2025-07-14T18:00:00Z", "2025-07-15T00:00:00Z"], [30.0, 40.0], "2025-07-14T12:00:00Z"),
            (["2025-07-15T12:00:00Z", "2025-07-15T18:00:00Z"], [50.0, 60.0], "2025-07-15T06:00:00Z"),
            (["2025-07-15T18:00:00Z", "2025-07-16T00:00:00Z"], [70.0, 80.0], "2025-07-15T12:00:00Z"),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (t, vals, fc_date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"]["t"]["values"] == t
            assert cov["ranges"]["dis06"]["values"] == vals
            assert cov["mars:metadata"] == {"Forecast date": fc_date, **EXPECTED_HDATE_METADATA}
