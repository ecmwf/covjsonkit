import json

import numpy as np
import orjson
import pytest
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
            "x": {"values": [11.0]},
            "y": {"values": [48.0]},
            "t": {"values": ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]},
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["t"],
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

    def test_standard_forecast_multiple_coverages(self):
        # ce/efas/fc/sfc flood forecast: 2 dates × 2 steps × 2 points.
        # With >1 point the PointSeries request auto-upgrades to MultiPointSeries:
        # one coverage per date, both points on a shared composite (x/y) axis.
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

        assert covjson["domainType"] == "MultiPointSeries"

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

        # One coverage per date; composite = [[lon, lat], ...]; ranges row-major [t, composite].
        expected = [
            (
                ["2026-01-01T06:00:00Z", "2026-01-02T06:00:00Z"],
                [12.5, 8.7, 19.3, 14.1],
                "2026-01-01T00:00:00Z",
            ),
            (
                ["2026-01-01T18:00:00Z", "2026-01-02T18:00:00Z"],
                [15.8, 10.2, 22.6, 16.9],
                "2026-01-01T12:00:00Z",
            ),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (t, vals, date) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"]["t"] == {"values": t}
            assert cov["domain"]["axes"]["composite"] == {
                "dataType": "tuple",
                "coordinates": ["x", "y"],
                "values": [[6.5, 51.5], [7.0, 52.0]],
            }
            assert cov["ranges"]["dis06"] == {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2, 2],
                "axisNames": ["t", "composite"],
                "values": vals,
            }
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
            "x": {"values": [11.0]},
            "y": {"values": [48.0]},
            "t": {"values": ["2025-01-01T00:00:00Z"]},
        }
        assert cov["ranges"] == {
            "2t": {"type": "NdArray", "dataType": "float", "shape": [1], "axisNames": ["t"], "values": [264.9]},
            "2d": {"type": "NdArray", "dataType": "float", "shape": [1], "axisNames": ["t"], "values": [250.1]},
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

    def test_real_level_single_point(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("pl",)),
            node("levelist", (850,)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
            make_point(48.0, 11.0, [264.931]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "x": {"values": [11.0]},
            "y": {"values": [48.0]},
            "z": {"values": [850]},
            "t": {"values": ["2025-01-01T00:00:00Z"]},
        }
        assert covjson["referencing"][-1] == {
            "coordinates": ["z"],
            "system": {"type": "VerticalCRS"},
        }
        assert cov["ranges"]["2t"]["axisNames"] == ["t"]
        assert cov["mars:metadata"]["levelist"] == 850


class TestTimeseriesFromPolytopeReforecast:
    @pytest.mark.parametrize("date", [np.datetime64("2024-03-01"), np.datetime64("2024-03-01T00:00:00")])
    def test_single_point(self, date):
        # 1 hdate (with time pre-merged by polytope-mars), 1 point
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (date,)),
            hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        # t = hdate(2025-07-14T06:00) + step(6h) = 2025-07-14T12:00:00Z
        assert cov["domain"]["axes"] == {
            "x": {"values": [6.5]},
            "y": {"values": [51.5]},
            "t": {"values": ["2025-07-14T12:00:00Z"]},
        }

        assert cov["ranges"] == {
            "dis06": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [1],
                "axisNames": ["t"],
                "values": [42.17],
            }
        }

        # Reanalysis collapse path omits the scalar "Forecast date".
        assert cov["mars:metadata"] == {
            **EXPECTED_HDATE_METADATA,
            "date": date.astype(str),
        }

    def test_multiple_times(self):
        # 2 hdate values (pre-merged times from same day), 1 point
        # → collapsed into 1 coverage with both valid-times on the t-axis.
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(hdate_branch(np.datetime64("2025-07-14T12:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"]
        assert cov["ranges"]["dis06"]["values"] == [42.17, 55.30]
        assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

    def test_multiple_hdates(self):
        # 2 hdates (different days), 1 point
        # → collapsed into 1 coverage with both valid-times on the t-axis.
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(hdate_branch(np.datetime64("2025-07-15T06:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z", "2025-07-15T12:00:00Z"]
        assert cov["ranges"]["dis06"]["values"] == [42.17, 55.30]
        assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

    def test_two_points(self):
        # 1 hdate, 2 points → 2 coverages (one per point)
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

        expected = [
            (51.5, 6.5, [42.17]),
            (52.0, 7.0, [38.91]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, lon, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"] == {
                "x": {"values": [lon]},
                "y": {"values": [lat]},
                "t": {"values": ["2025-07-14T12:00:00Z"]},
            }
            assert cov["ranges"]["dis06"]["values"] == vals
            assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

    def test_two_points_two_times(self):
        # 2 hdate values × 2 points → 2 coverages (one per point, hdates collapsed)
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

        expected = [
            (51.5, ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"], [42.17, 55.30]),
            (52.0, ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"], [38.91, 49.62]),
        ]
        assert len(covjson["coverages"]) == len(expected)
        for cov, (lat, t, vals) in zip(covjson["coverages"], expected):
            assert cov["domain"]["axes"]["y"]["values"] == [lat]
            assert cov["domain"]["axes"]["t"]["values"] == t
            assert cov["ranges"]["dis06"]["values"] == vals
            assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

    def test_multiple_steps(self):
        """Single hdate, two steps (6h, 12h), single point → 1 coverage with 2 t-values."""
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

        assert cov["domain"]["axes"] == {
            "x": {"values": [6.5]},
            "y": {"values": [51.5]},
            "t": {"values": ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"]},
        }
        assert cov["ranges"]["dis06"]["values"] == [42.17, 55.30]
        assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

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

        assert cov["domain"]["axes"] == {
            "x": {"values": [6.5]},
            "y": {"values": [51.5]},
            "t": {"values": ["2025-07-14T12:00:00Z"]},
        }
        assert cov["ranges"] == {
            "dis06": {"type": "NdArray", "dataType": "float", "shape": [1], "axisNames": ["t"], "values": [42.17]},
            "rowe": {"type": "NdArray", "dataType": "float", "shape": [1], "axisNames": ["t"], "values": [99.5]},
        }
        assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

    def test_multiple_hdates_and_steps(self):
        """4 hdates × 2 steps (6, 12) × 1 point → 1 collapsed coverage."""
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

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]
        # All hdate+step valid-times collapsed onto one t-axis, sorted chronologically
        # (ties preserve hdate insertion order), values concatenated in the same order.
        assert cov["domain"]["axes"]["t"]["values"] == [
            "2025-07-14T12:00:00Z",
            "2025-07-14T18:00:00Z",
            "2025-07-14T18:00:00Z",
            "2025-07-15T00:00:00Z",
            "2025-07-15T12:00:00Z",
            "2025-07-15T18:00:00Z",
            "2025-07-15T18:00:00Z",
            "2025-07-16T00:00:00Z",
        ]
        assert cov["ranges"]["dis06"]["values"] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        assert cov["mars:metadata"] == EXPECTED_HDATE_METADATA

    @pytest.mark.parametrize(
        "json_module,date", [(json, np.datetime64("2024-03-01")), (orjson, np.datetime64("2024-03-01T00:00:00"))]
    )
    def test_reforecast_covjson_is_json_serialisable(self, json_module, date):
        """CoverageJSON produced by from_polytope_reforecast must be serialisable
        with both stdlib json and orjson (i.e. no numpy.datetime64 left in the
        mars:metadata dict — regression test for the date axis leak)."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (date,)),
            hdate_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        # stdlib json.dumps must not raise TypeError
        serialised = json_module.dumps(covjson)

        # The 'date' value in mars:metadata must be a plain string, not numpy.datetime64
        for cov in covjson["coverages"]:
            date_val = cov["mars:metadata"].get("date")
            assert not isinstance(
                date_val, np.datetime64
            ), f"mars:metadata['date'] is still a numpy.datetime64: {date_val!r}"

        # Round-trip through the JSON module to ensure it can be deserialised back to a dict
        deserialised = json_module.loads(serialised)
        assert deserialised == covjson


class TestTimeseriesFromPolytopeMonthly:
    def test_monthly_surface_x_y_t(self):
        """from_polytope_month: one year, two months, surface → x/y/t axes, no z, two references."""
        tree = chain(
            TensorIndexTree(),
            node("class", ("d1",)),
            node("stream", ("clmn",)),
            node("levtype", ("sfc",)),
            node("number", (0,)),
            node("year", (2020,)),
            node("month", (2, 3)),
            node("param", ("167",)),
            make_point(48.0, 11.0, [300.0, 301.0]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_month(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert set(cov["domain"]["axes"].keys()) == {"x", "y", "t"}
        assert cov["domain"]["axes"]["x"] == {"values": [11.0]}
        assert cov["domain"]["axes"]["y"] == {"values": [48.0]}
        assert "z" not in cov["domain"]["axes"]
        assert cov["domain"]["axes"]["t"]["values"] == [
            "2020-02-01T00:00:00Z",
            "2020-03-01T00:00:00Z",
        ]

        assert cov["ranges"]["2t"]["axisNames"] == ["t"]
        assert cov["ranges"]["2t"]["shape"] == [2]

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


class TestTimeseriesFromPolytopeStep:
    def test_step_surface_x_y_t(self):
        """from_polytope_step: one date, two time offsets, surface → x/y/t axes, no z, two references."""
        from datetime import timedelta

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
            node("time", (timedelta(0), timedelta(hours=6))),
            make_point(48.0, 11.0, [264.0, 263.0]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_step(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert set(cov["domain"]["axes"].keys()) == {"x", "y", "t"}
        assert cov["domain"]["axes"]["x"] == {"values": [11.0]}
        assert cov["domain"]["axes"]["y"] == {"values": [48.0]}
        assert "z" not in cov["domain"]["axes"]
        assert len(cov["domain"]["axes"]["t"]["values"]) == 2

        assert cov["ranges"]["2t"]["axisNames"] == ["t"]
        assert cov["ranges"]["2t"]["shape"] == [2]

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

    def test_non_ce_efcl_hdate_not_collapsed(self):
        """The collapse is gated to class=ce + stream=efcl. Other reanalysis-style
        hdate requests must keep the legacy one-coverage-per-hdate output."""
        suffix = [
            ("domain", ("g",)),
            ("expver", ("4321",)),
            ("levtype", ("sfc",)),
            ("model", ("lisflood",)),
            ("origin", ("ecmf",)),
            ("param", ("240023",)),
            ("step", (6,)),
            ("stream", ("enfh",)),  # not efcl → gate must not trigger
            ("type", ("sfo",)),
        ]
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        for hdate_val, val in [
            (np.datetime64("2025-07-14T06:00:00"), [42.17]),
            (np.datetime64("2025-07-15T06:00:00"), [55.30]),
        ]:
            branch = chain(
                node("hdate", (hdate_val,)),
                *[node(n, v) for n, v in suffix],
                make_point(51.5, 6.5, val),
            )
            date.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_reforecast(tree)

        # Legacy behaviour: one coverage per hdate, each with a scalar "Forecast date".
        assert len(covjson["coverages"]) == 2
        for cov in covjson["coverages"]:
            assert "Forecast date" in cov["mars:metadata"]
            assert len(cov["domain"]["axes"]["t"]["values"]) == 1
