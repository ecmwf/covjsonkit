import json
import os
from datetime import timedelta

import numpy as np
import pytest
from conftest import (
    assert_valid_covjson,
    chain,
    make_point,
    node,
    reforecast_separate_datetime_vertical_tree,
    reforecast_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


def _surface_tree(points, steps=(0, 6), param="167", date="2025-01-01T00:00:00"):
    """Single-date surface forecast tree with the given points.

    ``points`` is a list of ``(lat, lon, values)`` where ``values`` has one entry
    per step.
    """
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64(date),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("sfc",)),
        node("param", (param,)),
        node("step", tuple(steps)),
        node("stream", ("oper",)),
        node("type", ("fc",)),
    )
    fc = tip(tree)
    for lat, lon, vals in points:
        fc.add_child(make_point(lat, lon, vals))
    return tree


class TestMultiPointSeriesEncoder:
    def test_auto_switch_single_point_stays_pointseries(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        assert cj["domainType"] == "PointSeries"
        assert "composite" not in cj["coverages"][0]["domain"]["axes"]

    def test_auto_switch_multi_point_becomes_multipointseries(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        assert cj["domainType"] == "MultiPointSeries"
        assert len(cj["coverages"]) == 1
        axes = cj["coverages"][0]["domain"]["axes"]
        assert axes["t"] == {"values": ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]}
        assert axes["composite"] == {
            "dataType": "tuple",
            "coordinates": ["x", "y"],
            "values": [[11.0, 48.0], [12.0, 50.0]],
        }
        assert cj["coverages"][0]["ranges"]["2t"] == {
            "type": "NdArray",
            "dataType": "float",
            "shape": [2, 2],
            "axisNames": ["t", "composite"],
            # row-major [t, composite]: t0=(p0,p1), t1=(p0,p1)
            "values": [264.9, 274.9, 263.8, 273.8],
        }

    def test_forced_single_point_multipointseries(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8])])
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        assert cj["domainType"] == "MultiPointSeries"
        axes = cj["coverages"][0]["domain"]["axes"]
        assert axes["composite"]["values"] == [[11.0, 48.0]]
        assert cj["coverages"][0]["ranges"]["2t"]["shape"] == [2, 1]

    def test_lowercase_alias(self):
        tree = _surface_tree([(48.0, 11.0, [264.9])], steps=(0,))
        cj = Covjsonkit().encode("CoverageCollection", "multipointseries").from_polytope(tree)
        assert cj["domainType"] == "MultiPointSeries"

    def test_multi_param(self):
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
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [264.9, 250.1]))
        fc.add_child(make_point(50.0, 12.0, [274.9, 260.1]))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        cov = cj["coverages"][0]
        assert cov["ranges"]["2t"]["values"] == [264.9, 274.9]
        assert cov["ranges"]["2d"]["values"] == [250.1, 260.1]
        assert cov["ranges"]["2t"]["shape"] == [1, 2]

    def test_level_z_axis(self):
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
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [264.9]))
        fc.add_child(make_point(50.0, 12.0, [274.9]))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        axes = cj["coverages"][0]["domain"]["axes"]
        # OGC 21-069r2 §9.10.5: only ``composite`` and ``t``; z lives in the tuple.
        assert set(axes) == {"t", "composite"}
        assert axes["composite"] == {
            "dataType": "tuple",
            "coordinates": ["x", "y", "z"],
            "values": [[11.0, 48.0, 850], [12.0, 50.0, 850]],
        }
        # Levels live in the composite z component, not in per-coverage metadata.
        assert "levelist" not in cj["coverages"][0]["mars:metadata"]
        assert cj["coverages"][0]["ranges"]["2t"]["axisNames"] == ["t", "composite"]
        assert cj["coverages"][0]["ranges"]["2t"]["shape"] == [1, 2]

    def test_multi_number_one_coverage_each(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("number", (0, 1)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("enfo",)),
            node("type", ("pf",)),
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [264.9, 265.9]))
        fc.add_child(make_point(50.0, 12.0, [274.9, 275.9]))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        # One coverage per number, each with the same composite axis.
        assert len(cj["coverages"]) == 2
        numbers = sorted(c["mars:metadata"]["number"] for c in cj["coverages"])
        assert numbers == [0, 1]
        for cov in cj["coverages"]:
            assert cov["domain"]["axes"]["composite"]["values"] == [[11.0, 48.0], [12.0, 50.0]]


class TestMultiPointSeriesDecoder:
    def test_to_xarray_single_dataset(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])])
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        ds = Covjsonkit().decode(cj).to_xarray()

        assert dict(ds.sizes) == {"t": 2, "points": 2}
        assert list(ds["latitude"].values) == [48.0, 50.0]
        assert list(ds["longitude"].values) == [11.0, 12.0]
        # (t, points)
        assert ds["2t"].values.tolist() == [[264.9, 274.9], [263.8, 273.8]]

    def test_to_geojson_feature_per_point_time(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])])
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        gj = Covjsonkit().decode(cj).to_geojson()
        assert len(gj["features"]) == 4  # 2 points × 2 times
        f0 = gj["features"][0]
        assert f0["geometry"]["coordinates"] == [11.0, 48.0]
        assert f0["properties"]["2t"] == 264.9
        assert f0["properties"]["datetime"] == "2025-01-01T00:00:00Z"

    def test_roundtrip_matches_pointseries(self):
        # The same data encoded as forced-single-point PointSeries vs auto MultiPointSeries
        # must decode to the same per-point values.
        pts = [(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])]
        multi = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(_surface_tree(pts))
        ds = Covjsonkit().decode(multi).to_xarray()
        # point 0 series over t
        assert ds["2t"].sel(points=0).values.tolist() == [264.9, 263.8]
        assert ds["2t"].sel(points=1).values.tolist() == [274.9, 273.8]


def _month_tree(points, year=2020, months=(2, 3), param="167"):
    """Monthly-mean (clmn) tree; ``points`` is ``(lat, lon, values-per-month)``."""
    tree = chain(
        TensorIndexTree(),
        node("class", ("d1",)),
        node("stream", ("clmn",)),
        node("levtype", ("sfc",)),
        node("number", (0,)),
        node("year", (year,)),
        node("month", tuple(months)),
        node("param", (param,)),
    )
    leaf = tip(tree)
    for lat, lon, vals in points:
        leaf.add_child(make_point(lat, lon, vals))
    return tree


def _step_tree(points, date="2025-01-01T00:00:00", param="167"):
    """Step (time-offset) tree; ``points`` is ``(lat, lon, values-per-time)``."""
    from datetime import timedelta

    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64(date),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("sfc",)),
        node("param", (param,)),
        node("step", (0,)),
        node("stream", ("oper",)),
        node("type", ("fc",)),
        node("time", (timedelta(0), timedelta(hours=6))),
    )
    leaf = tip(tree)
    for lat, lon, vals in points:
        leaf.add_child(make_point(lat, lon, vals))
    return tree


class TestMultiPointSeriesMonth:
    def test_month_multi_point_becomes_multipointseries(self):
        tree = _month_tree([(48.0, 11.0, [300.0, 301.0]), (50.0, 12.0, [310.0, 311.0])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_month(tree)

        assert cj["domainType"] == "MultiPointSeries"
        assert len(cj["coverages"]) == 1
        cov = cj["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == [
            "2020-02-01T00:00:00Z",
            "2020-03-01T00:00:00Z",
        ]
        assert cov["domain"]["axes"]["composite"]["values"] == [[11.0, 48.0], [12.0, 50.0]]
        assert cov["ranges"]["2t"]["shape"] == [2, 2]
        # Row-major [t, composite]: month0 → p0,p1 ; month1 → p0,p1.
        assert cov["ranges"]["2t"]["values"] == [300.0, 310.0, 301.0, 311.0]

    def test_month_single_point_stays_pointseries(self):
        tree = _month_tree([(48.0, 11.0, [300.0, 301.0])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_month(tree)
        assert cj["domainType"] == "PointSeries"
        assert "composite" not in cj["coverages"][0]["domain"]["axes"]

    def test_month_roundtrip_to_xarray(self):
        tree = _month_tree([(48.0, 11.0, [300.0, 301.0]), (50.0, 12.0, [310.0, 311.0])])
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_month(tree)
        ds = Covjsonkit().decode(cj).to_xarray()
        assert dict(ds.sizes) == {"t": 2, "points": 2}
        assert ds["2t"].sel(points=0).values.tolist() == [300.0, 301.0]
        assert ds["2t"].sel(points=1).values.tolist() == [310.0, 311.0]


class TestMultiPointSeriesStep:
    def test_step_multi_point_becomes_multipointseries(self):
        tree = _step_tree([(48.0, 11.0, [264.0, 263.0]), (50.0, 12.0, [274.0, 273.0])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_step(tree)

        assert cj["domainType"] == "MultiPointSeries"
        assert len(cj["coverages"]) == 1
        cov = cj["coverages"][0]
        assert len(cov["domain"]["axes"]["t"]["values"]) == 2
        assert cov["domain"]["axes"]["composite"]["values"] == [[11.0, 48.0], [12.0, 50.0]]
        assert cov["ranges"]["2t"]["shape"] == [2, 2]
        # Row-major [t, composite]: time0 → p0,p1 ; time1 → p0,p1.
        assert cov["ranges"]["2t"]["values"] == [264.0, 274.0, 263.0, 273.0]

    def test_step_single_point_stays_pointseries(self):
        tree = _step_tree([(48.0, 11.0, [264.0, 263.0])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_step(tree)
        assert cj["domainType"] == "PointSeries"
        assert "composite" not in cj["coverages"][0]["domain"]["axes"]

    def test_step_roundtrip_to_xarray(self):
        tree = _step_tree([(48.0, 11.0, [264.0, 263.0]), (50.0, 12.0, [274.0, 273.0])])
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_step(tree)
        ds = Covjsonkit().decode(cj).to_xarray()
        assert dict(ds.sizes) == {"t": 2, "points": 2}
        assert ds["2t"].sel(points=0).values.tolist() == [264.0, 263.0]
        assert ds["2t"].sel(points=1).values.tolist() == [274.0, 273.0]


# --- vertical (z) handling: composite tuples [lon, lat, level] ----------------


def _assert_spec_z(cj, levels, n_points=2, n_coverages=1):
    """MultiPointSeries with levels: ONE coverage holds all levels.

    Only t + composite axes, composite coordinates x/y/z, entries point-major then
    level, no per-coverage ``levelist`` metadata, range shape [n_t, n_points*n_levels].
    """
    assert cj["domainType"] == "MultiPointSeries"
    assert len(cj["coverages"]) == n_coverages
    for cov in cj["coverages"]:
        axes = cov["domain"]["axes"]
        assert set(axes) == {"t", "composite"}
        comp = axes["composite"]
        assert comp["dataType"] == "tuple"
        assert comp["coordinates"] == ["x", "y", "z"]
        assert len(comp["values"]) == n_points * len(levels)
        assert [pt[2] for pt in comp["values"]] == list(levels) * n_points  # level innermost
        assert all(len(pt) == 3 for pt in comp["values"])
        assert "levelist" not in cov["mars:metadata"]
        for rng in cov["ranges"].values():
            assert rng["axisNames"] == ["t", "composite"]
            assert rng["shape"] == [len(axes["t"]["values"]), n_points * len(levels)]
            assert len(rng["values"]) == rng["shape"][0] * rng["shape"][1]
    # z still referenced (vertical CRS) alongside the x/y CRS.
    assert {"coordinates": ["z"], "system": {"type": "VerticalCRS"}} in cj["referencing"]
    assert_valid_covjson(cj)
    _assert_official_schema(cj)


def _assert_official_schema(cj):
    """Validate against the official CoverageJSON schema when it is available offline.

    Set ``COVJSON_SCHEMA`` to a draft-07 CoverageJSON schema file. Unset: not checked.
    Set but nonexistent: the test FAILS (a typo must not silently disable the check).
    The schema is deliberately not vendored into the repository.
    """
    path = os.environ.get("COVJSON_SCHEMA")
    if not path:
        return
    assert os.path.isfile(path), f"COVJSON_SCHEMA={path!r} does not exist"
    jsonschema = pytest.importorskip("jsonschema")
    with open(path) as fh:
        schema = json.load(fh)
    doc = json.loads(json.dumps(cj))
    for cov in doc["coverages"]:
        cov["domain"].setdefault("domainType", doc["domainType"])
    errors = list(jsonschema.Draft7Validator(schema).iter_errors(doc))
    assert not errors, [e.message for e in errors[:3]]


def _raw_point(lat, lon, result):
    """Like ``make_point`` but keeps ``None`` results (``make_leaf`` would coerce them to NaN)."""
    pt = make_point(lat, lon, [0.0] * len(result))
    pt.children[0].result = list(result)
    return pt


def _forecast_pl_tree(p1=(5.0, 6.0, 7.0, 8.0)):
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("pl",)),
        node("levelist", (850, 500)),
        node("param", ("167",)),
        node("step", (0, 6)),
        node("stream", ("oper",)),
        node("type", ("fc",)),
    )
    fc = tip(tree)
    # per point: [lvl850 step0, lvl850 step6, lvl500 step0, lvl500 step6]
    fc.add_child(make_point(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]))
    fc.add_child(_raw_point(50.0, 12.0, list(p1)))
    return tree


def _step_pl_tree(levels=(850, 500), dates=("2025-01-01T00:00:00",), times=(0, 6)):
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", tuple(np.datetime64(d) for d in dates)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("pl",)),
        node("levelist", levels),
        node("param", ("167",)),
        node("step", (0,)),
        node("stream", ("oper",)),
        node("type", ("fc",)),
        node("time", tuple(timedelta(hours=h) for h in times)),
    )
    n = len(dates) * len(levels) * len(times)
    leaf = tip(tree)
    leaf.add_child(make_point(48.0, 11.0, [float(i) for i in range(n)]))
    leaf.add_child(make_point(50.0, 12.0, [float(100 + i) for i in range(n)]))
    return tree


def _month_pl_tree():
    tree = chain(
        TensorIndexTree(),
        node("class", ("d1",)),
        node("stream", ("clmn",)),
        node("levtype", ("pl",)),
        node("levelist", (850, 500)),
        node("number", (0,)),
        node("year", (2020,)),
        node("month", (2, 3)),
        node("param", ("167",)),
    )
    leaf = tip(tree)
    leaf.add_child(make_point(48.0, 11.0, [300.0, 301.0, 302.0, 303.0]))
    leaf.add_child(make_point(50.0, 12.0, [310.0, 311.0, 312.0, 313.0]))
    return tree


def _reforecast_levels_tree(p1=(10.0, 20.0, 30.0, 40.0)):
    hdates = (np.datetime64("2025-07-14T00:00:00"),)
    times = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))
    # result layout per point: product(hdate, levelist, step, time)
    points = [(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]), (50.0, 12.0, list(p1))]
    return reforecast_separate_datetime_vertical_tree(points, hdates, times, (500, 850), point_factory=_raw_point)


class TestMultiPointSeriesVerticalComposite:
    def test_forecast_all_levels_one_coverage(self):
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(_forecast_pl_tree())
        _assert_spec_z(cj, (850, 500))
        cov = cj["coverages"][0]
        # point-major, then level (tree order of levels)
        assert cov["domain"]["axes"]["composite"]["values"] == [
            [11.0, 48.0, 850],
            [11.0, 48.0, 500],
            [12.0, 50.0, 850],
            [12.0, 50.0, 500],
        ]
        # row-major [t, composite]: step0 -> (p0l850, p0l500, p1l850, p1l500), step6 -> same
        assert cov["ranges"]["2t"]["values"] == [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
        assert cov["ranges"]["2t"]["shape"] == [2, 4]
        assert "Forecast date" in cov["mars:metadata"]

    def test_forecast_split_by_number_not_level(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("pl",)),
            node("levelist", (850, 500)),
            node("number", (0, 1)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("enfo",)),
            node("type", ("pf",)),
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]))
        fc.add_child(make_point(50.0, 12.0, [5.0, 6.0, 7.0, 8.0]))
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        _assert_spec_z(cj, (850, 500), n_coverages=2)
        assert sorted(c["mars:metadata"]["number"] for c in cj["coverages"]) == [0, 1]

    def test_forecast_pointseries_still_one_coverage_per_level(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("pl",)),
            node("levelist", (850, 500)),
            node("param", ("167",)),
            node("step", (0, 6)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
        )
        tip(tree).add_child(make_point(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]))
        cj = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)
        assert cj["domainType"] == "PointSeries"
        assert [c["domain"]["axes"]["z"]["values"] for c in cj["coverages"]] == [[850], [500]]
        assert [c["mars:metadata"]["levelist"] for c in cj["coverages"]] == [850, 500]
        assert_valid_covjson(cj)
        _assert_official_schema(cj)

    def test_forecast_missing_level_cell_is_null(self):
        # second point has no data for level 500 (both steps): null, not an error
        cj = (
            Covjsonkit()
            .encode("CoverageCollection", "timeseries")
            .from_polytope(_forecast_pl_tree(p1=(5.0, 6.0, None, None)))
        )
        _assert_spec_z(cj, (850, 500))
        vals = cj["coverages"][0]["ranges"]["2t"]["values"]
        assert vals == [1.0, 3.0, 5.0, None, 2.0, 4.0, 6.0, None]

    def test_forecast_all_missing_still_raises(self):
        tree = _forecast_pl_tree()
        # drop a range key entirely: no data for any cell of a step -> error kept
        enc = Covjsonkit().encode("CoverageCollection", "timeseries")
        orig = enc.walk_tree

        def walk(tree, fields, coords, mars_metadata, range_dict, date_key="date"):
            orig(tree, fields, coords, mars_metadata, range_dict, date_key=date_key)
            range_dict.clear()

        enc.walk_tree = walk
        with pytest.raises(IndexError, match="not found in range_dict"):
            enc.from_polytope(tree)

    def test_reforecast_all_levels_one_coverage(self):
        cj = (
            Covjsonkit()
            .encode("CoverageCollection", "MultiPointSeries")
            .from_polytope_reforecast(_reforecast_levels_tree())
        )
        _assert_spec_z(cj, (500, 850))
        cov = cj["coverages"][0]
        assert cov["domain"]["axes"]["composite"]["values"] == [
            [11.0, 48.0, 500],
            [11.0, 48.0, 850],
            [12.0, 50.0, 500],
            [12.0, 50.0, 850],
        ]
        # t0 (00:00): p0l500=1, p0l850=3, p1l500=10, p1l850=30; t1 (12:00): 2,4,20,40
        assert cov["ranges"]["t"]["values"] == [1.0, 3.0, 10.0, 30.0, 2.0, 4.0, 20.0, 40.0]

    def test_reforecast_missing_level_cell_is_null(self):
        # second point has no data for level 500 -> that cell is absent -> null series
        tree = _reforecast_levels_tree(p1=(None, None, 30.0, 40.0))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_reforecast(tree)
        _assert_spec_z(cj, (500, 850))
        vals = cj["coverages"][0]["ranges"]["t"]["values"]
        assert vals == [1.0, 3.0, None, 30.0, 2.0, 4.0, None, 40.0]

    def test_reforecast_globally_null_level_is_kept(self):
        # level 500 is null at EVERY point: it must still be in the composite (as nulls).
        hdates = (np.datetime64("2025-07-14T00:00:00"),)
        times = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))
        points = [(48.0, 11.0, [None, None, 3.0, 4.0]), (50.0, 12.0, [None, None, 30.0, 40.0])]
        tree = reforecast_separate_datetime_vertical_tree(points, hdates, times, (500, 850), point_factory=_raw_point)
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_reforecast(tree)
        _assert_spec_z(cj, (500, 850))
        cov = cj["coverages"][0]
        assert cov["domain"]["axes"]["composite"]["values"] == [
            [11.0, 48.0, 500],
            [11.0, 48.0, 850],
            [12.0, 50.0, 500],
            [12.0, 50.0, 850],
        ]
        assert cov["ranges"]["t"]["values"] == [None, 3.0, None, 30.0, None, 4.0, None, 40.0]

    def test_reforecast_efas_globally_null_level_is_kept(self):
        # Same for the efas (forecast) grouping, which sorts points/levels.
        times = (np.timedelta64(0, "h"),)
        points = [(50.0, 12.0, [None, 30.0]), (48.0, 11.0, [None, 3.0])]
        tree = reforecast_separate_datetime_vertical_tree(
            points, (np.datetime64("2025-07-14T00:00:00"),), times, (500, 850), stream="efas", point_factory=_raw_point
        )
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_reforecast(tree)
        _assert_spec_z(cj, (500, 850))
        comp = cj["coverages"][0]["domain"]["axes"]["composite"]["values"]
        assert [pt[2] for pt in comp] == [500, 850, 500, 850]
        assert comp[0][:2] == [11.0, 48.0]  # efas: ascending (lat, lon)
        assert [v for v in cj["coverages"][0]["ranges"]["t"]["values"]] == [None, 3.0, None, 30.0]

    def test_reanalysis_all_levels_one_coverage(self):
        # class=ce / stream=efcl via from_polytope(date_key="hdate"): one hdate branch each.
        def branch(hdate, p0, p1):
            br = chain(
                node("hdate", (hdate,)),
                node("domain", ("g",)),
                node("expver", ("4321",)),
                node("levtype", ("pl",)),
                node("param", ("130",)),
                node("levelist", (500, 850)),
                node("step", (0,)),
                node("stream", ("efcl",)),
                node("type", ("sfo",)),
            )
            parent = tip(br)
            parent.add_child(make_point(48.0, 11.0, p0))
            parent.add_child(make_point(50.0, 12.0, p1))
            return br

        tree = reforecast_tree(
            [
                branch(np.datetime64("2025-07-14T00:00:00"), [1.0, 2.0], [10.0, 20.0]),
                branch(np.datetime64("2025-07-15T00:00:00"), [3.0, 4.0], [30.0, 40.0]),
            ]
        )
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree, date_key="hdate")
        _assert_spec_z(cj, (500, 850))
        cov = cj["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T00:00:00Z", "2025-07-15T00:00:00Z"]
        assert "Forecast date" not in cov["mars:metadata"]
        # [t, point, level]: hdate0 -> (p0: 1,2; p1: 10,20); hdate1 -> (p0: 3,4; p1: 30,40)
        assert cov["ranges"]["t"]["values"] == [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0]

    def test_month_all_levels_one_coverage(self):
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_month(_month_pl_tree())
        _assert_spec_z(cj, (850, 500))
        cov = cj["coverages"][0]
        assert cov["domain"]["axes"]["t"]["values"] == ["2020-02-01T00:00:00Z", "2020-03-01T00:00:00Z"]
        # per point the result is [feb: 850, 500; mar: 850, 500] -> [t, point, level]
        assert cov["ranges"]["2t"]["values"] == [300.0, 301.0, 310.0, 311.0, 302.0, 303.0, 312.0, 313.0]

    def test_step_all_levels_one_coverage(self):
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_step(_step_pl_tree())
        _assert_spec_z(cj, (850, 500))
        cov = cj["coverages"][0]
        assert len(cov["domain"]["axes"]["t"]["values"]) == 2
        # per point: [850 t0, 850 t1, 500 t0, 500 t1] = 0,1,2,3 (p1: +100)
        assert cov["ranges"]["2t"]["values"] == [0.0, 2.0, 100.0, 102.0, 1.0, 3.0, 101.0, 103.0]

    def test_surface_unchanged(self):
        # No levels: composite keeps the 2-tuple form, no vertical referencing,
        # and the scalar ``levelist`` metadata is kept.
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        axes = cj["coverages"][0]["domain"]["axes"]
        assert set(axes) == {"t", "composite"}
        assert axes["composite"]["coordinates"] == ["x", "y"]
        assert all(len(pt) == 2 for pt in axes["composite"]["values"])
        assert cj["coverages"][0]["mars:metadata"]["levelist"] == 0
        assert all(ref["coordinates"] != ["z"] for ref in cj["referencing"])
        assert_valid_covjson(cj)
        _assert_official_schema(cj)


class TestOfficialSchemaGuard:
    def test_nonexistent_schema_path_fails(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COVJSON_SCHEMA", str(tmp_path / "missing.json"))
        with pytest.raises(AssertionError, match="does not exist"):
            _assert_official_schema({"coverages": []})


class TestMultiPointSeriesVerticalDecoder:
    def _decoded_forecast(self):
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(_forecast_pl_tree())
        return cj, Covjsonkit().decode(cj)

    def test_to_xarray_levelist_dimension(self):
        _, dec = self._decoded_forecast()
        ds = dec.to_xarray()
        assert dict(ds.sizes) == {"t": 2, "points": 2, "levelist": 2}
        assert ds["levelist"].values.tolist() == [850, 500]
        assert list(ds["latitude"].values) == [48.0, 50.0]
        assert list(ds["longitude"].values) == [11.0, 12.0]
        assert ds["2t"].dims == ("t", "points", "levelist")
        # [t, point, level]
        assert ds["2t"].values.tolist() == [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]]
        assert ds["2t"].sel(points=1, levelist=500).values.tolist() == [7.0, 8.0]

    def test_to_xarray_null_cell_is_nan(self):
        cj = (
            Covjsonkit()
            .encode("CoverageCollection", "timeseries")
            .from_polytope(_forecast_pl_tree(p1=(5.0, 6.0, None, None)))
        )
        ds = Covjsonkit().decode(cj).to_xarray()
        assert np.isnan(ds["2t"].sel(points=1, levelist=500).values).all()
        assert ds["2t"].sel(points=1, levelist=850).values.tolist() == [5.0, 6.0]

    def test_to_xarray_single_level_scalar_coordinate(self):
        tree = _step_pl_tree(levels=(850,))
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_step(tree)
        ds = Covjsonkit().decode(cj).to_xarray()
        assert dict(ds.sizes) == {"t": 2, "points": 2}
        assert float(ds["levelist"]) == 850

    def test_to_xarray_non_grid_falls_back_to_per_entry_coordinate(self):
        cj, _ = self._decoded_forecast()
        # Drop one composite entry (and its values): no longer a full points x levels grid.
        cov = cj["coverages"][0]
        cov["domain"]["axes"]["composite"]["values"].pop(3)
        rng = cov["ranges"]["2t"]
        rng["values"] = [v for k, v in enumerate(rng["values"]) if k % 4 != 3]
        rng["shape"] = [2, 3]
        ds = Covjsonkit().decode(cj).to_xarray()
        assert dict(ds.sizes) == {"t": 2, "points": 3}
        assert ds["levelist"].dims == ("points",)
        assert ds["levelist"].values.tolist() == [850, 500, 850]
        assert ds["2t"].values.tolist() == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]

    def test_to_xarray_multi_number_list(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("pl",)),
            node("levelist", (850, 500)),
            node("number", (0, 1)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("enfo",)),
            node("type", ("pf",)),
        )
        fc = tip(tree)
        fc.add_child(make_point(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]))
        fc.add_child(make_point(50.0, 12.0, [5.0, 6.0, 7.0, 8.0]))
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        ds0, ds1 = Covjsonkit().decode(cj).to_xarray()
        assert dict(ds0.sizes) == {"t": 1, "points": 2, "levelist": 2}
        assert ds0.attrs["number"] == 0 and ds1.attrs["number"] == 1

    def test_to_geojson_entry_per_composite_element(self):
        cj, dec = self._decoded_forecast()
        gj = dec.to_geojson()
        assert len(gj["features"]) == 4 * 2  # composite entries x times
        f0 = gj["features"][0]
        assert f0["geometry"]["coordinates"] == [11.0, 48.0, 850]
        assert f0["properties"]["2t"] == 1.0
        # features are composite-entry major, time inner
        assert gj["features"][1]["geometry"]["coordinates"] == [11.0, 48.0, 850]
        assert gj["features"][1]["properties"]["2t"] == 2.0
        assert gj["features"][2]["geometry"]["coordinates"] == [11.0, 48.0, 500]
        assert gj["features"][2]["properties"]["2t"] == 3.0
        assert gj["features"][-1]["geometry"]["coordinates"] == [12.0, 50.0, 500]
        assert gj["features"][-1]["properties"]["2t"] == 8.0

    def test_get_coordinates_entry_per_composite_element(self):
        _, dec = self._decoded_forecast()
        coords = dec.get_coordinates()
        t0, t1 = "2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"
        assert len(coords["2t"]) == 1
        rows = coords["2t"][0]
        # [lat, lon, level, fct, t, number]; t outer, composite (point-major, level) inner
        assert rows == [
            [48.0, 11.0, 850, t0, t0, 0],
            [48.0, 11.0, 500, t0, t0, 0],
            [50.0, 12.0, 850, t0, t0, 0],
            [50.0, 12.0, 500, t0, t0, 0],
            [48.0, 11.0, 850, t0, t1, 0],
            [48.0, 11.0, 500, t0, t1, 0],
            [50.0, 12.0, 850, t0, t1, 0],
            [50.0, 12.0, 500, t0, t1, 0],
        ]
        assert len(rows) == len(dec.get_values()["2t"][0])

    def test_get_coordinates_surface_multipoint(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        dec = Covjsonkit().decode(cj)
        coords = dec.get_coordinates()
        t0, t1 = "2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"
        assert coords["2t"] == [
            [
                [48.0, 11.0, None, t0, t0, 0],
                [50.0, 12.0, None, t0, t0, 0],
                [48.0, 11.0, None, t0, t1, 0],
                [50.0, 12.0, None, t0, t1, 0],
            ]
        ]
        assert len(coords["2t"][0]) == len(dec.get_values()["2t"][0])

    def test_get_coordinates_pointseries_unchanged(self):
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8])])
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope(tree)
        assert cj["domainType"] == "PointSeries"
        coords = Covjsonkit().decode(cj).get_coordinates()
        t0, t1 = "2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"
        assert coords["2t"] == [[[48.0, 11.0, None, t0, t0, 0], [48.0, 11.0, None, t0, t1, 0]]]


class TestMultiPointSeriesDecoderLayouts:
    """Permanent regression tests for non-default composite layouts the decoder must accept."""

    @staticmethod
    def _surface_cj():
        tree = _surface_tree([(48.0, 11.0, [264.9, 263.8]), (50.0, 12.0, [274.9, 273.8])])
        return Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)

    def _legacy_separate_z(self):
        # Older layout: 2-tuples composite plus a separate single-valued ``z`` axis.
        cj = self._surface_cj()
        cj["coverages"][0]["domain"]["axes"]["z"] = {"values": [850]}
        return cj

    def _reordered_labels(self):
        # Spec allows any order of composite components: here [y, x, z] = [lat, lon, level].
        cj = self._surface_cj()
        comp = cj["coverages"][0]["domain"]["axes"]["composite"]
        comp["coordinates"] = ["y", "x", "z"]
        comp["values"] = [[lat, lon, 850] for lon, lat in comp["values"]]
        return cj

    @pytest.mark.parametrize("builder", ["_legacy_separate_z", "_reordered_labels"])
    def test_decodes_same_as_default_layout(self, builder):
        cj = getattr(self, builder)()
        dec = Covjsonkit().decode(cj)

        ds = dec.to_xarray()
        assert dict(ds.sizes) == {"t": 2, "points": 2}
        assert list(ds["latitude"].values) == [48.0, 50.0]
        assert list(ds["longitude"].values) == [11.0, 12.0]
        assert float(ds["levelist"]) == 850
        assert ds["2t"].values.tolist() == [[264.9, 274.9], [263.8, 273.8]]

        gj = dec.to_geojson()
        assert gj["features"][0]["geometry"]["coordinates"] == [11.0, 48.0, 850]
        assert gj["features"][2]["geometry"]["coordinates"] == [12.0, 50.0, 850]

        rows = dec.get_coordinates()["2t"][0]
        assert rows[0][:3] == [48.0, 11.0, 850]
        assert rows[1][:3] == [50.0, 12.0, 850]
        assert len(rows) == 4


class TestStepForecastDate:
    """from_polytope_step: "Forecast date" only when the t axis covers a single date."""

    DATES = ("2025-01-01T00:00:00", "2025-01-02T00:00:00")

    @pytest.mark.parametrize("domaintype", ["PointSeries", "MultiPointSeries"])
    def test_single_date_is_set(self, domaintype):
        cj = (
            Covjsonkit()
            .encode("CoverageCollection", domaintype)
            .from_polytope_step(_step_tree([(48.0, 11.0, [1.0, 2.0])]))
        )
        assert cj["domainType"] == domaintype
        assert cj["coverages"][0]["mars:metadata"]["Forecast date"] == "2025-01-01T00:00:00Z"

    def test_multi_date_is_absent_multipoint(self):
        tree = _step_pl_tree(levels=(850,), dates=self.DATES)
        cj = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_step(tree)
        assert cj["domainType"] == "MultiPointSeries"
        for cov in cj["coverages"]:
            assert "Forecast date" not in cov["mars:metadata"]
            assert len(cov["domain"]["axes"]["t"]["values"]) == 4

    def test_multi_date_absent_single_point_pointseries(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", tuple(np.datetime64(d) for d in self.DATES)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
            node("time", (timedelta(0), timedelta(hours=6))),
        )
        tip(tree).add_child(make_point(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]))
        cj = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_step(tree)
        assert cj["domainType"] == "PointSeries"
        assert "Forecast date" not in cj["coverages"][0]["mars:metadata"]
        assert len(cj["coverages"][0]["domain"]["axes"]["t"]["values"]) == 4


_NULL_LEAF_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="null spatial leaf dropped by walker; fixed on feature/add_identifiers",
)


class TestWhollyNullSpatialLeaf:
    """A spatial point that is null everywhere is dropped by the tree walkers.

    Pre-existing on 62ed28b (identical failure there, also for PointSeries); the
    walker rewrite arrives with feature/add_identifiers. These strict xfails turn into
    failures when that lands, prompting removal of the marker.
    """

    @_NULL_LEAF_XFAIL
    def test_forecast_null_point_kept_as_nulls(self):
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
        )
        parent = tip(tree)
        parent.add_child(_raw_point(48.0, 11.0, [1.0, 2.0]))
        parent.add_child(_raw_point(50.0, 12.0, [None, None]))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope(tree)
        assert cj["coverages"][0]["domain"]["axes"]["composite"]["values"] == [[11.0, 48.0], [12.0, 50.0]]
        assert cj["coverages"][0]["ranges"]["2t"]["values"] == [1.0, None, 2.0, None]

    @_NULL_LEAF_XFAIL
    def test_step_null_point_kept_as_nulls(self):
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
        )
        parent = tip(tree)
        parent.add_child(_raw_point(48.0, 11.0, [1.0, 2.0]))
        parent.add_child(_raw_point(50.0, 12.0, [None, None]))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_step(tree)
        assert cj["coverages"][0]["ranges"]["2t"]["values"] == [1.0, None, 2.0, None]

    @_NULL_LEAF_XFAIL
    def test_month_null_point_kept_as_nulls(self):
        tree = chain(
            TensorIndexTree(),
            node("class", ("d1",)),
            node("stream", ("clmn",)),
            node("levtype", ("sfc",)),
            node("number", (0,)),
            node("year", (2020,)),
            node("month", (2, 3)),
            node("param", ("167",)),
        )
        parent = tip(tree)
        parent.add_child(_raw_point(48.0, 11.0, [1.0, 2.0]))
        parent.add_child(_raw_point(50.0, 12.0, [None, None]))
        cj = Covjsonkit().encode("CoverageCollection", "MultiPointSeries").from_polytope_month(tree)
        assert cj["coverages"][0]["ranges"]["2t"]["values"] == [1.0, None, 2.0, None]
