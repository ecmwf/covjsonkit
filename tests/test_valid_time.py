"""Tests for the shared valid-time helpers and their use across encoders/decoders."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import (
    chain,
    forecast_tree,
    make_point,
    node,
    reforecast_separate_datetime_tree,
    reforecast_separate_datetime_vertical_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit
from covjsonkit.encoder.encoder import (
    iso_utc,
    range_shape,
    reference_datetime,
    set_forecast_date,
    step_to_timedelta,
    time_to_timedelta,
    valid_time,
)

HDATES = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
TIMES = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))


@pytest.mark.parametrize(
    "step,expected",
    [
        (6, timedelta(hours=6)),
        (6.5, timedelta(hours=6, minutes=30)),
        ("6", timedelta(hours=6)),
        ("6h", timedelta(hours=6)),
        ("30m", timedelta(minutes=30)),
        ("13h30m", timedelta(hours=13, minutes=30)),
        ("0-6", timedelta(hours=6)),
        ("12-18", timedelta(hours=18)),
        (timedelta(hours=3), timedelta(hours=3)),
        (np.timedelta64(90, "m"), timedelta(minutes=90)),
        (pd.Timedelta(hours=2), timedelta(hours=2)),
        ([6], timedelta(hours=6)),
        (None, timedelta(0)),
    ],
)
def test_step_to_timedelta(step, expected):
    assert step_to_timedelta(step) == expected


@pytest.mark.parametrize("step", ["abc", "6x", [0, 6]])
def test_step_to_timedelta_rejects_unknown(step):
    with pytest.raises(ValueError):
        step_to_timedelta(step)


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.timedelta64(6, "h"), timedelta(hours=6)),
        (timedelta(hours=12), timedelta(hours=12)),
        ("06:00:00", timedelta(hours=6)),
        ("0600", timedelta(hours=6)),
        (1230, timedelta(hours=12, minutes=30)),
        (6, timedelta(hours=6)),
        (None, timedelta(0)),
    ],
)
def test_time_to_timedelta(value, expected):
    assert time_to_timedelta(value) == expected


def test_valid_time_formats_iso_utc():
    assert valid_time("2025-01-01T00:00:00Z", "30m") == "2025-01-01T00:30:00Z"
    assert valid_time(np.datetime64("2025-01-01"), 6, np.timedelta64(12, "h")) == "2025-01-01T18:00:00Z"
    # Timezone-aware input is converted to UTC; no "+00:00Z" suffix.
    assert valid_time(pd.Timestamp("2025-01-01T01:00:00+01:00"), 0) == "2025-01-01T00:00:00Z"
    assert iso_utc(pd.Timestamp("2025-01-01", tz="UTC")) == "2025-01-01T00:00:00Z"
    assert reference_datetime("2025-01-01Z", "1200") == pd.Timestamp("2025-01-01T12:00:00")


def test_set_forecast_date_rules():
    ref = pd.Timestamp("2025-01-01T12:00:00")
    assert set_forecast_date({"class": "od"}, ref)["Forecast date"] == "2025-01-01T12:00:00Z"
    efcl = set_forecast_date({"class": "ce", "stream": "efcl", "Forecast date": "x"}, ref)
    assert "Forecast date" not in efcl
    efas = set_forecast_date({"class": "ce", "stream": "efas", "date": "2025-01-01"}, ref)
    assert efas["Forecast date"] == "2025-01-01T12:00:00Z"
    assert "date" not in efas


def test_range_shape():
    assert range_shape([1.0], "t") == ([], [])
    assert range_shape([1.0, 2.0], "t") == ([2], ["t"])


@pytest.mark.parametrize("feature", ["timeseries", "position", "boundingbox", "verticalprofile", "grid", "path"])
@pytest.mark.parametrize("step,expected", [("30m", "2025-01-01T00:30:00Z"), ("6", "2025-01-01T06:00:00Z")])
def test_string_steps_from_polytope(feature, step, expected):
    covjson = getattr(Covjsonkit().encode("CoverageCollection", feature), "from_polytope")(
        forecast_tree([(48.0, 11.0, [1.0])], step=(step,))
    )
    axes = covjson["coverages"][0]["domain"]["axes"]
    t = axes["t"]["values"] if "t" in axes else [axes["composite"]["values"][0][0]]
    assert t == [expected]


@pytest.mark.parametrize("feature", ["timeseries", "position", "verticalprofile"])
@pytest.mark.parametrize("step,expected", [("30m", "2025-07-14T00:30:00Z"), ("6", "2025-07-14T06:00:00Z")])
def test_string_steps_from_polytope_reforecast(feature, step, expected):
    tree = reforecast_separate_datetime_tree([(48.0, 11.0, [1.0])], HDATES[:1], TIMES[:1], step=(step,))
    covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope_reforecast(tree)
    assert covjson["coverages"][0]["domain"]["axes"]["t"]["values"] == [expected]


def _forecast_tree_with_time(times):
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
        node("time", times),
        node("type", ("fc",)),
    )
    tip(tree).add_child(make_point(48.0, 11.0, [1.0, 2.0]))
    return tree


def test_walk_tree_folds_single_time_into_date():
    covjson = (
        Covjsonkit()
        .encode("CoverageCollection", "boundingbox")
        .from_polytope(_forecast_tree_with_time((np.timedelta64(12, "h"),)))
    )
    seen = [(c["mars:metadata"]["Forecast date"], c["domain"]["axes"]["t"]["values"]) for c in covjson["coverages"]]
    assert seen == [
        ("2025-01-01T12:00:00Z", ["2025-01-01T12:00:00Z"]),
        ("2025-01-01T12:00:00Z", ["2025-01-01T18:00:00Z"]),
    ]


def test_walk_tree_rejects_multiple_times():
    with pytest.raises(ValueError, match="from_polytope_step or from_polytope_reforecast"):
        Covjsonkit().encode("CoverageCollection", "boundingbox").from_polytope(
            _forecast_tree_with_time((np.timedelta64(0, "h"), np.timedelta64(12, "h")))
        )


def _step_tree(step):
    return chain(
        TensorIndexTree(),
        node("class", ("d1",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("expver", ("0001",)),
        node("levtype", ("sfc",)),
        node("param", ("167",)),
        node("step", (step,)),
        node("stream", ("clte",)),
        node("time", (timedelta(0), timedelta(hours=6))),
        make_point(48.0, 11.0, [264.0, 263.0]),
    )


@pytest.mark.parametrize("feature", ["timeseries", "position"])
def test_step_path_valid_time_is_iso_and_includes_step(feature):
    covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope_step(_step_tree(1))
    assert covjson["coverages"][0]["domain"]["axes"]["t"]["values"] == [
        "2025-01-01T01:00:00Z",
        "2025-01-01T07:00:00Z",
    ]


def test_grid_reforecast_separate_datetime():
    points = [(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]), (48.0, 12.0, [5.0, 6.0, 7.0, 8.0])]
    tree = reforecast_separate_datetime_tree(points, HDATES, TIMES)
    covjson = Covjsonkit().encode("CoverageCollection", "grid").from_polytope_reforecast(tree)

    assert [r["coordinates"] for r in covjson["referencing"]] == [["x", "y"], ["t"]]
    assert len(covjson["coverages"]) == 4
    cov = covjson["coverages"][0]
    assert cov["domain"]["axes"] == {
        "t": {"values": ["2025-07-14T00:00:00Z"]},
        "y": {"values": [48.0]},
        "x": {"values": [11.0, 12.0]},
    }
    assert cov["ranges"]["2t"]["shape"] == [1, 1, 2]
    assert cov["ranges"]["2t"]["axisNames"] == ["t", "y", "x"]
    assert cov["ranges"]["2t"]["values"] == [1.0, 5.0]
    assert "Forecast date" not in cov["mars:metadata"]


def test_grid_reforecast_steps_share_a_coverage():
    points = [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])]
    tree = reforecast_separate_datetime_tree(points, HDATES, TIMES, step=(0, 6))
    covjson = Covjsonkit().encode("CoverageCollection", "grid").from_polytope_reforecast(tree)
    cov = covjson["coverages"][0]
    # h0 t0: step 0 -> idx0=1, step 6 -> idx2=3
    assert cov["domain"]["axes"]["t"]["values"] == ["2025-07-14T00:00:00Z", "2025-07-14T06:00:00Z"]
    assert cov["ranges"]["2t"]["values"] == [1.0, 3.0]


def test_path_reforecast_separate_datetime():
    points = [(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]), (50.0, 12.0, [5.0, 6.0, 7.0, 8.0])]
    tree = reforecast_separate_datetime_tree(points, HDATES, TIMES)
    covjson = Covjsonkit().encode("CoverageCollection", "path").from_polytope_reforecast(tree)

    assert [r["coordinates"] for r in covjson["referencing"]] == [["x", "y"], ["t"]]
    cov = covjson["coverages"][0]
    composite = cov["domain"]["axes"]["composite"]
    assert composite["coordinates"] == ["t", "x", "y"]
    assert composite["values"] == [["2025-07-14T00:00:00Z", 11.0, 48.0], ["2025-07-14T00:00:00Z", 12.0, 50.0]]
    assert cov["ranges"]["2t"]["values"] == [1.0, 5.0]


def test_path_reforecast_with_levels():
    points = [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])]
    tree = reforecast_separate_datetime_vertical_tree(points, HDATES, TIMES, (500, 850))
    covjson = Covjsonkit().encode("CoverageCollection", "path").from_polytope_reforecast(tree)
    composite = covjson["coverages"][0]["domain"]["axes"]["composite"]
    assert composite["coordinates"] == ["t", "x", "y", "z"]
    assert composite["values"] == [["2025-07-14T00:00:00Z", 11.0, 48.0, 500], ["2025-07-14T00:00:00Z", 11.0, 48.0, 850]]


def test_multipoint_multi_step_round_trips_to_xarray():
    tree = forecast_tree([(48.0, 11.0, [1.0, 2.0]), (50.0, 12.0, [3.0, 4.0])], step=(0, 6))
    covjson = Covjsonkit().encode("CoverageCollection", "boundingbox").from_polytope(tree)
    ds = Covjsonkit().decode(covjson).to_xarray()
    assert list(ds["steps"].values) == [0, 6]
    assert list(ds["datetimes"].values) == ["2025-01-01T00:00:00Z"]
    assert ds["2t"].values.tolist() == [[[[1.0, 3.0], [2.0, 4.0]]]]


@pytest.mark.parametrize("feature", ["boundingbox", "grid", "path", "verticalprofile", "position"])
def test_efcl_reforecast_round_trips_to_xarray(feature):
    if feature == "verticalprofile":
        tree = reforecast_separate_datetime_vertical_tree(
            [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])], HDATES, TIMES, (500, 850)
        )
    else:
        tree = reforecast_separate_datetime_tree([(48.0, 11.0, [1.0, 2.0, 3.0, 4.0])], HDATES, TIMES)
    covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope_reforecast(tree)
    assert all("Forecast date" not in c["mars:metadata"] for c in covjson["coverages"])
    decoder = Covjsonkit().decode(covjson)
    decoder.to_xarray()
    decoder.to_geojson()
