"""Tests for the separate-datetime (independent date/hdate/time axes) reforecast
path in the point-wise encoders that override ``from_polytope_reforecast``:
Position (PointSeries) and VerticalProfile.

These encoders don't use the composite/multi-point domain; instead they emit one
coverage per spatial point (Position: t across steps; VerticalProfile: levelist
across levels, single valid time per step).
"""

import numpy as np
from conftest import (
    reforecast_separate_datetime_tree,
    reforecast_separate_datetime_vertical_tree,
)

from covjsonkit.api import Covjsonkit

HDATES = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
TIMES = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))

# Two spatial points. Each result is the flattened product over (hdate, step, time).
# With a single step this is [h0t0, h0t1, h1t0, h1t1].
POINTS = [
    (48.0, 11.0, [1.0, 2.0, 3.0, 4.0]),
    (50.0, 12.0, [10.0, 20.0, 30.0, 40.0]),
]

# (point lat/lon, valid datetime, value). With a single step the valid time
# equals the reference datetime hdate + time.
POSITION_EXPECTED = [
    ((48.0, 11.0), "2025-07-14T00:00:00Z", 1.0),
    ((48.0, 11.0), "2025-07-14T12:00:00Z", 2.0),
    ((48.0, 11.0), "2025-07-15T00:00:00Z", 3.0),
    ((48.0, 11.0), "2025-07-15T12:00:00Z", 4.0),
    ((50.0, 12.0), "2025-07-14T00:00:00Z", 10.0),
    ((50.0, 12.0), "2025-07-14T12:00:00Z", 20.0),
    ((50.0, 12.0), "2025-07-15T00:00:00Z", 30.0),
    ((50.0, 12.0), "2025-07-15T12:00:00Z", 40.0),
]

POINT_SERIES_REFERENCING = [
    {
        "coordinates": ["x", "y"],
        "system": {"type": "GeographicCRS", "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
    },
    {"coordinates": ["t"], "system": {"type": "TemporalRS", "calendar": "Gregorian"}},
]


def test_position_separate_datetime_single_step():
    tree = reforecast_separate_datetime_tree(POINTS, HDATES, TIMES)
    covjson = Covjsonkit().encode("CoverageCollection", "position").from_polytope_reforecast(tree)

    assert covjson["domainType"] == "PointSeries"
    assert covjson["referencing"] == POINT_SERIES_REFERENCING
    assert len(covjson["coverages"]) == len(POSITION_EXPECTED)

    seen = set()
    for cov in covjson["coverages"]:
        axes = cov["domain"]["axes"]
        # Surface data: no levelist axis in the tree, so no z axis.
        assert set(axes) == {"x", "y", "t"}
        lon = axes["x"]["values"][0]
        lat = axes["y"]["values"][0]
        (valid_time,) = axes["t"]["values"]
        (param_name,) = cov["ranges"].keys()
        (value,) = cov["ranges"][param_name]["values"]
        # Reanalysis (efcl): only the valid-time is exposed.
        assert "Forecast date" not in cov["mars:metadata"]
        assert "step" not in cov["mars:metadata"]
        assert cov["mars:metadata"]["class"] == "ce"
        seen.add(((lat, lon), valid_time, value))

    assert seen == set(POSITION_EXPECTED)


def test_position_separate_datetime_non_efcl_keeps_forecast_date():
    tree = reforecast_separate_datetime_tree(POINTS, HDATES, TIMES, stream="enfh")
    covjson = Covjsonkit().encode("CoverageCollection", "position").from_polytope_reforecast(tree)

    for cov in covjson["coverages"]:
        # Single step -> the valid time equals the reference datetime.
        assert cov["domain"]["axes"]["t"]["values"] == [cov["mars:metadata"]["Forecast date"]]


def test_position_separate_datetime_multi_step_collapses_into_t():
    # 2 hdates x 2 times x 2 steps. result layout: product(hdate, step, time).
    # idx: 0 h0s0t0, 1 h0s0t1, 2 h0s1t0, 3 h0s1t1, 4 h1s0t0, 5 h1s0t1, 6 h1s1t0, 7 h1s1t1
    points = [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])]
    tree = reforecast_separate_datetime_tree(points, HDATES, TIMES, step=(0, 6))
    covjson = Covjsonkit().encode("CoverageCollection", "position").from_polytope_reforecast(tree)

    # One coverage per (point, level, number, reference) = 4 references, single point.
    assert len(covjson["coverages"]) == 4

    # Key by the first valid time (reference + step 0).
    by_ref = {c["domain"]["axes"]["t"]["values"][0]: c for c in covjson["coverages"]}
    # Reference 2025-07-14T00:00 (h0, t0): step0 -> idx0=1, step6 -> idx2=3.
    cov = by_ref["2025-07-14T00:00:00Z"]
    assert cov["domain"]["axes"]["t"]["values"] == [
        "2025-07-14T00:00:00Z",
        "2025-07-14T06:00:00Z",
    ]
    (param_name,) = cov["ranges"].keys()
    assert cov["ranges"][param_name]["values"] == [1, 3]


def test_verticalprofile_separate_datetime():
    # 2 levels, layout product(hdate, levelist, step, time) with single step:
    # idx: 0 h0 L0 t0, 1 h0 L0 t1, 2 h0 L1 t0, 3 h0 L1 t1,
    #      4 h1 L0 t0, 5 h1 L0 t1, 6 h1 L1 t0, 7 h1 L1 t1
    points = [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])]
    levels = (500, 850)
    tree = reforecast_separate_datetime_vertical_tree(points, HDATES, TIMES, levels)
    covjson = Covjsonkit().encode("CoverageCollection", "verticalprofile").from_polytope_reforecast(tree)

    assert covjson["domainType"] == "VerticalProfile"
    assert [r["coordinates"] for r in covjson["referencing"]] == [["x", "y"], ["z"], ["t"]]
    # One coverage per (point, number, reference, step) = 4 references, single point/step.
    assert len(covjson["coverages"]) == 4

    by_time = {c["domain"]["axes"]["t"]["values"][0]: c for c in covjson["coverages"]}
    cov = by_time["2025-07-14T00:00:00Z"]
    axes = cov["domain"]["axes"]
    assert axes["x"]["values"] == [11.0]
    assert axes["y"]["values"] == [48.0]
    assert axes["z"]["values"] == [500, 850]
    (param_name,) = cov["ranges"].keys()
    # L0 t0 -> idx0=1, L1 t0 -> idx2=3
    assert cov["ranges"][param_name]["values"] == [1, 3]
    assert cov["ranges"][param_name]["axisNames"] == ["z"]
    # Reanalysis (efcl): only the valid-time is exposed.
    assert "Forecast date" not in cov["mars:metadata"]
    assert "step" not in cov["mars:metadata"]


def test_verticalprofile_separate_datetime_non_efcl_keeps_forecast_date():
    points = [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])]
    tree = reforecast_separate_datetime_vertical_tree(points, HDATES, TIMES, (500, 850), stream="enfh")
    covjson = Covjsonkit().encode("CoverageCollection", "verticalprofile").from_polytope_reforecast(tree)

    for cov in covjson["coverages"]:
        assert cov["domain"]["axes"]["t"]["values"] == [cov["mars:metadata"]["Forecast date"]]
        assert cov["mars:metadata"]["step"] == 0


def test_pointwise_reforecast_backward_compat_delegates():
    # A legacy merged tree without a separate ``time`` axis should delegate to
    # from_polytope(date_key="hdate") and still produce PointSeries coverages.
    from conftest import reforecast_branch, reforecast_tree

    branch = reforecast_branch(np.datetime64("2025-07-14T00:00:00"), POINTS)
    tree = reforecast_tree([branch])
    covjson = Covjsonkit().encode("CoverageCollection", "position").from_polytope_reforecast(tree)
    assert covjson["domainType"] == "PointSeries"
    assert len(covjson["coverages"]) > 0
