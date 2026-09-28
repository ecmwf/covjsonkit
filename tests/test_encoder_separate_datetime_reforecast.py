"""Tests for the separate-datetime (independent date/hdate/time axes) reforecast
path in the base encoder's ``from_polytope_reforecast``.

Unlike the legacy merged representation (one coverage per hdate), separate-datetime
class=ce data has independent ``date``, ``hdate`` and ``time`` axes and produces one
coverage per ``(reference-datetime = hdate + time, step, number)``.
"""

import numpy as np
import pytest
from conftest import forecast_separate_datetime_tree, reforecast_separate_datetime_tree

from covjsonkit.api import Covjsonkit

# Two spatial points. Each result is the flattened product over (hdate, step, time)
# with a single step -> [h0t0, h0t1, h1t0, h1t1].
POINTS = [
    (48.0, 11.0, [1.0, 2.0, 3.0, 4.0]),
    (50.0, 12.0, [10.0, 20.0, 30.0, 40.0]),
]
HDATES = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
TIMES = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))

EXPECTED = [
    ("2025-07-14T00:00:00Z", [1.0, 10.0]),
    ("2025-07-14T12:00:00Z", [2.0, 20.0]),
    ("2025-07-15T00:00:00Z", [3.0, 30.0]),
    ("2025-07-15T12:00:00Z", [4.0, 40.0]),
]


@pytest.mark.parametrize("feature", ["BoundingBox", "Frame", "Circle", "Shapefile", "polygon"])
def test_separate_datetime_reforecast_composite_encoders(feature):
    tree = reforecast_separate_datetime_tree(POINTS, HDATES, TIMES)
    covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope_reforecast(tree)

    assert len(covjson["coverages"]) == len(EXPECTED)
    for cov, (ref, vals) in zip(covjson["coverages"], EXPECTED):
        assert cov["domain"]["axes"]["t"]["values"] == [ref]
        assert cov["domain"]["axes"]["composite"]["values"] == [
            [48.0, 11.0, 0],
            [50.0, 12.0, 0],
        ]
        # Single parameter (167 -> 2t).
        (param_name,) = cov["ranges"].keys()
        assert cov["ranges"][param_name]["values"] == vals
        # efcl coverages are delineated by their valid time (the t-axis).
        assert "Forecast date" not in cov["mars:metadata"]
        assert cov["mars:metadata"]["step"] == 0
        assert cov["mars:metadata"]["class"] == "ce"
        assert cov["mars:metadata"]["stream"] == "efcl"


def test_separate_datetime_reforecast_single_point_two_steps():
    # 2 hdates x 2 times x 2 steps = 8 coverages for a single point.
    # result layout: product(hdate, step, time) -> h0s0t0, h0s0t1, h0s1t0, h0s1t1, ...
    points = [(48.0, 11.0, [1, 2, 3, 4, 5, 6, 7, 8])]
    tree = reforecast_separate_datetime_tree(points, HDATES, TIMES, step=(0, 6))
    covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope_reforecast(tree)

    assert len(covjson["coverages"]) == 8
    # t = hdate + time + step; ordered by reference (hdate + time), then step.
    seen = [
        (c["domain"]["axes"]["t"]["values"][0], c["mars:metadata"]["step"], c["ranges"]["2t"]["values"][0])
        for c in covjson["coverages"]
    ]
    assert seen == [
        ("2025-07-14T00:00:00Z", 0, 1.0),
        ("2025-07-14T06:00:00Z", 6, 3.0),
        ("2025-07-14T12:00:00Z", 0, 2.0),
        ("2025-07-14T18:00:00Z", 6, 4.0),
        ("2025-07-15T00:00:00Z", 0, 5.0),
        ("2025-07-15T06:00:00Z", 6, 7.0),
        ("2025-07-15T12:00:00Z", 0, 6.0),
        ("2025-07-15T18:00:00Z", 6, 8.0),
    ]


def test_separate_datetime_reforecast_polygon_metadata_date_is_iso():
    # Climatology data carries ``date`` as a pd.Timestamp; it must render ISO-8601.
    import pandas as pd

    tree = reforecast_separate_datetime_tree(POINTS, HDATES, TIMES, date=pd.Timestamp("2023-01-01"))
    covjson = Covjsonkit().encode("CoverageCollection", "polygon").from_polytope_reforecast(tree)
    for cov in covjson["coverages"]:
        assert cov["mars:metadata"]["date"] == "2023-01-01T00:00:00"


def test_efas_forecast_polygon_valid_time_and_run_reference():
    # No hdate axis: reference = date + time (the run), t = date + time + step.
    # result layout: product(date, step, time) -> s6t0, s6t12, s24t0, s24t12.
    tree = forecast_separate_datetime_tree([(50.0, 7.0, [1, 2, 3, 4]), (50.1, 7.1, [5, 6, 7, 8])], TIMES, step=(6, 24))
    covjson = Covjsonkit().encode("CoverageCollection", "polygon").from_polytope_reforecast(tree)

    seen = [
        (
            c["mars:metadata"]["Forecast date"],
            c["mars:metadata"]["step"],
            c["domain"]["axes"]["t"]["values"],
            next(iter(c["ranges"].values()))["values"],
        )
        for c in covjson["coverages"]
    ]
    assert seen == [
        ("2026-05-01T00:00:00Z", 6, ["2026-05-01T06:00:00Z"], [1.0, 5.0]),
        ("2026-05-01T00:00:00Z", 24, ["2026-05-02T00:00:00Z"], [3.0, 7.0]),
        ("2026-05-01T12:00:00Z", 6, ["2026-05-01T18:00:00Z"], [2.0, 6.0]),
        ("2026-05-01T12:00:00Z", 24, ["2026-05-02T12:00:00Z"], [4.0, 8.0]),
    ]
    for cov in covjson["coverages"]:
        # Raw date is folded into "Forecast date".
        assert "date" not in cov["mars:metadata"]
