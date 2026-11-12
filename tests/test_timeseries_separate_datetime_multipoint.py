"""TimeSeries ``from_polytope_reforecast`` (separate date/hdate/time axes):
one point -> PointSeries, several points -> MultiPointSeries."""

import numpy as np
from conftest import forecast_separate_datetime_tree, reforecast_separate_datetime_tree

from covjsonkit.api import Covjsonkit

HDATES = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
TIMES = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))
P1 = (48.0, 11.0, [1.0, 2.0, 3.0, 4.0])
P2 = (50.0, 12.0, [10.0, 20.0, 30.0, 40.0])


def _encode(tree, domaintype="PointSeries"):
    return Covjsonkit().encode("CoverageCollection", domaintype).from_polytope_reforecast(tree)


def test_efcl_single_point_is_point_series():
    covjson = _encode(reforecast_separate_datetime_tree([P1], HDATES, TIMES))
    assert covjson["domainType"] == "PointSeries"
    assert len(covjson["coverages"]) == 1
    assert set(covjson["coverages"][0]["domain"]["axes"]) == {"x", "y", "t"}


def test_efcl_multi_point_is_multi_point_series():
    covjson = _encode(reforecast_separate_datetime_tree([P1, P2], HDATES, TIMES))
    assert covjson["domainType"] == "MultiPointSeries"
    assert len(covjson["coverages"]) == 1
    cov = covjson["coverages"][0]
    assert cov["domain"]["axes"]["t"]["values"] == [
        "2025-07-14T00:00:00Z",
        "2025-07-14T12:00:00Z",
        "2025-07-15T00:00:00Z",
        "2025-07-15T12:00:00Z",
    ]
    # composite is [x, y] = [lon, lat]
    assert cov["domain"]["axes"]["composite"] == {
        "dataType": "tuple",
        "coordinates": ["x", "y"],
        "values": [[11.0, 48.0], [12.0, 50.0]],
    }
    rng = next(iter(cov["ranges"].values()))
    assert rng["axisNames"] == ["t", "composite"]
    assert rng["shape"] == [4, 2]
    # Row-major [t, composite]
    assert rng["values"] == [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]


def test_efas_forecast_multi_point_one_coverage_per_run():
    tree = forecast_separate_datetime_tree(
        [(48.0, 11.0, [1.0, 2.0, 3.0, 4.0]), (50.0, 12.0, [10.0, 20.0, 30.0, 40.0])], TIMES, step=(6, 24)
    )
    covjson = _encode(tree)
    assert covjson["domainType"] == "MultiPointSeries"
    assert len(covjson["coverages"]) == 2
    first, second = covjson["coverages"]
    assert first["mars:metadata"]["Forecast date"] == "2026-05-01T00:00:00Z"
    assert second["mars:metadata"]["Forecast date"] == "2026-05-01T12:00:00Z"
    assert first["domain"]["axes"]["t"]["values"] == ["2026-05-01T06:00:00Z", "2026-05-02T00:00:00Z"]
    assert first["domain"]["axes"]["composite"]["values"] == [[11.0, 48.0], [12.0, 50.0]]
    assert next(iter(first["ranges"].values()))["values"] == [1.0, 10.0, 3.0, 30.0]
    assert next(iter(second["ranges"].values()))["values"] == [2.0, 20.0, 4.0, 40.0]


def test_forced_multipointseries_single_point():
    covjson = _encode(reforecast_separate_datetime_tree([P1], HDATES, TIMES), "multipointseries")
    assert covjson["domainType"] == "MultiPointSeries"
    cov = covjson["coverages"][0]
    assert cov["domain"]["axes"]["composite"]["values"] == [[11.0, 48.0]]
    assert next(iter(cov["ranges"].values()))["shape"] == [4, 1]
