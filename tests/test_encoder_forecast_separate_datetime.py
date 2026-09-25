"""Tests for the separate-datetime *forecast* (class=ce, stream=efas) path in
TimeSeries.from_polytope_reforecast.

Forecast data has no ``hdate`` axis: ``date + time`` is the forecast run
reference and ``step`` is the lead time. Each ``(date, time)`` run must become
its own coverage, with the steps forming that coverage's ``t``-axis -- unlike
the efcl reforecast path which collapses everything into a single coverage.
"""

import numpy as np
from conftest import chain, forecast_separate_datetime_tree, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

TIMES = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))


def test_efas_forecast_splits_runs_by_datetime():
    # result layout: product(date, step, time) with single date -> (step, time):
    #   idx 0: step6 time0 -> 1
    #   idx 1: step6 time1 -> 2
    #   idx 2: step24 time0 -> 3
    #   idx 3: step24 time1 -> 4
    points = [(50.73746373267438, 7.107723168102066, [1, 2, 3, 4])]
    tree = forecast_separate_datetime_tree(points, TIMES, step=(6, 24))
    covjson = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_reforecast(tree)

    # Two runs (2026-05-01 00:00 and 12:00), NOT a single collapsed coverage.
    assert len(covjson["coverages"]) == 2

    by_ref = {c["mars:metadata"]["Forecast date"]: c for c in covjson["coverages"]}
    assert set(by_ref) == {"2026-05-01T00:00:00Z", "2026-05-01T12:00:00Z"}

    # Run 00:00 -> steps 6, 24 -> valid times 06:00 and next-day 00:00.
    run0 = by_ref["2026-05-01T00:00:00Z"]
    assert run0["domain"]["axes"]["t"]["values"] == [
        "2026-05-01T06:00:00Z",
        "2026-05-02T00:00:00Z",
    ]
    (param_name,) = run0["ranges"].keys()
    assert run0["ranges"][param_name]["values"] == [1, 3]

    # Run 12:00 -> steps 6, 24 -> valid times 18:00 and next-day 12:00.
    run1 = by_ref["2026-05-01T12:00:00Z"]
    assert run1["domain"]["axes"]["t"]["values"] == [
        "2026-05-01T18:00:00Z",
        "2026-05-02T12:00:00Z",
    ]
    assert run1["ranges"][param_name]["values"] == [2, 4]

    for cov in covjson["coverages"]:
        assert cov["mars:metadata"]["class"] == "ce"
        assert cov["mars:metadata"]["stream"] == "efas"
        # The raw date/time axes must not leak into metadata; the run reference
        # is captured by "Forecast date" and the t-axis.
        assert "date" not in cov["mars:metadata"]
        assert "time" not in cov["mars:metadata"]


def test_efas_forecast_coverages_ordered_date_then_time():
    """A time-major tree (time above date) must still emit coverages ordered
    date -> time -> point.

    We build the tree manually with the ``time`` axis *above* ``date`` so the
    natural insertion/tree-traversal order is time-major
    (May29 00Z, May30 00Z, May29 12Z, May30 12Z). The encoder must reorder the
    output by the run reference (date + time) to
    (May29 00Z, May29 12Z, May30 00Z, May30 12Z).
    """
    dates = (np.datetime64("2026-05-29"), np.datetime64("2026-05-30"))
    times = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))

    # Single step so each run is a single value. Result is the flattened product
    # over (time, date) in tree order:
    #   idx 0: time0 (00Z) date0 (May29) -> 10
    #   idx 1: time0 (00Z) date1 (May30) -> 20
    #   idx 2: time1 (12Z) date0 (May29) -> 30
    #   idx 3: time1 (12Z) date1 (May30) -> 40
    tree = chain(
        TensorIndexTree(),
        node("class", ("ce",)),
        node("time", times),
        node("date", dates),
        node("domain", ("g",)),
        node("expver", ("8888",)),
        node("levtype", ("sfc",)),
        node("model", ("lisflood",)),
        node("origin", ("ecmf",)),
        node("param", ("240023",)),
        node("step", (6,)),
        node("stream", ("efas",)),
        node("type", ("cf",)),
    )
    tip(tree).add_child(make_point(50.0, 7.0, [10, 20, 30, 40]))

    covjson = Covjsonkit().encode("CoverageCollection", "timeseries").from_polytope_reforecast(tree)

    assert len(covjson["coverages"]) == 4

    refs = [c["mars:metadata"]["Forecast date"] for c in covjson["coverages"]]
    assert refs == [
        "2026-05-29T00:00:00Z",
        "2026-05-29T12:00:00Z",
        "2026-05-30T00:00:00Z",
        "2026-05-30T12:00:00Z",
    ]

    values = [next(iter(c["ranges"].values()))["values"][0] for c in covjson["coverages"]]
    assert values == [10, 30, 20, 40]
