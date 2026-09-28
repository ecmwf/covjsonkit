"""Spec-compliance of the class=ce PointSeries (timeseries) reforecast paths.

The separate-datetime reforecast/forecast encoders must emit the same
spec-compliant x/y/[z]/t domain as the other PointSeries paths: ``referencing``
must name exactly the domain's axes, and ``z`` is only present when the request
tree carries a ``levelist`` axis.
"""

import numpy as np
from conftest import (
    assert_valid_covjson,
    forecast_separate_datetime_tree,
    reforecast_branch,
    reforecast_separate_datetime_tree,
    reforecast_separate_datetime_vertical_tree,
    reforecast_tree,
)

from covjsonkit.api import Covjsonkit

LAT, LON = 50.73746373267438, 7.107723168102066


def _encoder():
    return Covjsonkit().encode("CoverageCollection", "timeseries")


def _assert_references_match_axes(covjson, include_z):
    expected = {"x", "y", "t", "z"} if include_z else {"x", "y", "t"}
    referenced = {c for ref in covjson["referencing"] for c in ref["coordinates"]}
    assert referenced == expected
    for cov in covjson["coverages"]:
        axes = cov["domain"]["axes"]
        assert set(axes) == expected
        assert axes["x"]["values"] == [LON]
        assert axes["y"]["values"] == [LAT]
        for rng in cov["ranges"].values():
            assert rng["axisNames"] == ["t"]
    assert_valid_covjson(covjson)


def test_efas_forecast_is_spec_compliant():
    times = (np.timedelta64(0, "h"), np.timedelta64(12, "h"))
    tree = forecast_separate_datetime_tree([(LAT, LON, [1, 2, 3, 4])], times)
    covjson = _encoder().from_polytope_reforecast(tree)

    _assert_references_match_axes(covjson, include_z=False)


def test_efcl_separate_datetime_sfc_is_spec_compliant():
    tree = reforecast_separate_datetime_tree(
        [(LAT, LON, [1, 2])],
        hdates=(np.datetime64("2020-03-01"), np.datetime64("2021-03-01")),
        times=(np.timedelta64(0, "h"),),
    )
    covjson = _encoder().from_polytope_reforecast(tree)

    _assert_references_match_axes(covjson, include_z=False)


def test_efcl_separate_datetime_levels_emit_z_axis():
    # result layout: product(hdate, levelist, step, time) -> (levelist,)
    tree = reforecast_separate_datetime_vertical_tree(
        [(LAT, LON, [1, 2])],
        hdates=(np.datetime64("2020-03-01"),),
        times=(np.timedelta64(0, "h"),),
        levels=(500, 850),
    )
    covjson = _encoder().from_polytope_reforecast(tree)

    _assert_references_match_axes(covjson, include_z=True)
    by_level = {c["domain"]["axes"]["z"]["values"][0]: c for c in covjson["coverages"]}
    assert set(by_level) == {500, 850}
    (param,) = by_level[500]["ranges"].keys()
    assert by_level[500]["ranges"][param]["values"] == [1]
    assert by_level[850]["ranges"][param]["values"] == [2]


def test_collapse_reanalysis_is_spec_compliant():
    tree = reforecast_tree(
        [
            reforecast_branch(np.datetime64("2020-03-01"), [(LAT, LON, [1])]),
            reforecast_branch(np.datetime64("2021-03-01"), [(LAT, LON, [2])]),
        ]
    )
    covjson = _encoder().from_polytope(tree, date_key="hdate")

    _assert_references_match_axes(covjson, include_z=False)
    assert len(covjson["coverages"]) == 1


def test_walk_tree_reforecast_levels_emit_z_axis():
    tree = reforecast_separate_datetime_vertical_tree(
        [(LAT, LON, [1, 2])],
        hdates=(np.datetime64("2020-03-01"),),
        times=(np.timedelta64(0, "h"),),
        levels=(500, 850),
    )
    covjson = _encoder().from_polytope(tree, date_key="hdate", reforecast=True)

    _assert_references_match_axes(covjson, include_z=True)
    assert sorted(c["domain"]["axes"]["z"]["values"][0] for c in covjson["coverages"]) == [500, 850]
