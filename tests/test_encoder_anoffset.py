"""anoffset on the separate-datetime path (``from_polytope_reforecast``) used for every class=ce request.

For ``class=ce, stream=efas`` the valid time is ``date + time + step - anoffset`` while
``"Forecast date"`` stays the run (``date + time``); ``anoffset`` is kept in mars:metadata as
integer hours. Cases mirror the polytope-efas-configs anoffset fixtures.
"""

import numpy as np
import pytest
from conftest import chain, forecast_separate_datetime_tree, make_point, node, tip
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit

T00 = (np.timedelta64(0, "h"),)
T12 = (np.timedelta64(12, "h"),)
MULTIPOINT_FEATURES = ["boundingbox", "circle", "frame", "shapefile", "polygon"]


def _encode(feature, tree):
    return Covjsonkit().encode("CoverageCollection", feature).from_polytope_reforecast(tree)


@pytest.mark.parametrize("anoffset", [18, "18"])
def test_timeseries_anoffset_shifts_valid_time(anoffset):
    # fu-ecmf: 2026-05-01 00:00 run, steps 6/12/18, anoffset 18h.
    tree = forecast_separate_datetime_tree(
        [(50.7, 7.1, [1.0, 2.0, 3.0])], T00, step=(6, 12, 18), anoffset=(anoffset,), type_="fu"
    )
    cov = _encode("timeseries", tree)["coverages"][0]
    assert cov["domain"]["axes"]["t"]["values"] == [
        "2026-04-30T12:00:00Z",
        "2026-04-30T18:00:00Z",
        "2026-05-01T00:00:00Z",
    ]
    assert cov["ranges"]["dis06"]["values"] == [1.0, 2.0, 3.0]
    meta = cov["mars:metadata"]
    assert meta["Forecast date"] == "2026-05-01T00:00:00Z"
    assert meta["anoffset"] == 18
    assert "date" not in meta


def test_position_anoffset_shifts_valid_time():
    # fu-edzw: 12:00 run, step 6h, anoffset 30h -> previous day 12:00.
    tree = forecast_separate_datetime_tree([(50.7, 7.1, [1.0])], T12, step=(6,), anoffset=(30,), type_="fu")
    cov = _encode("position", tree)["coverages"][0]
    assert cov["domain"]["axes"]["t"]["values"] == ["2026-04-30T12:00:00Z"]
    assert cov["mars:metadata"]["Forecast date"] == "2026-05-01T12:00:00Z"
    assert cov["mars:metadata"]["anoffset"] == 30


@pytest.mark.parametrize("feature", MULTIPOINT_FEATURES)
def test_multipoint_anoffset_shifts_valid_time(feature):
    # sfo-ecmf rhombus: 00:00 run, steps 6/12, anoffset 90h.
    points = [(50.7, 7.1, [1.0, 2.0]), (50.8, 7.2, [3.0, 4.0])]
    tree = forecast_separate_datetime_tree(points, T00, step=(6, 12), anoffset=(90,), type_="sfo")
    covjson = _encode(feature, tree)
    seen = [
        (c["mars:metadata"]["Forecast date"], c["mars:metadata"]["step"], c["domain"]["axes"]["t"]["values"])
        for c in covjson["coverages"]
    ]
    assert seen == [
        ("2026-05-01T00:00:00Z", 6, ["2026-04-27T12:00:00Z"]),
        ("2026-05-01T00:00:00Z", 12, ["2026-04-27T18:00:00Z"]),
    ]
    assert all(c["mars:metadata"]["anoffset"] == 90 for c in covjson["coverages"])


def test_grid_anoffset_shifts_valid_time():
    points = [(50.7, 7.1, [1.0, 2.0]), (50.7, 7.2, [3.0, 4.0])]
    tree = forecast_separate_datetime_tree(points, T00, step=(6, 12), anoffset=(18,), type_="fu")
    cov = _encode("grid", tree)["coverages"][0]
    assert cov["domain"]["axes"]["t"]["values"] == ["2026-04-30T12:00:00Z", "2026-04-30T18:00:00Z"]
    assert cov["mars:metadata"]["anoffset"] == 18


def test_path_anoffset_shifts_valid_time():
    points = [(50.7, 7.1, [1.0]), (50.8, 7.2, [2.0])]
    tree = forecast_separate_datetime_tree(points, T00, step=(6,), anoffset=(18,), type_="fu")
    cov = _encode("path", tree)["coverages"][0]
    assert [v[0] for v in cov["domain"]["axes"]["composite"]["values"]] == [
        "2026-04-30T12:00:00Z",
        "2026-04-30T12:00:00Z",
    ]
    assert cov["mars:metadata"]["anoffset"] == 18


def test_verticalprofile_anoffset_shifts_valid_time_and_round_trips():
    # Layout (date, levelist, step, time) with one step and time: [L500, L850].
    tree = forecast_separate_datetime_tree(
        [(50.7, 7.1, [1.0, 2.0])], T12, step=(6,), anoffset=(30,), levels=(500, 850), param="130", type_="fu"
    )
    covjson = _encode("verticalprofile", tree)
    cov = covjson["coverages"][0]
    assert cov["domain"]["axes"]["t"]["values"] == ["2026-04-30T12:00:00Z"]
    assert cov["domain"]["axes"]["z"]["values"] == [500, 850]
    assert cov["ranges"]["t"]["values"] == [1.0, 2.0]
    assert cov["mars:metadata"]["anoffset"] == 30

    # The decoder rebuilds date + step - anoffset to place each coverage.
    ds = Covjsonkit().decode(covjson).to_xarray()
    assert ds["t"].values.ravel().tolist() == [1.0, 2.0]


def test_no_anoffset_leaves_valid_time_unchanged():
    tree = forecast_separate_datetime_tree([(50.7, 7.1, [1.0])], T00, step=(6,), type_="fu")
    cov = _encode("timeseries", tree)["coverages"][0]
    assert cov["domain"]["axes"]["t"]["values"] == ["2026-05-01T06:00:00Z"]
    assert "anoffset" not in cov["mars:metadata"]


@pytest.mark.parametrize("feature", ["timeseries", "position", "boundingbox", "grid", "path"])
def test_multiple_anoffsets_rejected_on_reforecast_path(feature):
    # Layout (anoffset, date, step, time): two anoffsets x one step.
    tree = forecast_separate_datetime_tree([(50.7, 7.1, [1.0, 2.0])], T00, step=(6,), anoffset=(18, 42), type_="fu")
    with pytest.raises(ValueError, match="single anoffset"):
        _encode(feature, tree)


def test_multiple_anoffsets_rejected_on_from_polytope_path():
    tree = chain(
        TensorIndexTree(),
        node("anoffset", (18, 42)),
        node("class", ("ce",)),
        node("date", (np.datetime64("2026-05-01T00:00:00"),)),
        node("param", ("240023",)),
        node("step", (6,)),
        node("stream", ("efas",)),
    )
    tip(tree).add_child(make_point(50.7, 7.1, [1.0]))
    with pytest.raises(ValueError, match="single anoffset"):
        Covjsonkit().encode("CoverageCollection", "boundingbox").from_polytope(tree)
