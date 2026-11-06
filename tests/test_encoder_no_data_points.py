"""Points with no data are emitted with null values, not dropped.

A requested point whose values are all ``None`` (e.g. missing in the FDB) used to be
silently removed from the output, which shifted every later point and lost its label.
Now every requested point yields a coverage, in request order, with ``null`` range
values; the label, position and metadata are kept. This holds for labelled and
unlabelled requests. Only a *date* with no data for any point is still dropped.
"""

import numpy as np
import pytest
from conftest import (
    assert_valid_covjson,
    chain,
    forecast_tree,
    month_tree,
    node,
    reforecast_branch,
    reforecast_separate_datetime_tree,
    reforecast_separate_datetime_vertical_tree,
    reforecast_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree
from test_encoder_area_labels import AREA_KINDS, DECODERS, area_tree
from test_encoder_area_labels import decoded as decoded_area
from test_encoder_labels import (
    decoded,
    encode,
    labels,
    points_of,
    summary,
    tagged,
    vp_tree,
)

from covjsonkit.api import Covjsonkit

POINTS = [(48.0, 11.0), (49.0, 11.5), (50.0, 12.0)]
NAMES = {POINTS[0]: "A", POINTS[1]: "B", POINTS[2]: "C"}


def make_null_point(lat, lon, result):
    """Like ``conftest.make_point`` but keeps ``None`` entries in the leaf result."""
    lat_node = node("latitude", (lat,))
    leaf = node("longitude", (lon,))
    leaf.result = [None if r is None else np.float64(r) for r in result]
    lat_node.add_child(leaf)
    return lat_node


def point_factory(labelled):
    """Tag the three points 0, 1, 2 in tree order, with labels A, B, C only if ``labelled``."""
    tags = {point: [(i, NAMES[point] if labelled else None)] for i, point in enumerate(POINTS)}
    return tagged(make_null_point, tags)


def three_points(n, null_middle=True):
    """Three points with ``n`` values each; the middle one has no data."""
    return [
        (*POINTS[0], [1.0 + i for i in range(n)]),
        (*POINTS[1], [None] * n if null_middle else [5.0 + i for i in range(n)]),
        (*POINTS[2], [9.0 + i for i in range(n)]),
    ]


def leaves(tree):
    """Leaf nodes of ``tree`` in tree order."""
    if not tree.children:
        return [tree]
    return [leaf for child in tree.children for leaf in leaves(child)]


def range_values(covjson):
    return [list(c["ranges"].values())[0]["values"] for c in covjson["coverages"]]


def check_middle_is_null(covjson, labelled, n_values, expect_coverages=3):
    """Three coverages in request order; the middle one is labelled (if requested) and all null."""
    assert_valid_covjson(covjson)
    assert len(covjson["coverages"]) == expect_coverages
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    values = range_values(covjson)
    assert values[1] == [None] * n_values
    assert all(v is not None for v in values[0] + values[2])
    return values


LABELLED_NAMES = ["A", "B", "C"]
LABELLED = pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "unlabelled"])


# -- PointSeries / Position, forecast --


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_forecast_null_point_kept(kind, labelled):
    tree = forecast_tree(three_points(2), step=(0, 6), point_factory=point_factory(labelled))
    covjson = encode(kind, tree)
    check_middle_is_null(covjson, labelled, 2)
    assert points_of(covjson) == [(48.0, 11.0), (49.0, 11.5), (50.0, 12.0)]
    # Position (metadata, step) of the null point is the same as for its neighbours
    middle = covjson["coverages"][1]
    assert middle["domain"]["axes"]["t"] == covjson["coverages"][0]["domain"]["axes"]["t"]
    assert middle["mars:metadata"]["Forecast date"] == covjson["coverages"][0]["mars:metadata"]["Forecast date"]


@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_forecast_null_first_and_last_point_kept(kind):
    points = [
        (*POINTS[0], [None, None]),
        (*POINTS[1], [3.0, 4.0]),
        (*POINTS[2], [None, None]),
    ]
    covjson = encode(kind, forecast_tree(points, step=(0, 6), point_factory=point_factory(True)))
    assert_valid_covjson(covjson)
    assert labels(covjson) == LABELLED_NAMES
    assert range_values(covjson) == [[None, None], [3.0, 4.0], [None, None]]


@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_forecast_partial_null_values_kept(kind):
    """A point with only some null values is unchanged (nulls stay in place)."""
    points = [(*POINTS[0], [1.0, None]), (*POINTS[1], [None, 4.0])]
    covjson = encode(kind, forecast_tree(points, step=(0, 6), point_factory=make_null_point))
    assert range_values(covjson) == [[1.0, None], [None, 4.0]]


@LABELLED
def test_forecast_null_point_after_shared_grid_point(labelled):
    """Several requested points snapped to one all-null grid point each get a null coverage."""
    tags = {POINTS[0]: [(0, "A" if labelled else None), (1, "B" if labelled else None)], POINTS[1]: [(2, None)]}
    points = [(*POINTS[0], [None]), (*POINTS[1], [7.0])]
    covjson = encode("pointseries", forecast_tree(points, point_factory=tagged(make_null_point, tags)))
    assert_valid_covjson(covjson)
    assert labels(covjson) == (["A", "B", None] if labelled else [None] * 3)
    assert range_values(covjson) == [[None], [None], [7.0]]


def test_forecast_untagged_null_point_kept():
    covjson = encode("pointseries", forecast_tree(three_points(2), step=(0, 6), point_factory=make_null_point))
    assert_valid_covjson(covjson)
    assert points_of(covjson) == [(48.0, 11.0), (49.0, 11.5), (50.0, 12.0)]
    assert range_values(covjson)[1] == [None, None]


# -- dates without any data are still dropped --


def two_dates_tree(factory, second_date_results):
    """Two forecast dates; ``second_date_results`` gives the results of the second date's three points."""
    dates = (np.datetime64("2025-01-01T00:00:00"), np.datetime64("2025-01-02T00:00:00"))
    tree = chain(TensorIndexTree(), node("class", ("od",)))
    root = tip(tree)
    for date, results in zip(dates, ([[1.0], [2.0], [3.0]], second_date_results)):
        branch = chain(
            node("date", (date,)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0,)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
        )
        parent = tip(branch)
        for (lat, lon), result in zip(POINTS, results):
            parent.add_child(factory(lat, lon, result))
        root.add_child(branch)
    return tree


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_date_without_any_data_dropped(kind, labelled):
    """The legitimate case behind the old ``dates[:-1]``: no point has data for a date."""
    tree = two_dates_tree(point_factory(labelled), [[None], [None], [None]])
    covjson = encode(kind, tree)
    assert_valid_covjson(covjson)
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    assert {c["mars:metadata"]["Forecast date"] for c in covjson["coverages"]} == {"2025-01-01T00:00:00Z"}
    assert range_values(covjson) == [[1.0], [2.0], [3.0]]


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_null_point_in_one_date_only(kind, labelled):
    """A point missing on one date keeps its coverage there; earlier dates' data is untouched."""
    tree = two_dates_tree(point_factory(labelled), [[10.0], [None], [30.0]])
    covjson = encode(kind, tree)
    assert_valid_covjson(covjson)
    assert labels(covjson) == (["A", "A", "B", "B", "C", "C"] if labelled else [None] * 6)
    assert range_values(covjson) == [[1.0], [10.0], [2.0], [None], [3.0], [30.0]]


@pytest.mark.parametrize("kind", ["pointseries", "position", "polygon"])
def test_no_data_for_any_point_raises(kind):
    tree = forecast_tree([(*POINTS[0], [None]), (*POINTS[1], [None])], point_factory=make_null_point)
    with pytest.raises(ValueError, match="No data was returned"):
        encode(kind, tree)


# -- step / month / reforecast paths --


def step_tree3(factory, middle):
    """Tree with a ``time`` axis (two times) for ``from_polytope_step``."""
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
        node("time", (np.timedelta64(0, "h"), np.timedelta64(6, "h"))),
        node("type", ("fc",)),
    )
    parent = tip(tree)
    for (lat, lon), result in zip(POINTS, ([1.0, 2.0], middle, [9.0, 8.0])):
        parent.add_child(factory(lat, lon, result))
    return tree


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_step_null_point_kept(kind, labelled):
    covjson = encode(kind, step_tree3(point_factory(labelled), [None, None]), "from_polytope_step")
    values = check_middle_is_null(covjson, labelled, 2)
    assert values == [[1.0, 2.0], [None, None], [9.0, 8.0]]


@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_step_date_without_any_data_dropped(kind):
    tree = step_tree3(make_null_point, [None, None])
    for leaf in leaves(tree):  # every point null: nothing to return
        leaf.result = [None, None]
    with pytest.raises(ValueError, match="No data was returned"):
        encode(kind, tree, "from_polytope_step")


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_month_null_point_kept(kind, labelled):
    tree = month_tree(three_points(2), years=(2020, 2021), point_factory=point_factory(labelled))
    covjson = encode(kind, tree, "from_polytope_month")
    values = check_middle_is_null(covjson, labelled, 2)
    assert values == [[1.0, 2.0], [None, None], [9.0, 10.0]]


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_reforecast_separate_datetime_null_point_kept(kind, labelled):
    hdates = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
    times = (np.timedelta64(0, "h"),)
    tree = reforecast_separate_datetime_tree(three_points(2), hdates, times, point_factory=point_factory(labelled))
    covjson = encode(kind, tree, "from_polytope_reforecast")
    assert_valid_covjson(covjson)
    assert covjson["coverages"], "expected coverages"
    got = [(c["mars:metadata"].get("label"), v) for c, v in zip(covjson["coverages"], range_values(covjson))]
    names = LABELLED_NAMES if labelled else [None] * 3
    if kind == "pointseries":
        # One series per point over both hdates
        assert got == [(names[0], [1.0, 2.0]), (names[1], [None, None]), (names[2], [9.0, 10.0])]
    else:
        # Position: one coverage per (point, reference datetime)
        assert [lbl for lbl, _ in got] == [n for n in names for _ in range(2)]
        assert [v for _, v in got] == [[1.0], [2.0], [None], [None], [9.0], [10.0]]


@LABELLED
def test_reforecast_hdate_without_any_data_dropped(labelled):
    """Separate-datetime reforecast: an hdate with no data for any point is dropped."""
    hdates = (np.datetime64("2025-07-14T00:00:00"), np.datetime64("2025-07-15T00:00:00"))
    times = (np.timedelta64(0, "h"),)
    points = [(*POINTS[0], [1.0, None]), (*POINTS[1], [None, None]), (*POINTS[2], [3.0, None])]
    tree = reforecast_separate_datetime_tree(points, hdates, times, point_factory=point_factory(labelled))
    covjson = encode("pointseries", tree, "from_polytope_reforecast")
    assert_valid_covjson(covjson)
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    assert range_values(covjson) == [[1.0], [None], [3.0]]


@LABELLED
def test_reforecast_vertical_profile_null_point_kept(labelled):
    hdates = (np.datetime64("2025-07-14T00:00:00"),)
    times = (np.timedelta64(0, "h"),)
    tree = reforecast_separate_datetime_vertical_tree(
        three_points(2), hdates, times, (500, 850), point_factory=point_factory(labelled)
    )
    covjson = encode("verticalprofile", tree, "from_polytope_reforecast")
    values = check_middle_is_null(covjson, labelled, 2)
    assert values == [[1.0, 2.0], [None, None], [9.0, 10.0]]


@LABELLED
def test_reforecast_legacy_hdate_branches_null_point_kept(labelled):
    """class=ce / efcl reanalysis path (hdate branches collapsed into one series per point)."""
    hdates = (np.datetime64("2024-03-01T00:00:00"), np.datetime64("2024-03-02T00:00:00"))
    branches = [reforecast_branch(h, three_points(1), point_factory=point_factory(labelled)) for h in hdates]
    covjson = encode("pointseries", reforecast_tree(branches), "from_polytope_reforecast")
    values = check_middle_is_null(covjson, labelled, 2)
    assert values == [[1.0, 1.0], [None, None], [9.0, 9.0]]


# -- VerticalProfile, forecast --


@LABELLED
def test_vertical_profile_null_point_kept(labelled):
    tree = vp_tree(three_points(2), point_factory(labelled))
    covjson = encode("verticalprofile", tree)
    values = check_middle_is_null(covjson, labelled, 2)
    assert values == [[1.0, 2.0], [None, None], [9.0, 10.0]]
    assert points_of(covjson) == [(48.0, 11.0), (49.0, 11.5), (50.0, 12.0)]


# -- Area encoders --


def null_row_tags(labelled):
    p = (0, "P" if labelled else None)
    return {"row48": {p}, "row49": {p}, "row50": {p}}


@LABELLED
@pytest.mark.parametrize("kind", AREA_KINDS)
def test_area_null_cells_kept(kind, labelled):
    tree = area_tree(null_row_tags(labelled), step=(0,))
    # Row 49 (the middle point) has no data
    leaves(tree)[1].result = [None]  # row 49: the middle point
    covjson = encode(kind, tree)
    assert_valid_covjson(covjson)
    (coverage,) = covjson["coverages"]
    assert coverage["mars:metadata"].get("label") == ("P" if labelled else None)
    points = [tuple(p[:2]) for p in coverage["domain"]["axes"]["composite"]["values"]]
    assert points == [(11.0, 48.0), (11.1, 48.0), (11.5, 49.0), (12.0, 50.0)]
    assert list(coverage["ranges"].values())[0]["values"] == [1.0, 2.0, None, 4.0]


# -- Grid / Path: a null cell keeps the array shape --


def test_grid_null_cell_kept():
    cells = [(48.0, 11.0, [1.0]), (48.0, 12.0, [None]), (50.0, 11.0, [3.0]), (50.0, 12.0, [4.0])]
    covjson = encode("grid", forecast_tree(cells, point_factory=make_null_point))
    assert_valid_covjson(covjson)
    (coverage,) = covjson["coverages"]
    assert coverage["domain"]["axes"]["x"]["values"] == [11.0, 12.0]
    assert list(coverage["ranges"].values())[0]["shape"] == [1, 2, 2]
    assert range_values(covjson) == [[1.0, None, 3.0, 4.0]]


def test_path_null_vertex_kept():
    covjson = encode("path", forecast_tree(three_points(1), point_factory=make_null_point))
    assert_valid_covjson(covjson)
    (coverage,) = covjson["coverages"]
    assert [v[1:] for v in coverage["domain"]["axes"]["composite"]["values"]] == [
        [11.0, 48.0],
        [11.5, 49.0],
        [12.0, 50.0],
    ]
    assert range_values(covjson) == [[1.0, None, 9.0]]


# -- Decoders: null values become NaN --


def nan_mask(ds):
    var = list(ds.data_vars)[0]
    assert ds[var].dtype == np.float64, "null values must give a float array with NaN, not an object array"
    return np.isnan(np.asarray(ds[var].values, dtype=float)).ravel().tolist()


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_decode_null_point_is_nan(kind, labelled):
    tree = forecast_tree(three_points(2), step=(0, 6), point_factory=point_factory(labelled))
    datasets = decoded(encode(kind, tree))
    assert [ds.attrs.get("label") for ds in datasets] == (LABELLED_NAMES if labelled else [None] * 3)
    assert [float(ds["latitude"].values[0]) for ds in datasets] == [48.0, 49.0, 50.0]
    assert [nan_mask(ds) for ds in datasets] == [[False, False], [True, True], [False, False]]
    assert summary(datasets)[0][3] == [1.0, 2.0]


@LABELLED
def test_decode_vertical_profile_null_point_is_nan(labelled):
    datasets = decoded(encode("verticalprofile", vp_tree(three_points(2), point_factory(labelled))))
    assert [ds.attrs.get("label") for ds in datasets] == (LABELLED_NAMES if labelled else [None] * 3)
    assert [nan_mask(ds) for ds in datasets] == [[False, False], [True, True], [False, False]]


@LABELLED
@pytest.mark.parametrize("decoder", DECODERS, ids=lambda d: d.__name__)
def test_decode_area_null_cell_is_nan(decoder, labelled):
    tree = area_tree(null_row_tags(labelled), step=(0,))
    leaves(tree)[1].result = [None]  # row 49: the middle point
    (ds,) = decoded_area(encode("polygon", tree), decoder)
    assert nan_mask(ds) == [False, False, True, False]


def test_decode_all_null_month_point_is_nan():
    tree = month_tree(three_points(2), years=(2020, 2021), point_factory=point_factory(True))
    datasets = decoded(encode("pointseries", tree, "from_polytope_month"))
    assert [nan_mask(ds) for ds in datasets] == [[False, False], [True, True], [False, False]]


def test_decode_via_api_round_trip_is_json_null():
    """The null survives serialisation as JSON ``null``."""
    import orjson

    encoder = Covjsonkit().encode("CoverageCollection", "pointseries")
    encoder.from_polytope(forecast_tree(three_points(2), step=(0, 6), point_factory=point_factory(True)))
    payload = orjson.loads(encoder.get_json())
    assert payload["coverages"][1]["ranges"][next(iter(payload["coverages"][1]["ranges"]))]["values"] == [None, None]


def test_decode_grid_null_cell_is_nan():
    cells = [(48.0, 11.0, [1.0]), (48.0, 12.0, [None]), (50.0, 11.0, [3.0]), (50.0, 12.0, [4.0])]
    ds = decoded(encode("grid", forecast_tree(cells, point_factory=make_null_point)))[0]
    assert nan_mask(ds) == [False, True, False, False]


def test_decode_path_null_vertex_is_nan():
    ds = decoded(encode("path", forecast_tree(three_points(1), point_factory=make_null_point)))[0]
    assert nan_mask(ds) == [False, True, False]


# -- Decoders leave non-numeric ranges alone --


def set_range(covjson, index, values):
    coverage = covjson["coverages"][index]
    next(iter(coverage["ranges"].values()))["values"] = values


def as_text(values):
    """Values as text (``None`` kept): xarray itself turns an all-string or mixed array into a ``str`` array."""
    return [None if v is None else str(v) for v in values]


def range_of(ds):
    """The dataset's first variable, asserted not to have been turned into floats, and its flat values."""
    array = ds[list(ds.data_vars)[0]]
    assert array.dtype.kind in "OU", f"non-numeric range was cast to {array.dtype}"
    return as_text(array.values.ravel().tolist())


@pytest.mark.parametrize("values", [["001", None], ["001", "002"], [1.0, "x"], [None, "x"]], ids=str)
def test_decode_string_range_not_cast_to_float(values):
    """Only numeric ranges become float/NaN: a string range, even one that looks numeric, stays as it is."""
    covjson = encode("pointseries", forecast_tree(three_points(2), step=(0, 6), point_factory=point_factory(True)))
    set_range(covjson, 0, values)
    datasets = decoded(covjson)
    assert range_of(datasets[0]) == as_text(values)
    # The neighbouring numeric points and the null point are still handled as before
    assert nan_mask(datasets[1]) == [True, True]
    assert nan_mask(datasets[2]) == [False, False]


@pytest.mark.parametrize("values", [["001", "002", None, "004"], [1.0, "x", None, 4.0]], ids=str)
@pytest.mark.parametrize("decoder", DECODERS, ids=lambda d: d.__name__)
def test_decode_area_string_range_not_cast_to_float(decoder, values):
    covjson = encode("polygon", area_tree(null_row_tags(True), step=(0,)))
    set_range(covjson, 0, values)
    (ds,) = decoded_area(covjson, decoder)
    assert range_of(ds) == as_text(values)


# -- Tags must match the point count --


def area_args(n_composite, n_levels=1):
    composite = [[11.0 + i, 48.0] for i in range(n_composite)]
    values = {"167": [float(i) for i in range(n_composite)]}
    return {}, {"composite": composite, "t": ["2025-01-01T00:00:00Z"]}, values, n_levels


@pytest.mark.parametrize("kind", AREA_KINDS)
class TestAddCoveragesByTagCount:
    def encoder(self, kind):
        return Covjsonkit().encode("CoverageCollection", kind)

    def test_untagged_entries_with_wrong_count_raise(self, kind):
        """Three composite points but one (untagged) entry: a mismatch, not an unlabelled fallback."""
        meta, coords, values, n_levels = area_args(3)
        with pytest.raises(ValueError, match="1:1"):
            self.encoder(kind).add_coverages_by_tag(meta, coords, values, [[]], n_levels)

    def test_tagged_entries_with_wrong_count_raise(self, kind):
        meta, coords, values, n_levels = area_args(3)
        with pytest.raises(ValueError, match="1:1"):
            self.encoder(kind).add_coverages_by_tag(meta, coords, values, [[(0, "P")], [(1, "Q")]], n_levels)

    def test_count_checked_per_level(self, kind):
        meta, coords, values, n_levels = area_args(6, n_levels=2)  # 3 points x 2 levels
        encoder = self.encoder(kind)
        with pytest.raises(ValueError, match="1:1"):
            encoder.add_coverages_by_tag(meta, coords, values, [[(0, "P")]] * 2, n_levels)
        encoder.add_coverages_by_tag(meta, coords, values, [[(0, "P")]] * 3, n_levels)
        assert len(encoder.covjson["coverages"]) == 1

    def test_values_length_mismatch_raises(self, kind):
        meta, coords, values, n_levels = area_args(3)
        values["167"] = values["167"][:2]
        with pytest.raises(ValueError, match="1:1"):
            self.encoder(kind).add_coverages_by_tag(meta, coords, values, [[]] * 3, n_levels)

    @pytest.mark.parametrize("tags", [None, []], ids=["none", "empty"])
    def test_no_tags_gives_one_unlabelled_coverage(self, kind, tags):
        meta, coords, values, n_levels = area_args(3)
        encoder = self.encoder(kind)
        encoder.add_coverages_by_tag(meta, coords, values, tags, n_levels)
        (coverage,) = encoder.covjson["coverages"]
        assert "label" not in coverage["mars:metadata"]
        assert len(coverage["domain"]["axes"]["composite"]["values"]) == 3

    def test_one_untagged_entry_per_point_gives_one_coverage(self, kind):
        meta, coords, values, n_levels = area_args(3)
        encoder = self.encoder(kind)
        encoder.add_coverages_by_tag(meta, coords, values, [[]] * 3, n_levels)
        (coverage,) = encoder.covjson["coverages"]
        assert len(coverage["domain"]["axes"]["composite"]["values"]) == 3


# -- Multi-date pruning in the step and month walkers --


def step_two_dates_tree(factory, results):
    """``date`` (two values) x ``time`` (two values); each point's result is [date][time] flattened."""
    tree = chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"), np.datetime64("2025-01-02T00:00:00"))),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("sfc",)),
        node("param", ("167",)),
        node("step", (0,)),
        node("stream", ("oper",)),
        node("time", (np.timedelta64(0, "h"), np.timedelta64(6, "h"))),
        node("type", ("fc",)),
    )
    parent = tip(tree)
    for (lat, lon), result in zip(POINTS, results):
        parent.add_child(factory(lat, lon, result))
    return tree


def t_axes(covjson):
    return [c["domain"]["axes"]["t"]["values"] for c in covjson["coverages"]]


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_step_empty_date_dropped_other_kept(kind, labelled):
    """The second date has no data for any point: only that date goes, all points stay."""
    tree = step_two_dates_tree(
        point_factory(labelled), [[1.0, 2.0, None, None], [3.0, 4.0, None, None], [5.0, 6.0, None, None]]
    )
    covjson = encode(kind, tree, "from_polytope_step")
    assert_valid_covjson(covjson)
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    assert range_values(covjson) == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert t_axes(covjson) == [["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]] * 3


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_step_point_empty_in_one_date_kept_with_nulls(kind, labelled):
    tree = step_two_dates_tree(
        point_factory(labelled), [[1.0, 2.0, 7.0, 8.0], [3.0, 4.0, None, None], [5.0, 6.0, 9.0, 10.0]]
    )
    covjson = encode(kind, tree, "from_polytope_step")
    assert_valid_covjson(covjson)
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    assert range_values(covjson) == [[1.0, 2.0, 7.0, 8.0], [3.0, 4.0, None, None], [5.0, 6.0, 9.0, 10.0]]
    assert len(t_axes(covjson)[0]) == 4


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_month_empty_date_dropped_other_kept(kind, labelled):
    tree = month_tree(
        [(*POINTS[0], [1.0, None]), (*POINTS[1], [2.0, None]), (*POINTS[2], [3.0, None])],
        years=(2020, 2021),
        point_factory=point_factory(labelled),
    )
    covjson = encode(kind, tree, "from_polytope_month")
    assert_valid_covjson(covjson)
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    assert range_values(covjson) == [[1.0], [2.0], [3.0]]
    assert t_axes(covjson) == [["2020-01-01T00:00:00Z"]] * 3


@LABELLED
@pytest.mark.parametrize("kind", ["pointseries", "position"])
def test_month_point_empty_in_one_date_kept_with_nulls(kind, labelled):
    tree = month_tree(
        [(*POINTS[0], [1.0, 4.0]), (*POINTS[1], [2.0, None]), (*POINTS[2], [3.0, 6.0])],
        years=(2020, 2021),
        point_factory=point_factory(labelled),
    )
    covjson = encode(kind, tree, "from_polytope_month")
    assert_valid_covjson(covjson)
    assert labels(covjson) == (LABELLED_NAMES if labelled else [None] * 3)
    assert range_values(covjson) == [[1.0, 4.0], [2.0, None], [3.0, 6.0]]
    assert t_axes(covjson) == [["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"]] * 3
