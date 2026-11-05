"""Tests for labels on MultiPoint (area) features: polygon (Wkt), bounding box, circle, frame, shapefile.

polytope-mars tags every requested area as ``(index, label)``. The MultiPoint encoders emit one
coverage per area per (date, number, step), in request order, holding only that area's points;
a point inside several areas appears in each. ``mars:metadata.label`` is set only when a label was
requested, and trees without tags give the same output as before.
"""

import numpy as np
import pytest
from conftest import (
    chain,
    make_point,
    month_tree,
    node,
    reforecast_separate_datetime_tree,
    tip,
)
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit
from covjsonkit.decoder import BoundingBox as BoundingBoxDecoder
from covjsonkit.decoder import Circle as CircleDecoder
from covjsonkit.decoder import Frame as FrameDecoder
from covjsonkit.decoder import Shapefile as ShapefileDecoder
from covjsonkit.decoder import Wkt as WktDecoder
from covjsonkit.encoder.encoder import group_points_by_tags

P = (0, "P")
Q = (1, "Q")

AREA_KINDS = ["polygon", "boundingbox", "circle", "frame", "shapefile"]
MONTH_KINDS = ["polygon", "boundingbox", "circle"]


def area_tree(tags, step=(0,), levels=None):
    """Forecast tree with four grid points in two latitude rows.

    Row 48.0 has a compressed longitude leaf with two cells (11.0, 11.1), both inside P.
    Row 49.0 has a cell inside both P and Q (11.5). Row 50.0 has a cell inside Q only (12.0).
    ``tags`` maps each leaf to its tags, e.g. ``{"row48": {P}, "row49": {P, Q}, "row50": {Q}}``.
    Results are ordered [level][step][cell].
    """
    n_levels = len(levels) if levels else 1
    n_steps = len(step)

    def result(base, n_cells):
        return [base + 100 * lv + 10 * s + c for lv in range(n_levels) for s in range(n_steps) for c in range(n_cells)]

    rows = [
        ("row48", 48.0, (11.0, 11.1), 1.0),
        ("row49", 49.0, (11.5,), 3.0),
        ("row50", 50.0, (12.0,), 4.0),
    ]
    axes = [
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
    ]
    if levels:
        axes.append(node("levelist", levels))
    axes += [
        node("levtype", ("pl" if levels else "sfc",)),
        node("param", ("167",)),
        node("step", step),
        node("stream", ("oper",)),
        node("type", ("fc",)),
    ]
    tree = chain(TensorIndexTree(), *axes)
    parent = tip(tree)
    for name, lat, lons, base in rows:
        lat_node = node("latitude", (lat,))
        leaf = node("longitude", lons)
        leaf.result = [np.float64(r) for r in result(base, len(lons))]
        leaf.tags.update(tags.get(name, ()))
        lat_node.tags.update(tags.get(name, ()))
        lat_node.add_child(leaf)
        parent.add_child(lat_node)
    return tree


TAGS = {"row48": {P}, "row49": {P, Q}, "row50": {Q}}


def encode(kind, tree, method="from_polytope"):
    return getattr(Covjsonkit().encode("CoverageCollection", kind), method)(tree)


def summary(covjson):
    out = []
    for c in covjson["coverages"]:
        points = [tuple(p[:2]) for p in c["domain"]["axes"]["composite"]["values"]]
        values = list(c["ranges"].values())[0]["values"]
        out.append((c["mars:metadata"].get("label"), c["mars:metadata"].get("step"), points, values))
    return out


# -- Helper --


class TestGroupPointsByTags:
    def test_groups_in_request_order_with_shared_points(self):
        tags = [[(1, "Q")], [(0, "P"), (1, "Q")], [(0, "P")]]
        assert group_points_by_tags(tags, 3) == [(0, "P", [1, 2]), (1, "Q", [0, 1])]

    def test_untagged_is_one_group(self):
        assert group_points_by_tags([[], [], []], 3) == [(None, None, [0, 1, 2])]
        assert group_points_by_tags(None, 2) == [(None, None, [0, 1])]

    def test_untagged_points_last(self):
        assert group_points_by_tags([[], [(0, "P")]], 2) == [(0, "P", [1]), (None, None, [0])]


# -- Encoders --


@pytest.mark.parametrize("kind", AREA_KINDS)
class TestAreaEncoders:
    def test_one_coverage_per_area(self, kind):
        covjson = encode(kind, area_tree(TAGS))
        assert summary(covjson) == [
            ("P", 0, [(48.0, 11.0), (48.0, 11.1), (49.0, 11.5)], [1.0, 2.0, 3.0]),
            ("Q", 0, [(49.0, 11.5), (50.0, 12.0)], [3.0, 4.0]),
        ]

    def test_request_order_then_steps(self, kind):
        covjson = encode(kind, area_tree(TAGS, step=(0, 6)))
        assert [(label, step) for label, step, _, _ in summary(covjson)] == [("P", 0), ("P", 6), ("Q", 0), ("Q", 6)]
        # Second step of P: row48 cells [11, 12], row49 [13]
        assert summary(covjson)[1][3] == [11.0, 12.0, 13.0]

    def test_unlabelled_same_output(self, kind):
        unlabelled = {"row48": {(0, None)}, "row49": {(0, None), (1, None)}, "row50": {(1, None)}}
        labelled = summary(encode(kind, area_tree(TAGS)))
        plain = summary(encode(kind, area_tree(unlabelled)))
        assert [(None, *rest) for _, *rest in labelled] == plain
        assert all("label" not in c["mars:metadata"] for c in encode(kind, area_tree(unlabelled))["coverages"])

    def test_untagged_unchanged(self, kind):
        assert summary(encode(kind, area_tree({}))) == [
            (None, 0, [(48.0, 11.0), (48.0, 11.1), (49.0, 11.5), (50.0, 12.0)], [1.0, 2.0, 3.0, 4.0]),
        ]


@pytest.mark.parametrize("kind", ["polygon", "boundingbox", "circle"])
def test_levels_split_per_area(kind):
    covjson = encode(kind, area_tree(TAGS, levels=(500, 850)))
    p, q = covjson["coverages"]
    assert p["mars:metadata"]["label"] == "P" and q["mars:metadata"]["label"] == "Q"
    # Composite is level-major: each area's points repeated for each level
    assert [tuple(v) for v in q["domain"]["axes"]["composite"]["values"]] == [
        (49.0, 11.5, 500),
        (50.0, 12.0, 500),
        (49.0, 11.5, 850),
        (50.0, 12.0, 850),
    ]
    assert list(q["ranges"].values())[0]["values"] == [3.0, 4.0, 103.0, 104.0]


@pytest.mark.parametrize("kind", MONTH_KINDS)
def test_month(kind):
    def factory(lat, lon, result):
        point = make_point(lat, lon, result)
        tags = {48.0: {P}, 49.0: {P, Q}, 50.0: {Q}}[lat]
        point.tags.update(tags)
        point.children[0].tags.update(tags)
        return point

    tree = month_tree(
        [(48.0, 11.0, [1.0, 2.0]), (49.0, 11.5, [3.0, 4.0]), (50.0, 12.0, [5.0, 6.0])], point_factory=factory
    )
    covjson = encode(kind, tree, "from_polytope_month")
    labels = [c["mars:metadata"].get("label") for c in covjson["coverages"]]
    assert labels == ["P", "P", "Q", "Q"]  # two months each
    first_q = covjson["coverages"][2]
    assert [tuple(v[:2]) for v in first_q["domain"]["axes"]["composite"]["values"]] == [(49.0, 11.5), (50.0, 12.0)]


def test_reforecast_separate_datetime_split_per_area():
    def factory(lat, lon, result):
        point = make_point(lat, lon, result)
        point.children[0].tags.update({48.0: {P}, 50.0: {P, Q}}[lat])
        return point

    hdates = (np.datetime64("2025-07-14T00:00:00"),)
    times = (np.timedelta64(0, "h"),)
    tree = reforecast_separate_datetime_tree(
        [(48.0, 11.0, [1.0]), (50.0, 12.0, [2.0])], hdates, times, point_factory=factory
    )
    covjson = encode("polygon", tree, "from_polytope_reforecast")
    assert [(lbl, pts, vals) for lbl, _, pts, vals in summary(covjson)] == [
        ("P", [(48.0, 11.0), (50.0, 12.0)], [1.0, 2.0]),
        ("Q", [(50.0, 12.0)], [2.0]),
    ]
    assert all("__tags__" not in c["mars:metadata"] for c in covjson["coverages"])


# -- Decoders --


def decoded(covjson, decoder=None):
    if decoder is None:
        ds = Covjsonkit().decode(covjson).to_xarray()
    else:
        ds = decoder(covjson).to_xarray()
    return ds if isinstance(ds, list) else [ds]


DECODERS = [
    BoundingBoxDecoder.BoundingBox,
    CircleDecoder.Circle,
    FrameDecoder.Frame,
    ShapefileDecoder.Shapefile,
    WktDecoder.Wkt,
]


@pytest.mark.parametrize("decoder", DECODERS, ids=lambda d: d.__name__)
def test_decoder_one_dataset_per_area(decoder):
    datasets = decoded(encode("polygon", area_tree(TAGS, step=(0, 6))), decoder)
    assert [ds.attrs.get("label") for ds in datasets] == ["P", "Q"]
    assert [ds.sizes["points"] for ds in datasets] == [3, 2]


@pytest.mark.parametrize("decoder", DECODERS, ids=lambda d: d.__name__)
def test_decoder_unlabelled_areas_kept_apart(decoder):
    unlabelled = {"row48": {(0, None)}, "row49": {(0, None), (1, None)}, "row50": {(1, None)}}
    datasets = decoded(encode("polygon", area_tree(unlabelled, step=(0, 6))), decoder)
    assert [ds.sizes["points"] for ds in datasets] == [3, 2]
    assert all("label" not in ds.attrs for ds in datasets)


@pytest.mark.parametrize("decoder", DECODERS, ids=lambda d: d.__name__)
def test_decoder_untagged_single_dataset(decoder):
    covjson = encode("polygon", area_tree({}, step=(0, 6)))
    datasets = decoded(covjson, decoder)
    assert len(datasets) == 1
    assert datasets[0].identical(decoder(covjson)._to_xarray_single())


def test_default_decode_route_values():
    """Covjsonkit().decode routes MultiPoint to the BoundingBox decoder; check the values per area."""
    p, q = decoded(encode("polygon", area_tree(TAGS, step=(0, 6))))
    var = list(q.data_vars)[0]
    assert np.asarray(q[var].values).ravel().tolist() == [3.0, 4.0, 13.0, 14.0]
    assert q["latitude"].values.tolist() == [49.0, 50.0]
