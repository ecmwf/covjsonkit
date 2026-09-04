"""Spec-compliance tests for the TimeSeries / PointSeries feature (issues #122, #129).

Two guarantees are covered here:

1. Encoder output validates against the OGC CoverageJSON pydantic model
   (covjson-pydantic). Validation is a TEST-ONLY guard -- it is never wired into
   the library's runtime output path.

2. Decoders remain backwards-compatible: they must read both spec-compliant
   coverages (x/y/z axes, split referencing) and legacy coverages
   (latitude/longitude/levelist axes, single GeographicCRS), producing
   equivalent results.
"""

import copy
from datetime import timedelta

import numpy as np
import pytest
from conftest import assert_valid_covjson, chain, make_leaf, make_point, node
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


def _surface_forecast_tree():
    leaf = make_leaf(11.0, [264.931, 263.831])
    return chain(
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
        node("latitude", (48.0,)),
        leaf,
    )


def _level_forecast_tree():
    leaf = make_leaf(11.0, [264.931, 263.831])
    return chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("pl",)),
        node("levelist", (500,)),
        node("param", ("129",)),
        node("step", (0, 6)),
        node("stream", ("oper",)),
        node("type", ("fc",)),
        node("latitude", (48.0,)),
        leaf,
    )


def _month_tree():
    return chain(
        TensorIndexTree(),
        node("class", ("d1",)),
        node("stream", ("clmn",)),
        node("levtype", ("sfc",)),
        node("number", (0,)),
        node("year", (2020,)),
        node("month", (2, 3)),
        node("param", ("167",)),
        make_point(48.0, 11.0, [300.0, 301.0]),
    )


def _step_tree():
    return chain(
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
        make_point(48.0, 11.0, [264.0, 263.0]),
    )


def _vertical_profile_tree():
    return chain(
        TensorIndexTree(),
        node("class", ("od",)),
        node("date", (np.datetime64("2025-01-01T00:00:00"),)),
        node("domain", ("g",)),
        node("expver", ("0001",)),
        node("levtype", ("pl",)),
        node("param", ("130",)),
        node("step", (0,)),
        node("stream", ("oper",)),
        node("type", ("an",)),
        node("levelist", (1000, 850, 500)),
        make_point(48.0, 11.0, [290.1, 280.2, 250.3]),
    )


class TestSpecCompliantOutputValidates:
    """Encoder output must validate against covjson-pydantic (test-only guard)."""

    def test_surface_forecast_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(_surface_forecast_tree())
        assert_valid_covjson(covjson)

    def test_level_forecast_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(_level_forecast_tree())
        assert_valid_covjson(covjson)

    def test_month_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_month(_month_tree())
        assert_valid_covjson(covjson)

    def test_step_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_step(_step_tree())
        assert_valid_covjson(covjson)


def _to_legacy(covjson):
    """Rewrite a spec-compliant CoverageCollection into the pre-#122 legacy shape:
    x->longitude, y->latitude, z->levelist axes, and a single combined
    GeographicCRS referencing block. Used to exercise decoder backwards-compat.
    """
    legacy = copy.deepcopy(covjson)

    rename = {"x": "longitude", "y": "latitude", "z": "levelist"}
    for cov in legacy["coverages"]:
        axes = cov["domain"]["axes"]
        new_axes = {}
        for name, val in axes.items():
            new_axes[rename.get(name, name)] = val
        cov["domain"]["axes"] = new_axes

    coords = ["latitude", "longitude"]
    if any("levelist" in cov["domain"]["axes"] for cov in legacy["coverages"]):
        coords.append("levelist")
    legacy["referencing"] = [
        {
            "coordinates": coords,
            "system": {
                "type": "GeographicCRS",
                "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
            },
        }
    ]
    return legacy


class TestDecoderBackwardsCompatibility:
    """Decoders must read both new (x/y/z) and legacy (lat/lon/levelist) coverages."""

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_legacy_and_new_geojson_equivalent(self, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree_factory())
        legacy_covjson = _to_legacy(new_covjson)

        new_gj = Covjsonkit().decode(new_covjson).to_geojson()
        legacy_gj = Covjsonkit().decode(legacy_covjson).to_geojson()

        assert legacy_gj == new_gj

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_legacy_and_new_xarray_equivalent(self, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree_factory())
        legacy_covjson = _to_legacy(new_covjson)

        new_ds = Covjsonkit().decode(new_covjson).to_xarray()
        legacy_ds = Covjsonkit().decode(legacy_covjson).to_xarray()

        assert new_ds.identical(legacy_ds)

        # Prove the decoder read the legacy coverage by axis NAME (not position):
        # the resulting dataset must carry latitude/longitude coords, and levelist
        # when the source had a z axis.
        assert "latitude" in legacy_ds.coords
        assert "longitude" in legacy_ds.coords
        has_z = any("z" in cov["domain"]["axes"] for cov in new_covjson["coverages"])
        assert ("levelist" in legacy_ds.coords) == has_z

    def test_legacy_month_xarray_equivalent(self):
        new_covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_month(_month_tree())
        legacy_covjson = _to_legacy(new_covjson)

        new_ds = Covjsonkit().decode(new_covjson).to_xarray()
        legacy_ds = Covjsonkit().decode(legacy_covjson).to_xarray()

        assert new_ds.identical(legacy_ds)


class TestPositionSpecCompliance:
    """Position (PointSeries) encoder output validates and decodes with back-compat."""

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_position_output_validates(self, tree_factory):
        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope(tree_factory())
        assert_valid_covjson(covjson)

    def test_position_month_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope_month(_month_tree())
        assert_valid_covjson(covjson)

    def test_position_step_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope_step(_step_tree())
        assert_valid_covjson(covjson)

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_position_legacy_and_new_geojson_equivalent(self, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope(tree_factory())
        legacy_covjson = _to_legacy(new_covjson)

        new_gj = Covjsonkit().decode(new_covjson).to_geojson()
        legacy_gj = Covjsonkit().decode(legacy_covjson).to_geojson()

        assert legacy_gj == new_gj

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_position_legacy_and_new_xarray_equivalent(self, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", "Position").from_polytope(tree_factory())
        legacy_covjson = _to_legacy(new_covjson)

        new_ds = Covjsonkit().decode(new_covjson).to_xarray()
        legacy_ds = Covjsonkit().decode(legacy_covjson).to_xarray()

        assert new_ds.identical(legacy_ds)

        assert "latitude" in legacy_ds.coords
        assert "longitude" in legacy_ds.coords
        has_z = any("z" in cov["domain"]["axes"] for cov in new_covjson["coverages"])
        assert ("levelist" in legacy_ds.coords) == has_z


class TestVerticalProfileSpecCompliance:
    """VerticalProfile encoder output validates and decodes with back-compat."""

    def test_vertical_profile_output_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope(_vertical_profile_tree())
        assert_valid_covjson(covjson)

    def test_vertical_profile_legacy_and_new_geojson_equivalent(self):
        new_covjson = (
            Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope(_vertical_profile_tree())
        )
        legacy_covjson = _to_legacy(new_covjson)

        new_gj = Covjsonkit().decode(new_covjson).to_geojson()
        legacy_gj = Covjsonkit().decode(legacy_covjson).to_geojson()

        assert legacy_gj == new_gj

        # The vertical (z) axis must survive into the GeoJSON geometry as the
        # third coordinate, and lon/lat must be in the correct [lon, lat, z] order.
        first = new_gj["features"][0]["geometry"]["coordinates"]
        assert first == [11.0, 48.0, 1000]

    def test_vertical_profile_legacy_and_new_xarray_equivalent(self):
        new_covjson = (
            Covjsonkit().encode("CoverageCollection", "VerticalProfile").from_polytope(_vertical_profile_tree())
        )
        legacy_covjson = _to_legacy(new_covjson)

        new_ds = Covjsonkit().decode(new_covjson).to_xarray()
        legacy_ds = Covjsonkit().decode(legacy_covjson).to_xarray()

        assert new_ds.identical(legacy_ds)

        # Decoder read legacy coverage by axis NAME: lat/lon/levelist coords present.
        assert "latitude" in legacy_ds.coords
        assert "longitude" in legacy_ds.coords
        assert "levelist" in legacy_ds.coords


def _to_legacy_composite(covjson):
    """Rewrite a spec-compliant MultiPoint (composite-axis) CoverageCollection
    into the pre-#129 legacy shape: composite ``coordinates`` labelled
    latitude/longitude[/levelist] with tuple values reordered to
    [lat, lon[, level]], plus a single combined GeographicCRS referencing block.
    Used to exercise composite decoder backwards-compat.
    """
    legacy = copy.deepcopy(covjson)

    first_labels = legacy["coverages"][0]["domain"]["axes"]["composite"]["coordinates"]

    def _idx(labels, names):
        for name in names:
            if name in labels:
                return labels.index(name)
        return None

    include_z = _idx(first_labels, ("z", "levelist")) is not None

    for cov in legacy["coverages"]:
        composite = cov["domain"]["axes"]["composite"]
        labels = composite["coordinates"]
        x_idx = _idx(labels, ("x", "longitude"))
        y_idx = _idx(labels, ("y", "latitude"))
        z_idx = _idx(labels, ("z", "levelist"))
        new_values = []
        for tup in composite["values"]:
            entry = [tup[y_idx], tup[x_idx]]
            if z_idx is not None:
                entry.append(tup[z_idx])
            new_values.append(entry)
        composite["values"] = new_values
        composite["coordinates"] = ["latitude", "longitude", "levelist"] if include_z else ["latitude", "longitude"]

    coords = ["latitude", "longitude"]
    if include_z:
        coords.append("levelist")
    legacy["referencing"] = [
        {
            "coordinates": coords,
            "system": {
                "type": "GeographicCRS",
                "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
            },
        }
    ]
    return legacy


class TestBoundingBoxSpecCompliance:
    """BoundingBox (MultiPoint) encoder output validates and decodes with back-compat."""

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_bounding_box_output_validates(self, tree_factory):
        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(tree_factory())
        assert_valid_covjson(covjson)

    def test_bounding_box_month_validates(self):
        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope_month(_month_tree())
        assert_valid_covjson(covjson)

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_bounding_box_legacy_and_new_geojson_equivalent(self, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(tree_factory())
        legacy_covjson = _to_legacy_composite(new_covjson)

        new_gj = Covjsonkit().decode(new_covjson).to_geojson()
        legacy_gj = Covjsonkit().decode(legacy_covjson).to_geojson()

        assert legacy_gj == new_gj

    def test_bounding_box_geojson_lonlat_order(self):
        # Surface data: geometry must be [lon, lat] in the correct order.
        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(_surface_forecast_tree())
        gj = Covjsonkit().decode(covjson).to_geojson()
        assert gj["features"][0]["geometry"]["coordinates"] == [11.0, 48.0]

    def test_bounding_box_geojson_lonlat_z_order(self):
        # Level data: geometry must be [lon, lat, level] in the correct order.
        covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(_level_forecast_tree())
        gj = Covjsonkit().decode(covjson).to_geojson()
        assert gj["features"][0]["geometry"]["coordinates"] == [11.0, 48.0, 500]

    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_bounding_box_legacy_and_new_xarray_equivalent(self, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", "BoundingBox").from_polytope(tree_factory())
        legacy_covjson = _to_legacy_composite(new_covjson)

        new_ds = Covjsonkit().decode(new_covjson).to_xarray()
        legacy_ds = Covjsonkit().decode(legacy_covjson).to_xarray()

        assert new_ds.identical(legacy_ds)

        # Decoder read the composite by coordinate LABEL, not position: the
        # resulting dataset carries latitude/longitude, plus levelist when the
        # source had a z component.
        assert "latitude" in legacy_ds.coords
        assert "longitude" in legacy_ds.coords
        has_z = "z" in new_covjson["coverages"][0]["domain"]["axes"]["composite"]["coordinates"]
        assert ("levelist" in legacy_ds.coords) == has_z


class TestCompositeFeatureSpecCompliance:
    """Circle/Frame/Shapefile/Wkt (MultiPoint composite-axis) encoders must emit
    spec-compliant output and decode with composite backwards-compat."""

    FEATURES = ["Circle", "Frame", "Shapefile", "Polygon"]

    @pytest.mark.parametrize("feature", FEATURES)
    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_output_validates(self, feature, tree_factory):
        covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope(tree_factory())
        assert_valid_covjson(covjson)

    @pytest.mark.parametrize("feature", FEATURES)
    def test_geojson_lonlat_order(self, feature):
        covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope(_surface_forecast_tree())
        gj = Covjsonkit().decode(covjson).to_geojson()
        assert gj["features"][0]["geometry"]["coordinates"] == [11.0, 48.0]

    @pytest.mark.parametrize("feature", FEATURES)
    def test_geojson_lonlat_z_order(self, feature):
        covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope(_level_forecast_tree())
        gj = Covjsonkit().decode(covjson).to_geojson()
        assert gj["features"][0]["geometry"]["coordinates"] == [11.0, 48.0, 500]

    @pytest.mark.parametrize("feature", FEATURES)
    @pytest.mark.parametrize("tree_factory", [_surface_forecast_tree, _level_forecast_tree])
    def test_legacy_and_new_geojson_equivalent(self, feature, tree_factory):
        new_covjson = Covjsonkit().encode("CoverageCollection", feature).from_polytope(tree_factory())
        legacy_covjson = _to_legacy_composite(new_covjson)

        new_gj = Covjsonkit().decode(new_covjson).to_geojson()
        legacy_gj = Covjsonkit().decode(legacy_covjson).to_geojson()

        assert legacy_gj == new_gj
