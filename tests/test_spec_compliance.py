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

    def test_legacy_month_xarray_equivalent(self):
        new_covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope_month(_month_tree())
        legacy_covjson = _to_legacy(new_covjson)

        new_ds = Covjsonkit().decode(new_covjson).to_xarray()
        legacy_ds = Covjsonkit().decode(legacy_covjson).to_xarray()

        assert new_ds.identical(legacy_ds)
