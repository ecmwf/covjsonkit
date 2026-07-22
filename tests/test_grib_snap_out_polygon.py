"""Polygon-vs-MARS snap-out comparison at the prime meridian.

The BoundingBox path fails to wrap across longitude 0° (see
`test_grib_snap_out_comparison.py::crosses-prime-meridian`).  This test
checks whether the **polygon** feature handles the same geometry
correctly, using two equivalent shapes:

  1. Signed longitudes  — vertices in [-0.1, +0.1], unambiguous.
  2. Wrap longitudes    — vertices at 359.9 and 0.1, matching the bbox
                          form; tests whether the polygon walker treats
                          the 359.9→0.1 edge as a short arc across 0°
                          or a long arc across 180°.

Reference: MARS `area=1.0/359.9/-1.0/0.1` returns 84 points in a narrow
~0.14°-wide band straddling the meridian.

Run:
  RUN_INTEGRATION_TESTS=1 pytest tests/test_grib_snap_out_polygon.py -v -s
"""

import json
import os
import tempfile

import pytest

eccodes = pytest.importorskip("eccodes", reason="eccodes required")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Integration test — set RUN_INTEGRATION_TESTS=1 to run",
)


BASE_MARS_KEYS = {
    "class": "od",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "param": "2t",
    "step": "0",
    "date": "-1",
    "time": "0000",
}
COLLECTION = "ecmwf-mars"

# MARS `area=1.0/359.9/-1.0/0.1` returns this many points (empirical).
MARS_WRAP_POINT_COUNT = 84
POINT_MATCH_TOL = 0.02  # degrees


# Polygon geometry equivalent to bbox N=1, W=-0.1, S=-1, E=0.1 — the
# meridian-straddling region expressed with signed longitudes (the
# canonical polytope convention; wrap longitudes are invalid input).
POLYGON_CASES = [
    pytest.param(
        [[1.0, -0.1], [1.0, 0.1], [-1.0, 0.1], [-1.0, -0.1], [1.0, -0.1]],
        id="signed-lons-straddling-zero",
    ),
]


def _retrieve_polygon_covjson(shape):
    from polytope.api import Client

    req = {
        **BASE_MARS_KEYS,
        "feature": {"type": "polygon", "shape": shape},
    }
    print(f"  Polygon request: {json.dumps(req)}")
    with tempfile.NamedTemporaryFile(suffix=".covjson", delete=False, mode="w") as tmp:
        p = tmp.name
    try:
        Client(quiet=True).retrieve(COLLECTION, req, output_file=p)
        with open(p) as f:
            return json.load(f)
    finally:
        if os.path.exists(p):
            os.unlink(p)


def _retrieve_mars_area_points(bbox_area):
    """Retrieve MARS area and return (lat, lon) point list + count."""
    from polytope.api import Client

    req = {**BASE_MARS_KEYS, "area": bbox_area}
    print(f"  MARS area request: {json.dumps(req)}")
    with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
        p = tmp.name
    try:
        Client(quiet=True).retrieve(COLLECTION, req, output_file=p, pointer=False)
        with open(p, "rb") as f:
            gid = eccodes.codes_grib_new_from_file(f)
            try:
                npoints = eccodes.codes_get(gid, "numberOfDataPoints")
                pts = None
                try:
                    it = eccodes.codes_grib_iterator_new(gid, 0)
                    pts = []
                    try:
                        while True:
                            r = eccodes.codes_grib_iterator_next(it)
                            if not r:
                                break
                            pts.append((r[0], r[1]))
                    finally:
                        eccodes.codes_grib_iterator_delete(it)
                except Exception as e:
                    print(f"  (MARS iterator failed: {e}; count-only)")
                return pts, npoints
            finally:
                eccodes.codes_release(gid)
    finally:
        if os.path.exists(p):
            os.unlink(p)


def _covjson_points(covjson):
    cov = covjson["coverages"][0]
    coords = cov["domain"]["axes"]["composite"]["values"]
    return [(c[0], c[1]) for c in coords]


def _normalise_lon(lon):
    """Map lon into (-180, 180] for comparison."""
    x = ((lon + 180.0) % 360.0) - 180.0
    return x if x != -180.0 else 180.0


def _nearest_match(target, candidates, tol):
    best_d = float("inf")
    best = None
    for c in candidates:
        d = max(abs(target[0] - c[0]), abs(target[1] - c[1]))
        if d < best_d:
            best_d = d
            best = c
    return (best, best_d) if best_d <= tol else (None, best_d)


@pytest.mark.parametrize("shape", POLYGON_CASES)
def test_polygon_wrap_matches_mars_area(shape):
    """Polygon equivalents of the failing bbox should match MARS 84-pt wrap."""
    print(f"\n{'=' * 70}\n  Polygon shape: {shape}\n{'=' * 70}")

    mars_pts, mars_npoints = _retrieve_mars_area_points("1.0/359.9/-1.0/0.1")
    covjson = _retrieve_polygon_covjson(shape)
    poly_pts = _covjson_points(covjson)

    print(f"  MARS numberOfDataPoints:   {mars_npoints}")
    print(f"  Polygon coverage count:    {len(poly_pts)}")

    if mars_pts is not None:
        mars_norm = [(lat, _normalise_lon(lon)) for lat, lon in mars_pts]
        poly_norm = [(lat, _normalise_lon(lon)) for lat, lon in poly_pts]
        print(
            f"  MARS lon range (norm):     " f"[{min(l for _, l in mars_norm):.6f}, {max(l for _, l in mars_norm):.6f}]"
        )
        print(
            f"  Polygon lon range (norm):  " f"[{min(l for _, l in poly_norm):.6f}, {max(l for _, l in poly_norm):.6f}]"
        )

        # Pair by nearest neighbour on normalised coords
        matched = 0
        remaining = list(poly_norm)
        unmatched_mars = []
        max_delta = 0.0
        for mp in mars_norm:
            m, d = _nearest_match(mp, remaining, POINT_MATCH_TOL)
            if m is None:
                unmatched_mars.append(mp)
            else:
                matched += 1
                max_delta = max(max_delta, d)
                remaining.remove(m)

        print(f"  Matched pairs:             {matched}")
        print(f"  Unmatched in MARS:         {len(unmatched_mars)}")
        print(f"  Unmatched in Polygon:      {len(remaining)}")
        print(f"  Max per-pair Δ:            {max_delta:.6f}°")
        if unmatched_mars:
            print(f"    MARS-only sample:      {unmatched_mars[:3]}")
        if remaining:
            print(f"    Polygon-only sample:   {remaining[:3]}")

        assert len(unmatched_mars) == 0, f"{len(unmatched_mars)} MARS points not covered by polygon"
        assert len(remaining) == 0, f"{len(remaining)} polygon points not covered by MARS wrap area"
    else:
        # Fallback: count-only assertion
        assert len(poly_pts) == mars_npoints, f"Point count mismatch: polygon={len(poly_pts)} vs MARS={mars_npoints}"
