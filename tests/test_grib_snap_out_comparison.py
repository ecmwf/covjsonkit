"""Snap-out comparison: polytope BoundingBox vs MARS `area`.

Both paths take an N/W/S/E box and return a subset of the ECMWF reduced-gg
grid.  They should agree on WHICH grid points fall inside the box, but they
use different snap-out conventions:

  * MARS `area` — eckit-geo `BoundingBox` crop applied server-side, then
    eccodes writes the exact first/last lat/lon corners it kept.
  * Polytope BoundingBox — floor-truncated coordinate walk over the
    underlying datacube axes; returns the coverage's composite coordinates.

This test characterises the divergence across representative boxes so we
can detect regressions when either side changes its snap-out logic.

For each bbox:
  1. Retrieve MARS area GRIB → extract grid points via eccodes iterator
  2. Retrieve polytope BoundingBox → extract composite coordinates
  3. Compare: point count, per-point max lat/lon Δ, symmetric-diff count

Run:
  RUN_INTEGRATION_TESTS=1 pytest tests/test_grib_snap_out_comparison.py -v -s
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

# Tolerance for pairing "the same" grid point across the two sources.
# At O1280, grid spacing is ~0.07° near the equator, so 0.02° is a safe
# nearest-neighbour bucket that won't cross-pair distinct grid points.
POINT_MATCH_TOL = 0.02  # degrees


# ---------------------------------------------------------------------------
# Bbox test cases — each stresses a different edge of the snap-out logic
# ---------------------------------------------------------------------------

BBOX_CASES = [
    pytest.param(
        [0.18, 0.00, 0.03, 0.15],  # N W S E
        id="tiny-equatorial-off-grid",  # edges land between grid points
    ),
    pytest.param(
        [0.50, 0.00, 0.00, 0.50],
        id="small-equatorial-round-numbers",  # nominally clean bounds
    ),
    pytest.param(
        [45.10, 10.10, 44.90, 10.30],
        id="mid-latitude-tiny",  # denser grid; N pole reduction different
    ),
    pytest.param(
        [1.00, -0.10, -1.00, 0.10],
        id="crosses-prime-meridian-signed-lons",
        # Meridian-straddling region expressed with signed longitudes
        # (W < E numerically, W negative). This is the canonical polytope
        # convention: bbox uses [[SW],[NE]] with signed lons — wrap syntax
        # (W > E) is rejected/undefined at the polytope-fe layer.
    ),
]


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def _retrieve_mars_area(bbox, output_path):
    from polytope.api import Client

    req = {**BASE_MARS_KEYS, "area": f"{bbox[0]}/{bbox[1]}/{bbox[2]}/{bbox[3]}"}
    print(f"\n  MARS area request: {json.dumps(req)}")
    Client(quiet=True).retrieve(COLLECTION, req, output_file=output_path, pointer=False)


def _retrieve_boundingbox_covjson(bbox):
    from polytope.api import Client

    req = {
        **BASE_MARS_KEYS,
        "feature": {
            "type": "boundingbox",
            "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
        },
    }
    print(f"  BBox request:      {json.dumps(req)}")
    with tempfile.NamedTemporaryFile(suffix=".covjson", delete=False, mode="w") as tmp:
        p = tmp.name
    try:
        Client(quiet=True).retrieve(COLLECTION, req, output_file=p)
        with open(p) as f:
            return json.load(f)
    finally:
        if os.path.exists(p):
            os.unlink(p)


# ---------------------------------------------------------------------------
# Point extraction
# ---------------------------------------------------------------------------


def _mars_grid_points(grib_path):
    """Extract (lat, lon) of every point in a GRIB message, with fallbacks.

    Returns (pts, corners, npoints).  `pts` may be None if eccodes can't
    build a geometry for the sub-area (which is itself a diagnostic worth
    reporting).
    """
    with open(grib_path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        try:
            corners = {
                "N": eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"),
                "W": eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"),
                "S": eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees"),
                "E": eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees"),
            }
            npoints = eccodes.codes_get(gid, "numberOfDataPoints")

            # Try iterator first
            pts = None
            try:
                iterid = eccodes.codes_grib_iterator_new(gid, 0)
                try:
                    pts = []
                    while True:
                        r = eccodes.codes_grib_iterator_next(iterid)
                        if not r:
                            break
                        lat, lon, _ = r
                        pts.append((lat, lon))
                finally:
                    eccodes.codes_grib_iterator_delete(iterid)
            except Exception as e:
                print(f"  (iterator failed: {e}; trying latitudes/longitudes arrays)")
                pts = None

            # Fallback to raw lat/lon arrays
            if pts is None:
                try:
                    lats = eccodes.codes_get_array(gid, "latitudes")
                    lons = eccodes.codes_get_array(gid, "longitudes")
                    pts = list(zip(lats.tolist(), lons.tolist()))
                except Exception as e:
                    print(
                        f"  (latitudes/longitudes also failed: {e}; "
                        f"eccodes cannot decode geometry — reporting counts only)"
                    )
                    pts = None

            return pts, corners, npoints
        finally:
            eccodes.codes_release(gid)


def _covjson_grid_points(covjson):
    """Extract (lat, lon) tuples from the first coverage's composite axis."""
    cov = covjson["coverages"][0]
    coords = cov["domain"]["axes"]["composite"]["values"]
    return [(c[0], c[1]) for c in coords]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _nearest_match(target, candidates, tol):
    """Return (best_match, distance) or (None, inf) if no candidate within tol."""
    best = None
    best_d = float("inf")
    for c in candidates:
        d = max(abs(target[0] - c[0]), abs(target[1] - c[1]))
        if d < best_d:
            best_d = d
            best = c
    return (best, best_d) if best_d <= tol else (None, best_d)


def _normalise_lon(lon):
    """Map lon into (-180, 180] so [0,360) and [-180,180] representations compare equal."""
    x = ((lon + 180.0) % 360.0) - 180.0
    return x if x != -180.0 else 180.0


def _normalise_pts(pts):
    return [(lat, _normalise_lon(lon)) for lat, lon in pts]


def _compare_point_sets(mars_pts, mars_corners, mars_npoints, bbox_pts, bbox):
    lines = []
    lines.append("")
    lines.append(f"  bbox (N/W/S/E):        {bbox}")
    lines.append(
        f"  MARS corners kept:     N={mars_corners['N']:.6f} W={mars_corners['W']:.6f} "
        f"S={mars_corners['S']:.6f} E={mars_corners['E']:.6f}"
    )
    lines.append(f"  MARS numberOfDataPoints: {mars_npoints}")
    lines.append(f"  BBox point count:      {len(bbox_pts)}")

    if mars_pts is None:
        lines.append("  MARS points:           <eccodes could not decode geometry>")
        return {
            "mars_count": mars_npoints,
            "bbox_count": len(bbox_pts),
            "matched": 0,
            "unmatched_mars": None,
            "unmatched_bbox": None,
            "max_delta": None,
            "count_diff": abs(mars_npoints - len(bbox_pts)),
            "report": "\n".join(lines),
        }

    lines.append(f"  MARS point count:      {len(mars_pts)}")

    if mars_pts:
        lats = [p[0] for p in mars_pts]
        lons = [p[1] for p in mars_pts]
        lines.append(f"  MARS lat range:        [{min(lats):.6f}, {max(lats):.6f}]")
        lines.append(f"  MARS lon range:        [{min(lons):.6f}, {max(lons):.6f}]")

    if bbox_pts:
        lats = [p[0] for p in bbox_pts]
        lons = [p[1] for p in bbox_pts]
        lines.append(f"  BBox lat range:        [{min(lats):.6f}, {max(lats):.6f}]")
        lines.append(f"  BBox lon range:        [{min(lons):.6f}, {max(lons):.6f}]")

    # Pair up points by nearest-neighbour with a tight tolerance.
    # Normalise longitudes to (-180, 180] so [0,360) vs [-180,180]
    # representations don't spuriously mismatch across the prime meridian.
    matched = 0
    max_delta = 0.0
    unmatched_mars = []
    remaining = _normalise_pts(bbox_pts)
    for mp in _normalise_pts(mars_pts):
        m, d = _nearest_match(mp, remaining, POINT_MATCH_TOL)
        if m is None:
            unmatched_mars.append(mp)
        else:
            matched += 1
            max_delta = max(max_delta, d)
            remaining.remove(m)

    lines.append(f"  Matched pairs:         {matched}")
    lines.append(f"  Unmatched in MARS:     {len(unmatched_mars)}")
    lines.append(f"  Unmatched in BBox:     {len(remaining)}")
    lines.append(f"  Max per-pair Δ:        {max_delta:.6f}° (tol {POINT_MATCH_TOL}°)")

    if unmatched_mars:
        lines.append(f"    MARS-only sample:  {unmatched_mars[:3]}")
    if remaining:
        lines.append(f"    BBox-only sample:  {remaining[:3]}")

    return {
        "mars_count": len(mars_pts),
        "bbox_count": len(bbox_pts),
        "matched": matched,
        "unmatched_mars": len(unmatched_mars),
        "unmatched_bbox": len(remaining),
        "max_delta": max_delta,
        "count_diff": abs(len(mars_pts) - len(bbox_pts)),
        "report": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bbox", BBOX_CASES)
def test_snap_out_matches(bbox):
    """polytope BoundingBox and MARS `area` should agree on grid points kept."""
    print(f"\n{'=' * 70}\n  Snap-out case: {bbox}\n{'=' * 70}")
    mars_f = tempfile.NamedTemporaryFile(suffix="_mars.grib", delete=False)
    mars_f.close()
    try:
        _retrieve_mars_area(bbox, mars_f.name)
        mars_pts, mars_corners, mars_npoints = _mars_grid_points(mars_f.name)

        covjson = _retrieve_boundingbox_covjson(bbox)
        bbox_pts = _covjson_grid_points(covjson)

        result = _compare_point_sets(mars_pts, mars_corners, mars_npoints, bbox_pts, bbox)
        print(result["report"])

        # If eccodes can't decode the MARS geometry we can only compare counts.
        if result["unmatched_mars"] is None:
            if result["count_diff"] != 0:
                pytest.fail(
                    f"snap-out count divergence: MARS={result['mars_count']} "
                    f"vs BBox={result['bbox_count']} (Δ={result['count_diff']})"
                )
            return

        if result["unmatched_mars"] or result["unmatched_bbox"]:
            pytest.fail(
                f"snap-out divergence: "
                f"{result['unmatched_mars']} MARS-only, "
                f"{result['unmatched_bbox']} BBox-only "
                f"(tol {POINT_MATCH_TOL}°) — see printed report above"
            )
    finally:
        if os.path.exists(mars_f.name):
            os.unlink(mars_f.name)


if __name__ == "__main__":
    os.environ["RUN_INTEGRATION_TESTS"] = "1"
    for bbox in [c.values[0] for c in BBOX_CASES]:
        print(f"\n{'=' * 70}\n  Snap-out case: {bbox}\n{'=' * 70}")
        mars_path = f"/tmp/mars_snap_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.grib"
        _retrieve_mars_area(bbox, mars_path)
        mars_pts, mars_corners = _mars_grid_points(mars_path)
        covjson = _retrieve_boundingbox_covjson(bbox)
        bbox_pts = _covjson_grid_points(covjson)
        res = _compare_point_sets(mars_pts, mars_corners, bbox_pts, bbox)
        print(res["report"])
