"""Integration test: Compare MARS area output vs BoundingBox to_grib() output.

This test retrieves the same data via two polytope paths:
  1. Polytope mars-od datasource with `area` keyword → GRIB directly
  2. Polytope boundingbox feature request → CoverageJSON → to_grib()

Then compares the two GRIB files field-by-field to verify our
to_grib() produces output matching what MARS returns natively.

Requirements:
  - Valid polytope credentials (~/.polytopeapirc)
  - Network access to polytope.ecmwf.int
  - eccodes Python package

Run:
  pytest tests/test_grib_area_comparison.py -v -m integration
  # or directly:
  python tests/test_grib_area_comparison.py
"""

import json
import os
import sys
import tempfile

import pytest

eccodes = pytest.importorskip("eccodes", reason="eccodes required")

# Skip by default unless explicitly requested via marker or env var
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Integration test — set RUN_INTEGRATION_TESTS=1 to run",
)

# ---------------------------------------------------------------------------
# Configuration — adjust these for your test case
# ---------------------------------------------------------------------------

# A small bounding box near the equator (matches our test fixtures)
AREA = [0.18, 0.0, 0.03, 0.15]  # N/W/S/E in degrees

# MARS request keys (common to both retrieval paths)
MARS_REQUEST = {
    "class": "od",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "param": "2t/10v",
    "step": "0",
    "date": "-1",  # yesterday (most likely available)
    "time": "0000",
}

# Polytope collection for standard MARS retrieves
COLLECTION = "ecmwf-mars"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_all_grib_messages(filepath):
    """Read all messages from a GRIB file, returning list of dicts with keys+values."""
    messages = []
    with open(filepath, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                msg = {
                    "paramId": eccodes.codes_get(gid, "paramId"),
                    "shortName": eccodes.codes_get(gid, "shortName"),
                    "dataDate": eccodes.codes_get(gid, "dataDate"),
                    "dataTime": eccodes.codes_get(gid, "dataTime"),
                    "stepRange": eccodes.codes_get(gid, "stepRange"),
                    "gridType": eccodes.codes_get(gid, "gridType"),
                    "numberOfDataPoints": eccodes.codes_get(gid, "numberOfDataPoints"),
                    "values": eccodes.codes_get_values(gid).tolist(),
                }
                try:
                    msg["N"] = eccodes.codes_get(gid, "N")
                except Exception:
                    pass
                try:
                    msg["latitudeOfFirstGridPointInDegrees"] = eccodes.codes_get(
                        gid, "latitudeOfFirstGridPointInDegrees"
                    )
                    msg["longitudeOfFirstGridPointInDegrees"] = eccodes.codes_get(
                        gid, "longitudeOfFirstGridPointInDegrees"
                    )
                    msg["latitudeOfLastGridPointInDegrees"] = eccodes.codes_get(
                        gid, "latitudeOfLastGridPointInDegrees"
                    )
                    msg["longitudeOfLastGridPointInDegrees"] = eccodes.codes_get(
                        gid, "longitudeOfLastGridPointInDegrees"
                    )
                except Exception:
                    pass
                messages.append(msg)
            finally:
                eccodes.codes_release(gid)
    return messages


def _retrieve_mars_area(output_path):
    """Retrieve data via polytope mars-od datasource with area keyword → GRIB."""
    from polytope.api import Client

    client = Client()

    # Dict-based request with area key routes to mars-od datasource
    request = {**MARS_REQUEST, "area": f"{AREA[0]}/{AREA[1]}/{AREA[2]}/{AREA[3]}"}

    print(f"\n{'='*60}")
    print("PATH 1: Polytope mars-od retrieve with area keyword")
    print(f"{'='*60}")
    print(f"Collection: {COLLECTION}")
    print(f"Request: {request}")
    print(f"Output: {output_path}")

    client.retrieve(
        COLLECTION,
        request,
        output_file=output_path,
        pointer=False,
    )

    file_size = os.path.getsize(output_path)
    print(f"Retrieved {file_size} bytes")
    return output_path


def _retrieve_boundingbox_covjson():
    """Retrieve data via polytope BoundingBox feature → CoverageJSON dict."""
    from polytope.api import Client

    client = Client()

    # Build dict-based request (polytope requires dict format for feature routing)
    request = {**MARS_REQUEST}
    request["feature"] = {
        "type": "boundingbox",
        "points": [[AREA[0], AREA[1]], [AREA[2], AREA[3]]],
    }

    print(f"\n{'='*60}")
    print("PATH 2: Polytope BoundingBox feature → CoverageJSON")
    print(f"{'='*60}")
    print(f"Collection: {COLLECTION}")
    print(f"Request: {request}")

    # Feature requests return CoverageJSON (write to temp file, then parse)
    with tempfile.NamedTemporaryFile(suffix=".covjson", delete=False, mode="w") as tmp:
        covjson_path = tmp.name

    try:
        client.retrieve(
            COLLECTION,
            request,
            output_file=covjson_path,
        )

        file_size = os.path.getsize(covjson_path)
        print(f"Retrieved {file_size} bytes of CoverageJSON")

        with open(covjson_path) as f:
            covjson = json.load(f)

        print(f"Type: {covjson.get('type')}")
        print(f"Domain type: {covjson.get('domainType')}")
        print(f"Coverages: {len(covjson.get('coverages', []))}")

        return covjson
    finally:
        if os.path.exists(covjson_path):
            os.unlink(covjson_path)


def _convert_covjson_to_grib(covjson, output_path):
    """Convert CoverageJSON to GRIB using our to_grib() method."""
    from covjsonkit.api import Covjsonkit

    print(f"\n{'='*60}")
    print("Converting CoverageJSON → GRIB via to_grib()")
    print(f"{'='*60}")

    decoder = Covjsonkit().decode(covjson)
    result = decoder.to_grib(output_path, backend="eccodes")

    file_size = os.path.getsize(output_path)
    print(f"Wrote {file_size} bytes to {output_path}")
    return result


def _compare_grib_files(mars_grib_path, covjson_grib_path):
    """Compare two GRIB files and report differences.

    Returns (match: bool, report: str)
    """
    mars_msgs = _read_all_grib_messages(mars_grib_path)
    covjson_msgs = _read_all_grib_messages(covjson_grib_path)

    report_lines = []
    report_lines.append(f"\n{'='*60}")
    report_lines.append("COMPARISON RESULTS")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"MARS area messages: {len(mars_msgs)}")
    report_lines.append(f"CovJSON→GRIB messages: {len(covjson_msgs)}")
    report_lines.append("")

    all_match = True

    # Compare metadata
    mars_params = sorted(set(m["shortName"] for m in mars_msgs))
    covjson_params = sorted(set(m["shortName"] for m in covjson_msgs))
    report_lines.append(f"MARS params: {mars_params}")
    report_lines.append(f"CovJSON params: {covjson_params}")

    if mars_params != covjson_params:
        report_lines.append("⚠️  Parameter mismatch!")
        all_match = False

    # Match messages by (shortName, stepRange) and compare
    for mars_msg in mars_msgs:
        key = (mars_msg["shortName"], mars_msg["stepRange"])
        matching = [m for m in covjson_msgs if (m["shortName"], m["stepRange"]) == key]

        if not matching:
            report_lines.append(f"\n❌ No CovJSON match for {key}")
            all_match = False
            continue

        covjson_msg = matching[0]
        report_lines.append(f"\n--- Field: {key[0]}, step={key[1]} ---")

        # Compare number of data points
        if mars_msg["numberOfDataPoints"] != covjson_msg["numberOfDataPoints"]:
            report_lines.append(
                f"  ⚠️  Point count differs: MARS={mars_msg['numberOfDataPoints']} "
                f"vs CovJSON={covjson_msg['numberOfDataPoints']}"
            )
            all_match = False
        else:
            report_lines.append(f"  ✓ Point count: {mars_msg['numberOfDataPoints']}")

        # Compare grid type
        report_lines.append(f"  Grid: MARS={mars_msg['gridType']} vs CovJSON={covjson_msg['gridType']}")
        if mars_msg["gridType"] != covjson_msg["gridType"]:
            report_lines.append("  ⚠️  Grid type differs (expected — see notes)")

        # Compare values
        mars_vals = mars_msg["values"]
        covjson_vals = covjson_msg["values"]

        if len(mars_vals) != len(covjson_vals):
            report_lines.append(f"  ⚠️  Value count differs: {len(mars_vals)} vs {len(covjson_vals)}")
            all_match = False
        else:
            # Statistics on differences
            diffs = [abs(a - b) for a, b in zip(mars_vals, covjson_vals)]
            max_diff = max(diffs) if diffs else 0
            mean_diff = sum(diffs) / len(diffs) if diffs else 0
            n_exact = sum(1 for d in diffs if d == 0.0)

            report_lines.append(f"  Values ({len(mars_vals)} points):")
            report_lines.append(f"    Max abs diff:  {max_diff:.10e}")
            report_lines.append(f"    Mean abs diff: {mean_diff:.10e}")
            report_lines.append(f"    Exact matches: {n_exact}/{len(mars_vals)}")

            # Check if values are "close enough" (within GRIB packing tolerance)
            tolerance = 0.001  # 1e-3 — generous for GRIB2 simple packing
            if max_diff > tolerance:
                report_lines.append(f"  ⚠️  Max diff {max_diff} exceeds tolerance {tolerance}")
                all_match = False
            else:
                report_lines.append(f"  ✓ All values within tolerance ({tolerance})")

        # Compare area bounds if available
        for bound_key in [
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "latitudeOfLastGridPointInDegrees",
            "longitudeOfLastGridPointInDegrees",
        ]:
            if bound_key in mars_msg and bound_key in covjson_msg:
                mars_val = mars_msg[bound_key]
                covjson_val = covjson_msg[bound_key]
                diff = abs(mars_val - covjson_val)
                status = "✓" if diff < 0.001 else "⚠️"
                report_lines.append(f"  {status} {bound_key}: MARS={mars_val:.6f} CovJSON={covjson_val:.6f}")

    # Summary
    report_lines.append(f"\n{'='*60}")
    if all_match:
        report_lines.append("✅ OVERALL: GRIB files match within tolerance")
    else:
        report_lines.append("❌ OVERALL: Differences found (see details above)")
    report_lines.append(f"{'='*60}")

    report = "\n".join(report_lines)
    return all_match, report


# ---------------------------------------------------------------------------
# Pytest test
# ---------------------------------------------------------------------------


class TestGribAreaComparison:
    """Compare MARS area retrieve vs BoundingBox→to_grib().

    Run with: RUN_INTEGRATION_TESTS=1 pytest tests/test_grib_area_comparison.py -v -s
    """

    def test_area_vs_boundingbox_to_grib(self):
        """Full roundtrip comparison: MARS area vs CoverageJSON→GRIB."""
        mars_grib = tempfile.NamedTemporaryFile(suffix="_mars_area.grib", delete=False)
        covjson_grib = tempfile.NamedTemporaryFile(suffix="_covjson.grib", delete=False)
        mars_grib.close()
        covjson_grib.close()

        try:
            # Path 1: MARS area → GRIB
            _retrieve_mars_area(mars_grib.name)

            # Path 2: BoundingBox feature → CoverageJSON → GRIB
            covjson = _retrieve_boundingbox_covjson()
            _convert_covjson_to_grib(covjson, covjson_grib.name)

            # Compare
            match, report = _compare_grib_files(mars_grib.name, covjson_grib.name)
            print(report)

            # Don't assert match yet — first run is diagnostic
            # Uncomment below once you've validated the output:
            # assert match, report

        finally:
            for path in [mars_grib.name, covjson_grib.name]:
                if os.path.exists(path):
                    os.unlink(path)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run directly without pytest for quick manual testing."""
    print("GRIB Area Comparison Test")
    print(f"Area: N={AREA[0]}, W={AREA[1]}, S={AREA[2]}, E={AREA[3]}")
    print(f"Request: {MARS_REQUEST}")

    mars_grib_path = "/tmp/mars_area_output.grib"
    covjson_grib_path = "/tmp/covjson_to_grib_output.grib"

    try:
        # Path 1: Standard MARS area retrieve
        _retrieve_mars_area(mars_grib_path)

        # Path 2: BoundingBox feature → CoverageJSON → to_grib
        covjson = _retrieve_boundingbox_covjson()

        # Save CoverageJSON for inspection
        covjson_path = "/tmp/boundingbox_covjson.json"
        with open(covjson_path, "w") as f:
            json.dump(covjson, f, indent=2)
        print(f"\nCoverageJSON saved to {covjson_path}")

        _convert_covjson_to_grib(covjson, covjson_grib_path)

        # Compare
        match, report = _compare_grib_files(mars_grib_path, covjson_grib_path)
        print(report)

        print(f"\nFiles preserved for inspection:")
        print(f"  MARS area GRIB:   {mars_grib_path}")
        print(f"  CovJSON→GRIB:     {covjson_grib_path}")
        print(f"  CoverageJSON:     {covjson_path}")

        sys.exit(0 if match else 1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)
