"""Integration test: Config-driven grid metadata → to_grib() vs MARS area.

This test simulates the full polytope-mars pipeline:
  1. Retrieve CoverageJSON via polytope BoundingBox feature request
  2. Simulate polytope-mars _get_grid_metadata() by extracting grid info
     from a config (as polytope-server would provide)
  3. Inject mars:grid into each coverage (as the encoder would do)
  4. Convert enriched CoverageJSON → GRIB via to_grib()
  5. Compare against MARS area retrieve (reference GRIB)

This validates the full chain:
  polytope-server config → _get_grid_metadata() → encoder mars:grid
  → decoder reads mars:grid → to_grib() produces correct output

Requirements:
  - Valid polytope credentials (~/.polytopeapirc)
  - Network access to polytope.ecmwf.int
  - eccodes Python package

Run:
  RUN_INTEGRATION_TESTS=1 pytest tests/test_grib_config_driven.py -v -s
"""

import json
import os
import tempfile

import pytest

eccodes = pytest.importorskip("eccodes", reason="eccodes required")

_skip_integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Integration test — set RUN_INTEGRATION_TESTS=1 to run",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AREA = [0.18, 0.0, 0.03, 0.15]  # N/W/S/E

MARS_REQUEST = {
    "class": "od",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "param": "2t/10v",
    "step": "0",
    "date": "-1",
    "time": "0000",
}

COLLECTION = "ecmwf-mars"

# Simulated polytope-server axis_config (matches polytope-od production config)
AXIS_CONFIG_OCTAHEDRAL = {
    "axis_name": "values",
    "transformations": [
        {
            "name": "mapper",
            "type": "octahedral",
            "resolution": 1280,
            "axes": ["latitude", "longitude"],
            "local": None,
        }
    ],
}

# EFAS-style regular_ll config for testing alternate grid types
AXIS_CONFIG_REGULAR_LL = {
    "axis_name": "values",
    "transformations": [
        {
            "name": "mapper",
            "type": "local_regular",
            "resolution": [2969, 4529],
            "axes": ["latitude", "longitude"],
            "local": [22.76, 72.24, -25.24, 50.24],
        }
    ],
}


# ---------------------------------------------------------------------------
# Simulate polytope-mars _get_grid_metadata()
# ---------------------------------------------------------------------------


def _get_grid_metadata_from_config(axis_config: dict) -> dict:
    """Replicate polytope-mars _get_grid_metadata() logic from a config dict.

    This is a standalone version of PolytopeMars._get_grid_metadata() that
    operates on a raw axis_config dict rather than requiring a full
    PolytopeMarsConfig instance.
    """
    for transform in axis_config.get("transformations", []):
        if transform.get("name") != "mapper":
            continue

        mapper_type = transform.get("type")
        resolution = transform.get("resolution")

        if mapper_type == "octahedral":
            return {"gridType": "reduced_gg", "N": int(resolution)}
        elif mapper_type == "local_regular":
            meta = {"gridType": "regular_ll"}
            if isinstance(resolution, list) and len(resolution) == 2:
                meta["Nj"] = int(resolution[0])
                meta["Ni"] = int(resolution[1])
            local = transform.get("local")
            if local and isinstance(local, list) and len(local) == 4:
                meta["area"] = [float(v) for v in local]
            return meta
        else:
            return {"mapperType": mapper_type}

    return {}


def _inject_grid_metadata(covjson: dict, grid_metadata: dict) -> dict:
    """Inject mars:grid into each coverage (as the encoder would do)."""
    if not grid_metadata:
        return covjson

    for coverage in covjson.get("coverages", []):
        coverage["mars:grid"] = grid_metadata

    return covjson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_all_grib_messages(filepath):
    """Read all messages from a GRIB file."""
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
                    "bitsPerValue": eccodes.codes_get(gid, "bitsPerValue"),
                    "values": eccodes.codes_get_values(gid).tolist(),
                }
                try:
                    msg["N"] = eccodes.codes_get(gid, "N")
                except Exception:
                    pass
                try:
                    msg["Nj"] = eccodes.codes_get(gid, "Nj")
                except Exception:
                    pass
                try:
                    msg["Ni"] = eccodes.codes_get(gid, "Ni")
                except Exception:
                    pass
                try:
                    msg["latitudeOfFirstGridPointInDegrees"] = eccodes.codes_get(
                        gid, "latitudeOfFirstGridPointInDegrees"
                    )
                    msg["longitudeOfFirstGridPointInDegrees"] = eccodes.codes_get(
                        gid, "longitudeOfFirstGridPointInDegrees"
                    )
                    msg["latitudeOfLastGridPointInDegrees"] = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
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
    """Retrieve reference GRIB via polytope mars-od with area keyword."""
    from polytope.api import Client

    client = Client()
    request = {**MARS_REQUEST, "area": f"{AREA[0]}/{AREA[1]}/{AREA[2]}/{AREA[3]}"}

    print(f"\n{'='*60}")
    print("REFERENCE: Polytope mars-od retrieve with area keyword")
    print(f"{'='*60}")
    print(f"Request: {request}")

    client.retrieve(COLLECTION, request, output_file=output_path, pointer=False)

    file_size = os.path.getsize(output_path)
    print(f"Retrieved {file_size} bytes")
    return output_path


def _retrieve_boundingbox_covjson():
    """Retrieve CoverageJSON via polytope BoundingBox feature."""
    from polytope.api import Client

    client = Client()
    request = {**MARS_REQUEST}
    request["feature"] = {
        "type": "boundingbox",
        "points": [[AREA[0], AREA[1]], [AREA[2], AREA[3]]],
    }

    print(f"\n{'='*60}")
    print("FEATURE: Polytope BoundingBox → CoverageJSON")
    print(f"{'='*60}")
    print(f"Request: {request}")

    with tempfile.NamedTemporaryFile(suffix=".covjson", delete=False, mode="w") as tmp:
        covjson_path = tmp.name

    try:
        client.retrieve(COLLECTION, request, output_file=covjson_path)
        with open(covjson_path) as f:
            covjson = json.load(f)

        print(f"Coverages: {len(covjson.get('coverages', []))}")
        return covjson
    finally:
        if os.path.exists(covjson_path):
            os.unlink(covjson_path)


def _convert_to_grib(covjson, output_path):
    """Convert CoverageJSON to GRIB via to_grib()."""
    from covjsonkit.api import Covjsonkit

    decoder = Covjsonkit().decode(covjson)
    decoder.to_grib(output_path, backend="eccodes")

    file_size = os.path.getsize(output_path)
    print(f"Wrote {file_size} bytes GRIB")
    return output_path


def _compare_messages(mars_msgs, covjson_msgs, tolerance=0.001):
    """Compare two sets of GRIB messages. Returns (all_match, report_lines)."""
    report = []
    all_match = True

    report.append(f"MARS messages: {len(mars_msgs)}, CovJSON messages: {len(covjson_msgs)}")

    if len(mars_msgs) != len(covjson_msgs):
        report.append("  ⚠️  Message count mismatch")
        all_match = False

    for mars_msg in mars_msgs:
        key = (mars_msg["shortName"], mars_msg["stepRange"])
        matching = [m for m in covjson_msgs if (m["shortName"], m["stepRange"]) == key]

        if not matching:
            report.append(f"  ❌ No match for {key}")
            all_match = False
            continue

        covjson_msg = matching[0]
        report.append(f"\n  Field: {key[0]} step={key[1]}")

        # Point count
        if mars_msg["numberOfDataPoints"] != covjson_msg["numberOfDataPoints"]:
            report.append(
                f"    ⚠️  Points: MARS={mars_msg['numberOfDataPoints']} " f"CovJSON={covjson_msg['numberOfDataPoints']}"
            )
            all_match = False
        else:
            report.append(f"    ✓ Points: {mars_msg['numberOfDataPoints']}")

        # Grid type
        if mars_msg["gridType"] != covjson_msg["gridType"]:
            report.append(f"    ⚠️  Grid: MARS={mars_msg['gridType']} CovJSON={covjson_msg['gridType']}")
            all_match = False
        else:
            report.append(f"    ✓ Grid: {mars_msg['gridType']}")

        # N value (for reduced_gg)
        if "N" in mars_msg and "N" in covjson_msg:
            if mars_msg["N"] != covjson_msg["N"]:
                report.append(f"    ⚠️  N: MARS={mars_msg['N']} CovJSON={covjson_msg['N']}")
                all_match = False
            else:
                report.append(f"    ✓ N: {mars_msg['N']}")

        # Values
        mars_vals = mars_msg["values"]
        covjson_vals = covjson_msg["values"]

        if len(mars_vals) != len(covjson_vals):
            report.append(f"    ⚠️  Value count: {len(mars_vals)} vs {len(covjson_vals)}")
            all_match = False
        else:
            diffs = [abs(a - b) for a, b in zip(mars_vals, covjson_vals)]
            max_diff = max(diffs) if diffs else 0
            mean_diff = sum(diffs) / len(diffs) if diffs else 0
            n_exact = sum(1 for d in diffs if d == 0.0)

            report.append(
                f"    Values: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, exact={n_exact}/{len(mars_vals)}"
            )

            if max_diff > tolerance:
                report.append(f"    ⚠️  Exceeds tolerance {tolerance}")
                all_match = False
            else:
                report.append("    ✓ Within tolerance")

        # Area bounds
        for bound_key in [
            "latitudeOfFirstGridPointInDegrees",
            "latitudeOfLastGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "longitudeOfLastGridPointInDegrees",
        ]:
            if bound_key in mars_msg and bound_key in covjson_msg:
                diff = abs(mars_msg[bound_key] - covjson_msg[bound_key])
                status = "✓" if diff < 0.001 else "⚠️"
                short_key = bound_key.replace("InDegrees", "").replace("OfGridPoint", "")
                report.append(f"    {status} {short_key}: {mars_msg[bound_key]:.6f} vs {covjson_msg[bound_key]:.6f}")
                if diff >= 0.001:
                    all_match = False

    return all_match, report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_integration
class TestConfigDrivenGridToGrib:
    """Full pipeline: config → grid_metadata → mars:grid → to_grib() vs MARS.

    Run: RUN_INTEGRATION_TESTS=1 pytest tests/test_grib_config_driven.py -v -s
    """

    def test_octahedral_config_vs_mars_area(self):
        """Octahedral N=1280 config (polytope-od production) → GRIB matches MARS area.

        Pipeline:
          1. _get_grid_metadata_from_config(octahedral, resolution=1280)
             → {"gridType": "reduced_gg", "N": 1280}
          2. Retrieve CovJSON via feature request
          3. Inject mars:grid (simulating encoder)
          4. to_grib() → GRIB
          5. Compare vs MARS area retrieve
        """
        # Step 1: Extract grid metadata from config (as polytope-mars would)
        grid_metadata = _get_grid_metadata_from_config(AXIS_CONFIG_OCTAHEDRAL)
        print(f"\nGrid metadata from config: {grid_metadata}")
        assert grid_metadata == {"gridType": "reduced_gg", "N": 1280}

        # Step 2: Retrieve reference GRIB from MARS
        mars_grib = tempfile.NamedTemporaryFile(suffix="_mars.grib", delete=False)
        mars_grib.close()

        # Step 3: Retrieve CoverageJSON via feature request
        covjson_grib = tempfile.NamedTemporaryFile(suffix="_config.grib", delete=False)
        covjson_grib.close()

        try:
            _retrieve_mars_area(mars_grib.name)
            covjson = _retrieve_boundingbox_covjson()

            # Step 4: Inject mars:grid (simulating what encoder does with grid_metadata)
            print(f"\n{'='*60}")
            print("INJECT: Adding mars:grid from config to each coverage")
            print(f"{'='*60}")
            covjson = _inject_grid_metadata(covjson, grid_metadata)

            # Verify injection
            for i, cov in enumerate(covjson.get("coverages", [])):
                assert "mars:grid" in cov, f"Coverage {i} missing mars:grid"
                assert cov["mars:grid"]["gridType"] == "reduced_gg"
                assert cov["mars:grid"]["N"] == 1280
                print(f"  Coverage {i}: mars:grid = {cov['mars:grid']}")

            # Step 5: Convert to GRIB
            print(f"\n{'='*60}")
            print("CONVERT: CoverageJSON (with mars:grid) → GRIB")
            print(f"{'='*60}")
            _convert_to_grib(covjson, covjson_grib.name)

            # Step 6: Compare
            print(f"\n{'='*60}")
            print("COMPARE: MARS area vs config-driven to_grib()")
            print(f"{'='*60}")
            mars_msgs = _read_all_grib_messages(mars_grib.name)
            covjson_msgs = _read_all_grib_messages(covjson_grib.name)

            match, report = _compare_messages(mars_msgs, covjson_msgs)
            print("\n".join(report))

            assert match, "Config-driven GRIB does not match MARS area output"

        finally:
            for path in [mars_grib.name, covjson_grib.name]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_without_config_uses_defaults_still_matches(self):
        """Without mars:grid (no config), fallback defaults should still produce matching values.

        This confirms backward compatibility — even without the config pipeline,
        to_grib() uses _OPER_GRID_DEFAULTS and still produces correct values.
        """
        mars_grib = tempfile.NamedTemporaryFile(suffix="_mars.grib", delete=False)
        mars_grib.close()
        covjson_grib = tempfile.NamedTemporaryFile(suffix="_noconfig.grib", delete=False)
        covjson_grib.close()

        try:
            _retrieve_mars_area(mars_grib.name)
            covjson = _retrieve_boundingbox_covjson()

            # No mars:grid injection — uses defaults
            assert "mars:grid" not in covjson.get("coverages", [{}])[0]

            print(f"\n{'='*60}")
            print("CONVERT: CoverageJSON (no mars:grid, using defaults) → GRIB")
            print(f"{'='*60}")
            _convert_to_grib(covjson, covjson_grib.name)

            mars_msgs = _read_all_grib_messages(mars_grib.name)
            covjson_msgs = _read_all_grib_messages(covjson_grib.name)

            match, report = _compare_messages(mars_msgs, covjson_msgs)
            print("\n".join(report))

            assert match, "Default-driven GRIB does not match MARS area output"

        finally:
            for path in [mars_grib.name, covjson_grib.name]:
                if os.path.exists(path):
                    os.unlink(path)


class TestGridMetadataExtraction:
    """Unit tests for _get_grid_metadata_from_config() — no network needed."""

    def test_octahedral_config(self):
        """Octahedral mapper → reduced_gg with N."""
        result = _get_grid_metadata_from_config(AXIS_CONFIG_OCTAHEDRAL)
        assert result == {"gridType": "reduced_gg", "N": 1280}

    def test_regular_ll_config(self):
        """local_regular mapper → regular_ll with Ni, Nj, area."""
        result = _get_grid_metadata_from_config(AXIS_CONFIG_REGULAR_LL)
        assert result == {
            "gridType": "regular_ll",
            "Nj": 2969,
            "Ni": 4529,
            "area": [22.76, 72.24, -25.24, 50.24],
        }

    def test_no_mapper_transform(self):
        """Config without mapper transform → empty dict."""
        config = {
            "axis_name": "values",
            "transformations": [{"name": "type_change", "type": "int"}],
        }
        result = _get_grid_metadata_from_config(config)
        assert result == {}

    def test_empty_transformations(self):
        """Config with no transformations → empty dict."""
        config = {"axis_name": "values", "transformations": []}
        result = _get_grid_metadata_from_config(config)
        assert result == {}

    def test_unknown_mapper_type(self):
        """Unknown mapper type → returns mapperType key."""
        config = {
            "axis_name": "values",
            "transformations": [{"name": "mapper", "type": "healpix", "resolution": 512}],
        }
        result = _get_grid_metadata_from_config(config)
        assert result == {"mapperType": "healpix"}

    def test_injection_adds_mars_grid(self):
        """_inject_grid_metadata adds mars:grid to every coverage."""
        covjson = {
            "coverages": [
                {"mars:metadata": {"class": "od"}},
                {"mars:metadata": {"class": "od"}},
            ]
        }
        grid_meta = {"gridType": "reduced_gg", "N": 1280}
        result = _inject_grid_metadata(covjson, grid_meta)

        for cov in result["coverages"]:
            assert cov["mars:grid"] == grid_meta

    def test_injection_empty_metadata_no_op(self):
        """Empty grid_metadata → no mars:grid injected."""
        covjson = {"coverages": [{"mars:metadata": {"class": "od"}}]}
        result = _inject_grid_metadata(covjson, {})

        assert "mars:grid" not in result["coverages"][0]
