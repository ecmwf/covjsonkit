"""Smoke tests for pymars2grib backend.

Tests the hybrid approach: mars2grib generates the GRIB template (sections
1, 4, 5 from MARS keys), then eccodes overwrites section 3 (grid definition)
with sub-area geometry and fills in the data values.

NOTE: `skipSection3=True` is unusable with the current eccodes build — it
triggers the `gridSpec` string setter which is not implemented
(`GridSpec::pack_string not available`). Instead we let mars2grib produce a
normal template and then overwrite section 3 via eccodes.

Run from the Python 3.11 venv where pymars2grib is importable:
    source mars2grib-bundle/env.sh
    python tests/test_pymars2grib_smoke.py
"""

import os
import tempfile

import pytest

# pymars2grib requires a locally-built metkit bundle (see mars2grib-bundle/).
# Skip the whole module in CI environments where it isn't available.
pymars2grib = pytest.importorskip(
    "pymars2grib",
    reason="pymars2grib is not installed; build metkit from source "
    "(see mars2grib-bundle/) to enable these smoke tests",
)
pytest.importorskip("eccodes", reason="eccodes required for section-3 overwrite")


def test_pymars2grib_basic_encode():
    """Verify pymars2grib can encode and eccodes can overwrite section 3."""
    from pymars2grib import Mars2Grib

    # Plain Mars2Grib (no skipSection3 — that path hits an unimplemented
    # gridSpec::pack_string in eccodes). mars2grib builds sections 1,4,5;
    # eccodes then overwrites section 3 with sub-area geometry.
    encoder = Mars2Grib()

    values = [280.5, 281.0, 279.3, 282.1, 280.0, 281.5, 279.8, 280.2, 281.3]
    mars = {
        "origin": "ecmf",
        "class": "od",
        "stream": "oper",
        "type": "fc",
        "expver": "0001",
        "date": 20250702,
        "time": 0,
        "step": 0,
        "levtype": "sfc",
        "param": 167,
        "packing": "ccsds",
        "grid": "O1280",
    }

    # mars2grib produces a small template; eccodes will overwrite section 3
    data = encoder.encode(values, mars)
    print(f"  mars2grib encoded → {len(data)} bytes (template)")
    assert len(data) > 0, "Empty GRIB output"

    # Now use eccodes to fill in section 3 (grid geometry) for our sub-area
    import eccodes

    gid = eccodes.codes_new_from_message(data)

    # Set sub-area grid geometry
    eccodes.codes_set(gid, "gridType", "reduced_gg")
    eccodes.codes_set_long(gid, "numberOfParallelsBetweenAPoleAndTheEquator", 1280)
    eccodes.codes_set_long(gid, "Nj", 3)  # 3 latitude rows in our sub-area
    eccodes.codes_set_long_array(gid, "pl", [3, 3, 3])  # 3 points per row
    eccodes.codes_set_double(gid, "latitudeOfFirstGridPointInDegrees", 0.176)
    eccodes.codes_set_double(gid, "latitudeOfLastGridPointInDegrees", 0.035)
    eccodes.codes_set_double(gid, "longitudeOfFirstGridPointInDegrees", 0.0)
    eccodes.codes_set_double(gid, "longitudeOfLastGridPointInDegrees", 0.141)

    # Set the actual data values
    eccodes.codes_set_long(gid, "bitsPerValue", 16)
    eccodes.codes_set_values(gid, values)

    # Write final GRIB
    with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as f:
        path = f.name
    eccodes.codes_write(gid, open(path, "wb"))
    eccodes.codes_release(gid)

    # Read back and verify
    try:
        with open(path, "rb") as f:
            gid = eccodes.codes_grib_new_from_file(f)
            param_id = eccodes.codes_get(gid, "paramId")
            grid_type = eccodes.codes_get(gid, "gridType")
            n_points = eccodes.codes_get(gid, "numberOfDataPoints")
            decoded = eccodes.codes_get_values(gid).tolist()
            eccodes.codes_release(gid)

        print(f"  paramId: {param_id}")
        print(f"  gridType: {grid_type}")
        print(f"  numberOfDataPoints: {n_points}")

        assert param_id == 167, f"Expected paramId=167, got {param_id}"
        assert grid_type == "reduced_gg", f"Expected reduced_gg, got {grid_type}"
        assert n_points == len(values), f"Expected {len(values)} points, got {n_points}"

        max_diff = max(abs(a - b) for a, b in zip(values, decoded))
        print(f"  max value diff: {max_diff:.2e}")
        assert max_diff < 0.01, f"Value mismatch too large: {max_diff}"

        print("✓ Hybrid mars2grib+eccodes encode works")
    finally:
        os.unlink(path)


def test_pymars2grib_skip_section3_ensemble():
    """Verify ensemble fields resolve correctly (perturbationNumber)."""
    from pymars2grib import Mars2Grib

    encoder = Mars2Grib()

    values = [280.5, 281.0, 279.3]
    mars = {
        "origin": "ecmf",
        "class": "od",
        "stream": "enfo",
        "type": "pf",
        "expver": "0001",
        "date": 20250702,
        "time": 0,
        "step": 0,
        "levtype": "sfc",
        "param": 167,
        "packing": "ccsds",
        "grid": "O1280",
        "number": 1,
    }

    # Ensemble encoding requires `numberOfForecastsInEnsemble` in misc —
    # mars2grib has no default deduction for it. Aborts the process otherwise.
    misc = {"numberOfForecastsInEnsemble": 51}

    data = encoder.encode(values, mars, misc)
    print(f"  Ensemble template: {len(data)} bytes")
    assert len(data) > 0

    # Verify mars2grib set the ensemble keys correctly
    import eccodes

    gid = eccodes.codes_new_from_message(data)

    try:
        pdt = eccodes.codes_get(gid, "productDefinitionTemplateNumber")
        print(f"  productDefinitionTemplateNumber: {pdt}")
        # Template 1 = individual ensemble forecast member
        assert pdt == 1, f"Expected template 1 for ensemble, got {pdt}"

        perturb = eccodes.codes_get(gid, "perturbationNumber")
        print(f"  perturbationNumber: {perturb}")
        assert perturb == 1, f"Expected perturbationNumber=1, got {perturb}"

        print("✓ Ensemble template correctly resolved by mars2grib")
    finally:
        eccodes.codes_release(gid)


def test_pymars2grib_mars_keys_only():
    """Verify mars2grib correctly resolves MARS keys to GRIB keys (sections 1,4,5)."""
    from pymars2grib import Mars2Grib

    encoder = Mars2Grib()

    mars = {
        "origin": "ecmf",
        "class": "od",
        "stream": "oper",
        "type": "fc",
        "expver": "0001",
        "date": 20250702,
        "time": 1200,
        "step": 6,
        "levtype": "pl",
        "levelist": 500,
        "param": 130,  # temperature
        "packing": "ccsds",
        "grid": "O1280",
    }

    data = encoder.encode([250.0] * 10, mars)
    assert len(data) > 0

    import eccodes

    gid = eccodes.codes_new_from_message(data)
    try:
        print(f"  paramId: {eccodes.codes_get(gid, 'paramId')}")
        print(f"  dataDate: {eccodes.codes_get(gid, 'dataDate')}")
        print(f"  dataTime: {eccodes.codes_get(gid, 'dataTime')}")
        print(f"  stepRange: {eccodes.codes_get(gid, 'stepRange')}")

        assert eccodes.codes_get(gid, "paramId") == 130
        assert eccodes.codes_get(gid, "dataDate") == 20250702
        assert eccodes.codes_get(gid, "dataTime") in (12, 1200)
        assert eccodes.codes_get(gid, "level") in (5, 500)

        print("✓ MARS key resolution correct (sections 1, 4, 5)")
    finally:
        eccodes.codes_release(gid)


def test_pymars2grib_via_backend_factory():
    """Verify the covjsonkit backend factory picks up pymars2grib."""
    from covjsonkit.decoder.grib_backends import get_backend
    from covjsonkit.decoder.grib_backends.mars2grib_backend import Mars2GribBackend

    backend = get_backend("auto")
    print(f"\n  get_backend('auto') → {type(backend).__name__}")
    assert isinstance(backend, Mars2GribBackend), f"Expected Mars2GribBackend, got {type(backend).__name__}"
    print("✓ Auto-detection picks mars2grib when available")

    backend = get_backend("mars2grib")
    assert isinstance(backend, Mars2GribBackend)
    print("✓ Explicit 'mars2grib' works")


def test_pymars2grib_full_covjson_roundtrip():
    """Full test: load CovJSON fixture → to_grib(backend='mars2grib') → verify."""
    import json

    from covjsonkit.api import Covjsonkit

    fixture_path = os.path.join(os.path.dirname(__file__), "data", "test_oper_multipoint_coverage.json")
    with open(fixture_path) as f:
        covjson = json.load(f)

    decoder = Covjsonkit().decode(covjson)

    with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
        output = tmp.name

    try:
        result = decoder.to_grib(output, backend="mars2grib")
        assert result == output
        file_size = os.path.getsize(output)
        print(f"\n  to_grib(backend='mars2grib') wrote {file_size} bytes")

        # Read back and verify
        import eccodes

        messages = []
        with open(output, "rb") as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break
                msg = {
                    "paramId": eccodes.codes_get(gid, "paramId"),
                    "shortName": eccodes.codes_get(gid, "shortName"),
                    "gridType": eccodes.codes_get(gid, "gridType"),
                    "numberOfDataPoints": eccodes.codes_get(gid, "numberOfDataPoints"),
                    "values": eccodes.codes_get_values(gid).tolist(),
                }
                eccodes.codes_release(gid)
                messages.append(msg)

        print(f"  Messages: {len(messages)}")
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        params = sorted(m["shortName"] for m in messages)
        print(f"  Params: {params}")
        assert "2t" in params
        assert "10v" in params

        for msg in messages:
            assert msg["numberOfDataPoints"] == 9
            assert msg["gridType"] == "reduced_gg"

        print("✓ Full CovJSON → mars2grib → GRIB roundtrip works")
    finally:
        os.unlink(output)


if __name__ == "__main__":
    print("=" * 60)
    print("pymars2grib Smoke Test")
    print("=" * 60)

    tests = [
        test_pymars2grib_basic_encode,
        test_pymars2grib_skip_section3_ensemble,
        test_pymars2grib_mars_keys_only,
        test_pymars2grib_via_backend_factory,
        test_pymars2grib_full_covjson_roundtrip,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
