"""Integration test: MARS `area` retrieve vs polytope BoundingBox→to_grib(),
compared for **both** GRIB backends (eccodes and mars2grib).

For each backend, this test:
  1. Prints the exact MARS request and polytope BoundingBox request
  2. Retrieves the reference GRIB via `polytope retrieve` with `area`
  3. Retrieves CoverageJSON via `polytope retrieve` with `feature=boundingbox`
  4. Converts CovJSON → GRIB via `Covjsonkit.decode(...).to_grib(backend=...)`
  5. Compares the two GRIB files (metadata + values) and reports diffs
  6. Asserts they match within tolerance

Prerequisites:
  - ~/.polytopeapirc with valid credentials
  - Network access to polytope.ecmwf.int
  - eccodes (Python) always; pymars2grib for the mars2grib backend
  - For mars2grib: `source mars2grib-bundle/env.sh` before invoking pytest

Run:
  export RUN_INTEGRATION_TESTS=1
  source mars2grib-bundle/env.sh   # only needed for the mars2grib parametrisation
  .venv311/bin/pytest tests/test_grib_mars_vs_bbox_backends.py -v -s
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

# ---------------------------------------------------------------------------
# Request configuration — identical MARS keys used across both paths.
# Small equatorial box keeps GRIB messages tiny and easy to diff.
# ---------------------------------------------------------------------------

AREA = [0.18, 0.0, 0.03, 0.15]  # N / W / S / E (degrees)

MARS_KEYS = {
    "class": "od",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "param": "2t/10v",
    "step": "0",
    "date": "-1",  # yesterday
    "time": "0000",
}

COLLECTION = "ecmwf-mars"

# Value-comparison tolerance. GRIB2 with bitsPerValue=16 quantises to
# ~2e-5 K; both backends use identical section-3+7 encoding paths so
# they must land in the same bucket.
VALUE_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# GRIB helpers
# ---------------------------------------------------------------------------


def _read_all_grib_messages(filepath):
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
                for k in (
                    "latitudeOfFirstGridPointInDegrees",
                    "longitudeOfFirstGridPointInDegrees",
                    "latitudeOfLastGridPointInDegrees",
                    "longitudeOfLastGridPointInDegrees",
                ):
                    try:
                        msg[k] = eccodes.codes_get(gid, k)
                    except Exception:
                        pass
                messages.append(msg)
            finally:
                eccodes.codes_release(gid)
    return messages


# ---------------------------------------------------------------------------
# Polytope retrieval helpers
# ---------------------------------------------------------------------------


def _mars_area_request():
    """Exact dict sent to polytope for the MARS `area` path."""
    return {**MARS_KEYS, "area": f"{AREA[0]}/{AREA[1]}/{AREA[2]}/{AREA[3]}"}


def _boundingbox_request():
    """Exact dict sent to polytope for the BoundingBox feature path."""
    return {
        **MARS_KEYS,
        "feature": {
            "type": "boundingbox",
            "points": [[AREA[0], AREA[1]], [AREA[2], AREA[3]]],
        },
    }


def _retrieve_mars_area(output_path):
    from polytope.api import Client

    request = _mars_area_request()
    print("\n" + "=" * 70)
    print("PATH 1 — MARS `area` retrieve")
    print("=" * 70)
    print(f"Collection: {COLLECTION}")
    print(f"Request:    {json.dumps(request, indent=2)}")
    print(f"Output:     {output_path}")

    Client().retrieve(COLLECTION, request, output_file=output_path, pointer=False)
    print(f"Retrieved   {os.path.getsize(output_path)} bytes")


def _retrieve_boundingbox_covjson():
    from polytope.api import Client

    request = _boundingbox_request()
    print("\n" + "=" * 70)
    print("PATH 2 — polytope BoundingBox → CoverageJSON → to_grib()")
    print("=" * 70)
    print(f"Collection: {COLLECTION}")
    print(f"Request:    {json.dumps(request, indent=2)}")

    with tempfile.NamedTemporaryFile(suffix=".covjson", delete=False, mode="w") as tmp:
        cj_path = tmp.name
    try:
        Client().retrieve(COLLECTION, request, output_file=cj_path)
        print(f"Retrieved   {os.path.getsize(cj_path)} bytes CoverageJSON")
        with open(cj_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(cj_path):
            os.unlink(cj_path)


def _covjson_to_grib(covjson, output_path, backend):
    from covjsonkit.api import Covjsonkit

    print(f"\nConverting CoverageJSON → GRIB (backend={backend!r})")
    decoder = Covjsonkit().decode(covjson)
    decoder.to_grib(output_path, backend=backend)
    print(f"Wrote       {os.path.getsize(output_path)} bytes to {output_path}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _compare_grib_files(mars_path, ours_path, backend):
    mars = _read_all_grib_messages(mars_path)
    ours = _read_all_grib_messages(ours_path)

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"COMPARISON — backend={backend!r}")
    lines.append("=" * 70)
    lines.append(f"MARS messages:      {len(mars)}")
    lines.append(f"CovJSON→GRIB msgs:  {len(ours)}")

    all_match = True
    if len(mars) != len(ours):
        lines.append(f"❌ message count mismatch: {len(mars)} vs {len(ours)}")
        all_match = False

    mars_params = sorted({m["shortName"] for m in mars})
    ours_params = sorted({m["shortName"] for m in ours})
    lines.append(f"MARS params:        {mars_params}")
    lines.append(f"Ours params:        {ours_params}")
    if mars_params != ours_params:
        lines.append("❌ parameter mismatch")
        all_match = False

    for mm in mars:
        key = (mm["shortName"], mm["stepRange"])
        cand = [m for m in ours if (m["shortName"], m["stepRange"]) == key]
        if not cand:
            lines.append(f"\n❌ no match for {key}")
            all_match = False
            continue
        om = cand[0]
        lines.append(f"\n--- {key[0]} step={key[1]} ---")

        if mm["numberOfDataPoints"] != om["numberOfDataPoints"]:
            lines.append(f"  ❌ point count: MARS={mm['numberOfDataPoints']} vs ours={om['numberOfDataPoints']}")
            all_match = False
            continue
        lines.append(f"  ✓ point count: {mm['numberOfDataPoints']}")

        if mm["gridType"] != om["gridType"]:
            lines.append(f"  ⚠ gridType differs: MARS={mm['gridType']} vs ours={om['gridType']}")

        diffs = [abs(a - b) for a, b in zip(mm["values"], om["values"])]
        max_d = max(diffs) if diffs else 0.0
        mean_d = (sum(diffs) / len(diffs)) if diffs else 0.0
        n_exact = sum(1 for d in diffs if d == 0.0)
        lines.append(f"  values: max_diff={max_d:.3e}  mean={mean_d:.3e}  " f"exact={n_exact}/{len(diffs)}")
        if max_d > VALUE_TOLERANCE:
            lines.append(f"  ❌ max_diff {max_d:.3e} exceeds tolerance {VALUE_TOLERANCE:.0e}")
            all_match = False

        for bk in (
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "latitudeOfLastGridPointInDegrees",
            "longitudeOfLastGridPointInDegrees",
        ):
            if bk in mm and bk in om:
                d = abs(mm[bk] - om[bk])
                mark = "✓" if d < 1e-3 else "⚠"
                lines.append(f"  {mark} {bk}: MARS={mm[bk]:.6f} ours={om[bk]:.6f} (Δ={d:.2e})")

    lines.append("\n" + ("✅ MATCH" if all_match else "❌ DIFFERENCES") + f"  (backend={backend!r})")
    return all_match, "\n".join(lines)


# ---------------------------------------------------------------------------
# Parametrised test — one row per backend
# ---------------------------------------------------------------------------


def _backend_available(name):
    if name == "eccodes":
        return True
    if name in ("mars2grib", "mars2grib_native"):
        try:
            import pymars2grib  # noqa: F401

            return True
        except ImportError:
            return False
    return False


@pytest.mark.parametrize("backend", ["eccodes", "mars2grib", "mars2grib_native"])
def test_mars_area_vs_boundingbox_to_grib(backend):
    """MARS `area` output must equal polytope BoundingBox → to_grib(backend)."""
    if not _backend_available(backend):
        pytest.skip(f"backend '{backend}' unavailable (source mars2grib-bundle/env.sh)")

    mars_f = tempfile.NamedTemporaryFile(suffix=f"_mars_{backend}.grib", delete=False)
    ours_f = tempfile.NamedTemporaryFile(suffix=f"_ours_{backend}.grib", delete=False)
    mars_f.close()
    ours_f.close()

    try:
        _retrieve_mars_area(mars_f.name)
        covjson = _retrieve_boundingbox_covjson()
        _covjson_to_grib(covjson, ours_f.name, backend=backend)

        match, report = _compare_grib_files(mars_f.name, ours_f.name, backend)
        print(report)
        assert match, f"backend={backend}: GRIB files differ — see report above"
    finally:
        for p in (mars_f.name, ours_f.name):
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    """Run directly (bypasses pytest for quick manual iteration)."""
    print("MARS area vs BoundingBox→to_grib comparison")
    print(f"AREA: N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}")

    os.environ["RUN_INTEGRATION_TESTS"] = "1"
    mars_path = "/tmp/mars_area_ref.grib"
    _retrieve_mars_area(mars_path)
    covjson = _retrieve_boundingbox_covjson()

    for backend in ("eccodes", "mars2grib", "mars2grib_native"):
        if not _backend_available(backend):
            print(f"\n[skip] backend={backend} unavailable")
            continue
        ours_path = f"/tmp/covjson_{backend}.grib"
        _covjson_to_grib(covjson, ours_path, backend=backend)
        match, report = _compare_grib_files(mars_path, ours_path, backend)
        print(report)
        print(f"\n→ files preserved: {mars_path}, {ours_path}")
