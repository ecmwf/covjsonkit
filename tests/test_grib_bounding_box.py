"""Tests for BoundingBox.to_grib() — CoverageJSON → GRIB conversion."""

import copy
import json
import os
import tempfile

import pytest

eccodes = pytest.importorskip("eccodes", reason="eccodes required for GRIB tests")

from covjsonkit.api import Covjsonkit  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oper_covjson():
    """Load the oper (no-ensemble) multipoint test fixture."""
    path = os.path.join(os.path.dirname(__file__), "data", "test_oper_multipoint_coverage.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def enfo_covjson():
    """Load the enfo (ensemble) multipoint test fixture."""
    path = os.path.join(os.path.dirname(__file__), "data", "test_multipoint_coverage.json")
    with open(path) as f:
        return json.load(f)


def _read_grib_messages(filepath):
    """Read all GRIB messages from a file and return a list of (keys, values) dicts."""
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
                    "gridType": eccodes.codes_get(gid, "gridType"),
                    "numberOfDataPoints": eccodes.codes_get(gid, "numberOfDataPoints"),
                    "values": eccodes.codes_get_values(gid).tolist(),
                }
                # Optional keys that may not be in every message
                try:
                    msg["marsClass"] = eccodes.codes_get(gid, "marsClass")
                except Exception:
                    pass
                try:
                    msg["marsStream"] = eccodes.codes_get(gid, "marsStream")
                except Exception:
                    pass
                try:
                    msg["marsType"] = eccodes.codes_get(gid, "marsType")
                except Exception:
                    pass
                try:
                    msg["typeOfLevel"] = eccodes.codes_get(gid, "typeOfLevel")
                except Exception:
                    pass
                try:
                    msg["perturbationNumber"] = eccodes.codes_get(gid, "perturbationNumber")
                except Exception:
                    msg["perturbationNumber"] = 0
                try:
                    msg["stepRange"] = eccodes.codes_get(gid, "stepRange")
                except Exception:
                    pass
                messages.append(msg)
            finally:
                eccodes.codes_release(gid)
    return messages


# ---------------------------------------------------------------------------
# Step 6 stub: other decoders raise NotImplementedError
# ---------------------------------------------------------------------------


class TestToGribStubs:
    """Verify that non-BoundingBox decoders raise NotImplementedError."""

    def test_timeseries_raises(self):
        from covjsonkit.decoder.TimeSeries import TimeSeries

        with pytest.raises(NotImplementedError):
            # We just need to call to_grib on the class — but we can't
            # instantiate without valid covjson, so test the method directly.
            TimeSeries.to_grib(None)

    def test_vertical_profile_raises(self):
        from covjsonkit.decoder.VerticalProfile import VerticalProfile

        with pytest.raises(NotImplementedError):
            VerticalProfile.to_grib(None)


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


class TestToGribOper:
    """Tests using the oper (no-ensemble) fixture."""

    def test_message_count(self, oper_covjson):
        """1 coverage × 2 params = 2 GRIB messages."""
        decoder = Covjsonkit().decode(oper_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            assert len(messages) == 2
        finally:
            os.unlink(output)

    def test_roundtrip_values(self, oper_covjson):
        """Values in GRIB should match the CoverageJSON input (within packing tolerance)."""
        decoder = Covjsonkit().decode(oper_covjson)
        original_2t = oper_covjson["coverages"][0]["ranges"]["2t"]["values"]
        original_10v = oper_covjson["coverages"][0]["ranges"]["10v"]["values"]

        # Compute expected N→S/W→E sort order (same as to_grib applies)
        coords = oper_covjson["coverages"][0]["domain"]["axes"]["composite"]["values"]
        from covjsonkit.decoder.BoundingBox import BoundingBox

        sort_idx = BoundingBox._nswe_sort_indices(coords)
        expected_2t = [original_2t[i] for i in sort_idx]
        expected_10v = [original_10v[i] for i in sort_idx]

        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)

            # Find messages by paramId
            msg_2t = next(m for m in messages if m["paramId"] == 167)
            msg_10v = next(m for m in messages if m["paramId"] == 166)

            assert len(msg_2t["values"]) == len(expected_2t)
            assert len(msg_10v["values"]) == len(expected_10v)

            # GRIB packing introduces small rounding — allow tolerance
            for orig, decoded in zip(expected_2t, msg_2t["values"]):
                assert abs(orig - decoded) < 0.01, f"2t mismatch: {orig} vs {decoded}"
            for orig, decoded in zip(expected_10v, msg_10v["values"]):
                assert abs(orig - decoded) < 0.01, f"10v mismatch: {orig} vs {decoded}"
        finally:
            os.unlink(output)

    def test_mars_keys(self, oper_covjson):
        """GRIB keys should match the mars:metadata from CoverageJSON."""
        decoder = Covjsonkit().decode(oper_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)

            for msg in messages:
                assert msg["dataDate"] == 20250623
                assert msg["dataTime"] == 0
                # MARS levtype=sfc maps to different eccodes typeOfLevel
                # depending on the parameter (e.g. "surface" for sp,
                # "heightAboveGround" for 2t/10v)
                assert msg["typeOfLevel"] in ("surface", "heightAboveGround")
                assert msg["numberOfDataPoints"] == 9
        finally:
            os.unlink(output)

    def test_grid_type_defaults(self, oper_covjson):
        """Without mars:grid, the decoder should apply reduced_gg defaults."""
        decoder = Covjsonkit().decode(oper_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            for msg in messages:
                assert msg["gridType"] == "reduced_gg"
        finally:
            os.unlink(output)

    def test_output_path_returned(self, oper_covjson):
        """to_grib() should return the output path."""
        decoder = Covjsonkit().decode(oper_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            result = decoder.to_grib(output, backend="eccodes")
            assert result == output
        finally:
            os.unlink(output)


class TestToGribEnfo:
    """Tests using the enfo (ensemble) fixture."""

    def test_message_count(self, enfo_covjson):
        """4 coverages (2 numbers × 2 steps) × 2 params = 8 GRIB messages."""
        decoder = Covjsonkit().decode(enfo_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            assert len(messages) == 8
        finally:
            os.unlink(output)

    def test_ensemble_numbers_present(self, enfo_covjson):
        """Each ensemble member should have perturbationNumber set."""
        decoder = Covjsonkit().decode(enfo_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            perturb_nums = {m["perturbationNumber"] for m in messages}
            assert 1 in perturb_nums
            assert 2 in perturb_nums
        finally:
            os.unlink(output)

    def test_per_field_uniqueness(self, enfo_covjson):
        """Each GRIB message should represent a unique (param, step, number) combo."""
        decoder = Covjsonkit().decode(enfo_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            combos = set()
            for msg in messages:
                key = (msg["paramId"], msg.get("stepRange", "0"), msg["perturbationNumber"])
                combos.add(key)
            # Should have as many unique combos as messages
            assert len(combos) == len(messages)
        finally:
            os.unlink(output)


class TestToGribExplicitGrid:
    """Test with explicit mars:grid metadata (as polytope-mars would provide from config)."""

    def test_octahedral_grid_from_config(self, oper_covjson):
        """mars:grid from octahedral mapper config (type=octahedral, resolution=1280)."""
        covjson = copy.deepcopy(oper_covjson)
        # This is what polytope-mars _get_grid_metadata() produces for:
        #   axis_name: values, transformations: [{name: mapper, type: octahedral, resolution: 1280}]
        covjson["coverages"][0]["mars:grid"] = {
            "gridType": "reduced_gg",
            "N": 1280,
        }

        decoder = Covjsonkit().decode(covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            for msg in messages:
                assert msg["gridType"] == "reduced_gg"
                assert msg["numberOfDataPoints"] == 9
        finally:
            os.unlink(output)

    def test_regular_ll_grid_from_config(self, oper_covjson):
        """mars:grid from local_regular mapper config (e.g. EFAS)."""
        covjson = copy.deepcopy(oper_covjson)
        # This is what polytope-mars _get_grid_metadata() produces for:
        #   axis_name: values, transformations: [{name: mapper, type: local_regular,
        #     resolution: [2969, 4529], local: [22.76, 72.24, -25.24, 50.24]}]
        covjson["coverages"][0]["mars:grid"] = {
            "gridType": "regular_ll",
            "Ni": 3,
            "Nj": 3,
            "Dx": 0.07,
            "Dy": 0.07,
            "area": [0.18, 0.0, 0.03, 0.15],
        }

        decoder = Covjsonkit().decode(covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            for msg in messages:
                assert msg["gridType"] == "regular_ll"
        finally:
            os.unlink(output)

    def test_no_grid_metadata_uses_defaults(self, oper_covjson):
        """Without mars:grid, decoder falls back to reduced_gg N=1280 defaults."""
        # oper_covjson has no mars:grid key — verify it uses defaults
        assert "mars:grid" not in oper_covjson["coverages"][0]

        decoder = Covjsonkit().decode(oper_covjson)
        with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as tmp:
            output = tmp.name
        try:
            decoder.to_grib(output, backend="eccodes")
            messages = _read_grib_messages(output)
            for msg in messages:
                # Defaults applied
                assert msg["gridType"] == "reduced_gg"
        finally:
            os.unlink(output)


class TestBackendSelection:
    """Test backend factory behaviour."""

    def test_auto_selects_eccodes(self):
        """With pymars2grib unavailable, auto should fall back to eccodes."""
        from covjsonkit.decoder.grib_backends import get_backend
        from covjsonkit.decoder.grib_backends.eccodes_backend import EccodesBackend

        backend = get_backend("auto")
        assert isinstance(backend, EccodesBackend)

    def test_explicit_eccodes(self):
        """Explicitly requesting eccodes should work."""
        from covjsonkit.decoder.grib_backends import get_backend
        from covjsonkit.decoder.grib_backends.eccodes_backend import EccodesBackend

        backend = get_backend("eccodes")
        assert isinstance(backend, EccodesBackend)

    def test_mars2grib_raises_when_unavailable(self):
        """Explicitly requesting mars2grib should raise if not installed."""
        from covjsonkit.decoder.grib_backends import get_backend

        with pytest.raises(ImportError, match="pymars2grib"):
            get_backend("mars2grib")


class TestBuildHelpers:
    """Unit tests for the private helper methods on BoundingBox."""

    def test_build_mars_dict_date_parsing(self, oper_covjson):
        """_build_mars_dict should parse ISO-8601 Forecast date correctly."""
        from covjsonkit.decoder.BoundingBox import BoundingBox

        coverage = oper_covjson["coverages"][0]
        mars_metadata = coverage["mars:metadata"]
        mars = BoundingBox._build_mars_dict(mars_metadata, coverage)

        assert mars["date"] == "20250623"
        assert mars["time"] == "0000"
        assert mars["class"] == "od"
        assert mars["stream"] == "oper"
        assert mars["type"] == "an"
        assert mars["step"] == "0"

    def test_build_misc_dict_defaults(self, oper_covjson):
        """_build_misc_dict should apply oper defaults when mars:grid is absent."""
        from covjsonkit.decoder.BoundingBox import BoundingBox

        coverage = oper_covjson["coverages"][0]
        misc = BoundingBox._build_misc_dict({}, coverage)

        assert misc["gridType"] == "reduced_gg"
        assert misc["N"] == 1280
        assert "area" in misc
        # Area should be [max_lat, min_lon, min_lat, max_lon]
        assert misc["area"][0] > misc["area"][2]  # N > S

    def test_build_misc_dict_explicit_grid(self, oper_covjson):
        """_build_misc_dict should use explicit grid metadata when provided."""
        from covjsonkit.decoder.BoundingBox import BoundingBox

        coverage = oper_covjson["coverages"][0]
        grid_meta = {"gridType": "regular_ll", "Ni": 10, "Nj": 10}
        misc = BoundingBox._build_misc_dict(grid_meta, coverage)

        assert misc["gridType"] == "regular_ll"
        assert misc["Ni"] == 10
        # Should NOT have oper defaults
        assert "N" not in misc

    def test_shortname_to_param_id(self, oper_covjson):
        """_shortname_to_param_id should resolve known shortnames."""
        decoder = Covjsonkit().decode(oper_covjson)
        assert decoder._shortname_to_param_id("2t") == "167"
        assert decoder._shortname_to_param_id("10v") == "166"

    def test_shortname_to_param_id_unknown(self, oper_covjson):
        """Unknown shortnames should pass through as-is."""
        decoder = Covjsonkit().decode(oper_covjson)
        result = decoder._shortname_to_param_id("totally_unknown_param_xyz")
        assert result == "totally_unknown_param_xyz"
