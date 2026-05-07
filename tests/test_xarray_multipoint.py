import json
import os

from covjsonkit.api import Covjsonkit


class TestMultipointXarray:
    def setup_method(self):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "data/test_multipoint_coverage.json")

        with open(file_path, "r") as f:
            result = json.load(f)
        self.test_covjson = result

    def test_to_xarray(self):
        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()
        assert ds.latitude.values[0] == 0.035149384216
        assert ds.longitude.values[0] == 0.0
        assert len(ds.number.values) == 2
        assert len(ds.steps.values) == 2
        encoder_obj = Covjsonkit().encode("CoverageCollection", "BoundingBox")
        assert encoder_obj.covjson["type"] == "CoverageCollection"

    def test_from_xarray(self):
        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()

        encoder_obj = Covjsonkit().encode("CoverageCollection", "BoundingBox")
        covjson_result = encoder_obj.from_xarray(ds)

        assert covjson_result["type"] == self.test_covjson["type"]
        assert len(covjson_result["coverages"]) == len(self.test_covjson["coverages"])
        assert (
            covjson_result["coverages"][0]["domain"]["axes"]["composite"]["values"]
            == self.test_covjson["coverages"][0]["domain"]["axes"]["composite"]["values"]
        )
        assert (
            covjson_result["coverages"][0]["mars:metadata"]["number"]
            == self.test_covjson["coverages"][0]["mars:metadata"]["number"]
        )
        assert (
            covjson_result["coverages"][0]["ranges"]["2t"]["values"][0]
            == self.test_covjson["coverages"][0]["ranges"]["2t"]["values"][0]
        )

    def test_to_xarray_param_t_no_collision(self):
        """to_xarray works with param 't' - no collision since dims use 'datetimes'."""
        # Add a param named 't' to verify no collision
        self.test_covjson["parameters"]["t"] = {
            "type": "Parameter",
            "description": {"en": "Temperature"},
            "unit": {"symbol": "K"},
            "observedProperty": {"id": "t", "label": {"en": "Temperature"}},
        }
        for cov in self.test_covjson["coverages"]:
            cov["ranges"]["t"] = cov["ranges"]["2t"].copy()

        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()

        # Param 't' should be in data_vars (no collision with 'datetimes' dim)
        assert "t" in ds.data_vars, f"Expected 't' in data_vars, got {list(ds.data_vars)}"
        # Time dimension is 'datetimes', not 't'
        assert "datetimes" in ds.dims
