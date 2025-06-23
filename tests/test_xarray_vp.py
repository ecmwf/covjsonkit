import json
import os

from covjsonkit.api import Covjsonkit


class TestVerticalProfileXarray:
    def setup_method(self):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "data/test_verticalprofile_coverage.json")

        with open(file_path, "r") as f:
            result = json.load(f)
        self.test_covjson = result

    def test_to_xarray(self):
        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()
        assert len(ds) == 2
        assert ds[0].latitude.values[0] == -0.035149384216
        assert ds[0].longitude.values[0] == 0.981308411215
        assert len(ds[0].number.values) == 2
        assert len(ds[0].time.values) == 2
        encoder_obj = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        assert encoder_obj.covjson["type"] == "CoverageCollection"

    def test_from_xarray(self):
        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()

        encoder_obj = Covjsonkit().encode("CoverageCollection", "VerticalProfile")
        covjson_result = encoder_obj.from_xarray(ds)

        assert covjson_result["type"] == self.test_covjson["type"]
        assert len(covjson_result["coverages"]) == len(self.test_covjson["coverages"])
        assert (
            covjson_result["coverages"][0]["domain"]["axes"]["latitude"]["values"][0]
            == self.test_covjson["coverages"][0]["domain"]["axes"]["latitude"]["values"][0]
        )
        assert (
            covjson_result["coverages"][0]["domain"]["axes"]["longitude"]["values"][0]
            == self.test_covjson["coverages"][0]["domain"]["axes"]["longitude"]["values"][0]
        )
        assert (
            covjson_result["coverages"][0]["mars:metadata"]["number"]
            == self.test_covjson["coverages"][0]["mars:metadata"]["number"]
        )
        assert (
            covjson_result["coverages"][0]["ranges"]["q"]["values"][0]
            == self.test_covjson["coverages"][0]["ranges"]["q"]["values"][0]
        )
