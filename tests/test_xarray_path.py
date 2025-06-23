import json
import os

from covjsonkit.api import Covjsonkit


class TestPathXarray:
    def setup_method(self):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "data/test_path_coverage.json")

        with open(file_path, "r") as f:
            result = json.load(f)
        self.test_covjson = result

    def test_to_xarray(self):
        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()
        assert ds.latitude.values[0] == -0.105448152647
        assert ds.longitude.values[0] == 0.0
        assert len(ds.number.values) == 1
        assert len(ds.steps.values) == 1
        encoder_obj = Covjsonkit().encode("CoverageCollection", "Path")
        assert encoder_obj.covjson["type"] == "CoverageCollection"

    def test_from_xarray(self):
        decoder_obj = Covjsonkit().decode(self.test_covjson)
        ds = decoder_obj.to_xarray()

        encoder_obj = Covjsonkit().encode("CoverageCollection", "Path")
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
