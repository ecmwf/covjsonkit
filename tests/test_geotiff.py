import json
import os

from covjsonkit.api import Covjsonkit


class TestGeotiffConversion:
    def setup_method(self):
        current_dir = os.path.dirname(__file__)
        self.path = os.path.join(current_dir, "data/test_geotiff.json")
        with open(self.path) as f:
            self.covjson = json.load(f)

    def test_geotiff_multipoint(self):
        cov = Covjsonkit().decode(self.covjson)
        mp = cov.to_geotiff()
        os.remove("multipoint_2t.tif")
        os.remove("multipoint_10u.tif")

