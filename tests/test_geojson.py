import json
import os

from covjsonkit.api import Covjsonkit


class TestMultipointXarray:
    def setup_method(self):
        current_dir = os.path.dirname(__file__)
        timeseries = os.path.join(current_dir, "data/test_timeseries_coverage.json")
        with open(timeseries, "r") as f:
            self.timeseries = json.load(f)

        verticalprofile = os.path.join(current_dir, "data/test_verticalprofile_coverage.json")
        with open(verticalprofile, "r") as f:
            self.verticalprofile = json.load(f)

        path = os.path.join(current_dir, "data/test_path_coverage.json")
        with open(path, "r") as f:
            self.path = json.load(f)

        multipoint = os.path.join(current_dir, "data/test_multipoint_coverage.json")
        with open(multipoint, "r") as f:
            self.multipoint = json.load(f)

    def test_geojson_timeseries(self):
        cov = Covjsonkit().decode(self.timeseries)
        ts = cov.to_geojson()
        assert ts["type"] == "FeatureCollection"
        assert len(ts["features"]) == 16

    def test_geojson_verticalprofile(self):
        cov = Covjsonkit().decode(self.verticalprofile)
        vp = cov.to_geojson()
        assert vp["type"] == "FeatureCollection"
        assert len(vp["features"]) == 56

    def test_geojson_path(self):
        cov = Covjsonkit().decode(self.path)
        path = cov.to_geojson()
        assert path["type"] == "FeatureCollection"
        assert len(path["features"]) == 30

    def test_geojson_multipoint(self):
        cov = Covjsonkit().decode(self.multipoint)
        mp = cov.to_geojson()
        assert mp["type"] == "FeatureCollection"
        assert len(mp["features"]) == 36
