import json
import os

from covjsonkit.api import Covjsonkit

current_dir = os.path.dirname(__file__)
timeseries = os.path.join(current_dir, "data/test_timeseries_coverage.json")
with open(timeseries, "r") as f:
    timeseries = json.load(f)


cov = Covjsonkit().decode(timeseries)
ts = cov.to_geojson()
print(ts)
assert ts["type"] == "FeatureCollection"
assert len(ts["features"]) == 16
