from .encoder import Encoder
import xarray as xr
import datetime as dt


class TimeSeries(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)

    def add_coverage(self, mars_metadata, coords, values):
        new_coverage = {}
        new_coverage["mars:metadata"] = {}
        new_coverage["type"] = "Coverage"
        new_coverage["domain"] = {}
        new_coverage["ranges"] = {}
        self.add_mars_metadata(new_coverage, mars_metadata)
        self.add_domain(new_coverage, coords)
        self.add_range(new_coverage, values)
        self.covjson["coverages"].append(new_coverage)

    def add_domain(self, coverage, coords):
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = {}
        coverage["domain"]["axes"]["x"] = {}
        coverage["domain"]["axes"]["y"] = {}
        coverage["domain"]["axes"]["z"] = {}
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["x"]["values"] = []  # [coords["x"]]
        coverage["domain"]["axes"]["y"]["values"] = []  # [coords["y"]]
        coverage["domain"]["axes"]["z"]["values"] = []  # [coords["z"]]
        coverage["domain"]["axes"]["t"]["values"] = []  # [coords["t"]]

    def add_range(self, coverage, values):
        for parameter in self.parameters:
            coverage["ranges"][parameter] = {}
            coverage["ranges"][parameter]["type"] = "NdArray"
            coverage["ranges"][parameter]["dataType"] = "float"
            coverage["ranges"][parameter]["shape"] = []
            coverage["ranges"][parameter]["axisNames"] = ["t"]
            coverage["ranges"][parameter]["values"] = []  # [values[parameter]]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def from_xarray(self, dataset):
        pass
