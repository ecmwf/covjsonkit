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
        coverage["domain"]["axes"]["x"]["values"] = [coords["x"]]
        coverage["domain"]["axes"]["y"]["values"] = [coords["y"]]
        coverage["domain"]["axes"]["z"]["values"] = [coords["z"]]
        coverage["domain"]["axes"]["t"]["values"] = [coords["t"]]

    def add_range(self, coverage, values):
        for parameter in self.parameters:
            coverage["ranges"][parameter] = {}
            coverage["ranges"][parameter]["type"] = "NdArray"
            coverage["ranges"][parameter]["dataType"] = "float"
            coverage["ranges"][parameter]["shape"] = [values[parameter].shape[0]]
            coverage["ranges"][parameter]["axisNames"] = ["t"]
            coverage["ranges"][parameter]["values"] = values[
                parameter
            ]  # [values[parameter]]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def from_xarray(self, dataset):
        for parameter in dataset.data_vars:
            if parameter == "Temperature":
                self.add_parameter(
                    "t",
                    {
                        "type": "Parameter",
                        "description": "Temperature",
                        "unit": {"symbol": "K"},
                        "observedProperty": {"id": "t", "label": {"en": "Temperature"}},
                    },
                )
            elif parameter == "Pressure":
                self.add_parameter(
                    "p",
                    {
                        "type": "Parameter",
                        "description": "Pressure",
                        "unit": {"symbol": "pa"},
                        "observedProperty": {"id": "p", "label": {"en": "Pressure"}},
                    },
                )
        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        for fc_time in dataset["fct"]:
            self.add_coverage(
                {
                    "date": fc_time.values.astype("M8[ms]")
                    .astype("O")
                    .strftime("%m/%d/%Y"),
                    "type": "forecast",
                    "step": 0,
                },
                {
                    "x": dataset["x"].values,
                    "y": dataset["y"].values,
                    "z": dataset["z"].values,
                    "t": dataset["t"].values,
                },
                {
                    "t": dataset["Temperature"].sel(fct=fc_time).values[0][0][0],
                    "p": dataset["Pressure"].sel(fct=fc_time).values[0][0][0],
                },
            )
        return self.covjson


"""
<xarray.Dataset>
Dimensions:      (x: 1, y: 1, z: 1, fct: 2, t: 3)
Coordinates:
  * x            (x) int64 3
  * y            (y) int64 7
  * z            (z) int64 1
  * fct          (fct) datetime64[ns] 2017-01-01 2017-01-02
  * t            (t) datetime64[ns] 2017-01-02 ... 2017-01-02T12:00:00
Data variables:
    Temperature  (x, y, z, fct, t) float64 264.9 263.8 265.1 263.8 265.1 264.9
    Pressure     (x, y, z, fct, t) float64 9.931 7.831 14.12 13.83 14.12 7.931
Attributes:
    class:    od
    stream:   oper
    levtype:  pl
    number:   0
<xarray.DataArray 'Temperature' (x: 1, y: 1, z: 1, fct: 2, t: 3)>
array([[[[[264.93115234, 263.83115234, 265.12313132],
          [263.83115234, 265.12313132, 264.93115234]]]]])
          """
