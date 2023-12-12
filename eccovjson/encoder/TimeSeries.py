from .encoder import Encoder
import xarray as xr
from datetime import timedelta, datetime
import datetime


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
        coverage["domain"]["axes"]["x"]["values"] = coords["x"]
        coverage["domain"]["axes"]["y"]["values"] = coords["y"]
        coverage["domain"]["axes"]["z"]["values"] = coords["z"]
        coverage["domain"]["axes"]["t"]["values"] = coords["t"]

    def add_range(self, coverage, values):
        for parameter in self.parameters:
            coverage["ranges"][parameter] = {}
            coverage["ranges"][parameter]["type"] = "NdArray"
            coverage["ranges"][parameter]["dataType"] = "float"
            coverage["ranges"][parameter]["shape"] = [len(values[parameter])]
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
        for num in dataset["number"].values:
            self.add_coverage(
                {
                    # "date": fc_time.values.astype("M8[ms]")
                    # .astype("O")
                    # .strftime("%m/%d/%Y"),
                    "number": num,
                    "type": "forecast",
                    "step": 0,
                },
                {
                    "x": list(dataset["x"].values),
                    "y": list(dataset["y"].values),
                    "z": list(dataset["z"].values),
                    "t": [str(x) for x in dataset["t"].values],
                },
                {
                    "t": list(dataset["Temperature"].sel(number=num).values[0][0][0]),
                    # "p": dataset["Pressure"].sel(fct=fc_time).values[0][0][0],
                },
            )
        return self.covjson

    def from_polytope(self, result, request):
        # ancestors = [val.get_ancestors() for val in result.leaves]
        values = [val.result for val in result.leaves]

        mars_metadata = {}
        coords = {}
        for key in request.keys():
            if (
                key != "latitude"
                and key != "longitude"
                and key != "param"
                and key != "number"
                and key != "step"
            ):
                mars_metadata[key] = request[key]
            elif key == "latitude":
                coords["x"] = [request[key]]
            elif key == "longitude":
                coords["y"] = [request[key]]

        if request["param"] == "167":
            self.add_parameter(
                "t",
                {
                    "type": "Parameter",
                    "description": "Temperature",
                    "unit": {"symbol": "K"},
                    "observedProperty": {"id": "t", "label": {"en": "Temperature"}},
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

        coords["z"] = ["sfc"]
        if "/" in request["number"]:
            numbers = request["number"].split("/")
        else:
            numbers = request["number"]
        steps = request["step"]

        times = []
        date_format = "%Y%m%dT%H%M%S"
        start_time = datetime.datetime.strptime(mars_metadata["date"], date_format)
        for step in steps:
            # add current date to list by converting  it to iso format
            stamp = start_time + timedelta(hours=step)
            times.append(stamp.isoformat())
            # increment start date by timedelta

        coords["t"] = times
        vals = []
        start = 0
        end = len(times)
        new_metadata = mars_metadata.copy()
        for num in numbers:
            mars_metadata["number"] = num
            new_metadata = mars_metadata.copy()
            self.add_coverage(new_metadata, coords, {"t": values[start:end]})
            # vals.append(values[start:end])
            start = end
            end += len(times)

        return self.covjson
