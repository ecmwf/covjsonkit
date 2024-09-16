import pandas as pd
import xarray as xr

from .decoder import Decoder


class TimeSeries(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()

    def get_domains(self):
        domains = []
        for coverage in self.coverage.coverages:
            domains.append(coverage["domain"])
        return domains

    def get_ranges(self):
        ranges = []
        for coverage in self.coverage.coverages:
            ranges.append(coverage["ranges"])
        return ranges

    def get_values(self):
        values = {}
        for parameter in self.parameters:
            values[parameter] = []
            for range in self.ranges:
                values[parameter].append(range[parameter]["values"])
        return values

    def get_coordinates(self):
        coord_dict = {}
        for param in self.parameters:
            coord_dict[param] = []
        # Get x,y,z,t coords and unpack t coords and match to x,y,z coords
        for ind, domain in enumerate(self.domains):
            x = domain["axes"]["x"]["values"][0]
            y = domain["axes"]["y"]["values"][0]
            z = domain["axes"]["z"]["values"][0]
            fct = domain["axes"]["t"]["values"][0]
            ts = domain["axes"]["t"]["values"]
            if "number" in self.mars_metadata[ind]:
                num = self.mars_metadata[ind]["number"]
            else:
                num = 0
            for param in self.parameters:
                coords = []
                for t in ts:
                    # Have to replicate these coords for each parameter
                    # coordinates.append([x, y, z, t])
                    coords.append([x, y, z, fct, t, num])
                coord_dict[param].append(coords)
        return coord_dict

    def to_geopandas(self):
        pass

    # function to convert covjson to xarray dataset
    def to_xarray(self):
        dims = ["x", "y", "z", "number", "datetime", "t"]
        dataarraydict = {}

        # Get coordinates
        coords = self.get_domains()
        x = coords[0]["axes"]["x"]["values"]
        y = coords[0]["axes"]["y"]["values"]
        z = coords[0]["axes"]["z"]["values"]
        steps = coords[0]["axes"]["t"]["values"]
        steps = [step.replace("Z", "") for step in steps]
        steps = pd.to_datetime(steps)
        # steps = list(range(len(steps)))

        num = []
        datetime = []
        for coverage in self.covjson["coverages"]:
            num.append(coverage["mars:metadata"]["number"])
            datetime.append(coverage["mars:metadata"]["Forecast date"])

        nums = list(set(num))
        datetime = list(set(datetime))

        param_values = {}

        for parameter in self.parameters:
            param_values[parameter] = []
            for i, num in enumerate(nums):
                param_values[parameter].append([])
                for j, date in enumerate(datetime):
                    param_values[parameter][i].append([])
                    for k, step in enumerate(steps):
                        for coverage in self.covjson["coverages"]:
                            if (
                                coverage["mars:metadata"]["number"] == num
                                and coverage["mars:metadata"]["Forecast date"] == date
                            ):
                                param_values[parameter][i][j] = coverage["ranges"][parameter]["values"]

        for parameter in self.parameters:
            param_coords = {"x": x, "y": y, "z": z, "number": nums, "datetime": datetime, "t": steps}
            dataarray = xr.DataArray(
                [[[param_values[parameter]]]],
                dims=dims,
                coords=param_coords,
                name=parameter,
            )

            dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
            dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        ds = xr.Dataset(dataarraydict)
        for mars_metadata in self.mars_metadata[0]:
            if mars_metadata != "date" and mars_metadata != "step":
                ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        return ds
