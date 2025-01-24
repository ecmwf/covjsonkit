import pandas as pd
import xarray as xr

from .decoder import Decoder


class VerticalProfile(Decoder):
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

    def get_coordinates(self):
        coord_dict = {}
        for param in self.parameters:
            coord_dict[param] = []
        # Get x,y,z coords and unpack z coords and match to x,y coords
        for ind, domain in enumerate(self.domains):
            x = domain["axes"]["x"]["values"][0]
            y = domain["axes"]["y"]["values"][0]
            t = domain["axes"]["t"]["values"]
            zs = domain["axes"]["z"]["values"]
            num = self.mars_metadata[ind]["number"]
            for param in self.parameters:
                coords = []
                for z in zs:
                    # Have to replicate these coords for each parameter
                    # coordinates.append([x, y, z, t])
                    coords.append([x, y, z, num, t])
                coord_dict[param].append(coords)
        return coord_dict

    def get_values(self):
        values = {}
        for parameter in self.parameters:
            values[parameter] = []
            for range in self.ranges:
                values[parameter].append(range[parameter]["values"])
        return values

    def to_geopandas(self):
        pass

    def to_xarray(self):
        dims = [
            "x",
            "y",
            "number",
            "datetime",
            "time",
            "level",
        ]
        dataarraydict = {}

        # Get coordinates
        coords = self.get_domains()
        x = coords[0]["axes"]["x"]["values"]
        y = coords[0]["axes"]["y"]["values"]
        level = coords[0]["axes"]["z"]["values"]
        steps = coords[0]["axes"]["t"]["values"]
        steps = [step.replace("Z", "") for step in steps]
        steps = pd.to_datetime(steps)
        # steps = list(range(len(steps)))

        num = []
        datetime = []
        steps = []
        for coverage in self.covjson["coverages"]:
            num.append(coverage["mars:metadata"]["number"])
            datetime.append(coverage["mars:metadata"]["Forecast date"])
            steps.append(coverage["mars:metadata"]["step"])

        nums = list(set(num))
        datetime = list(set(datetime))
        steps = list(set(steps))

        param_values = {}

        for parameter in self.parameters:
            param_values[parameter] = []
            for i, num in enumerate(nums):
                param_values[parameter].append([])
                for j, date in enumerate(datetime):
                    param_values[parameter][i].append([])
                    for k, step in enumerate(steps):
                        param_values[parameter][i][j].append([])
                        for coverage in self.covjson["coverages"]:
                            if (
                                coverage["mars:metadata"]["number"] == num
                                and coverage["mars:metadata"]["Forecast date"] == date
                                and coverage["mars:metadata"]["step"] == step
                            ):
                                param_values[parameter][i][j][k] = coverage["ranges"][parameter]["values"]

        for parameter in self.parameters:
            param_coords = {
                "x": x,
                "y": y,
                "number": nums,
                "datetime": datetime,
                "time": steps,
                "level": level,
            }

            dataarray = xr.DataArray(
                [[param_values[parameter]]],
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
