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
        ds = []

        # Get coordinates for all domains
        all_coords = self.get_domains()

        unique_coords = set()  # To track unique coordinate tuples
        unique_domains = []  # To store unique domains

        for domain in self.domains:
            # Extract coordinate values
            x = domain["axes"]["x"]["values"][0]
            y = domain["axes"]["y"]["values"][0]
            z = domain["axes"]["z"]["values"][0]
            t = tuple(domain["axes"]["t"]["values"])  # Use tuple for hashable type

            # Create a unique identifier for the domain
            coord_tuple = (x, y, z, t)

            # Check if this coordinate combination is already seen
            if coord_tuple not in unique_coords:
                unique_coords.add(coord_tuple)  # Mark as seen
                unique_domains.append(domain)  # Add to unique domains

        all_coords = unique_domains
        param_values = {}

        # Initialize parameter values for all parameters
        for parameter in self.parameters:
            param_values[parameter] = []

        # Process each coordinate domain
        for domain_idx, coords in enumerate(all_coords):
            dataarraydict = {}
            x = coords["axes"]["x"]["values"]
            y = coords["axes"]["y"]["values"]
            z = coords["axes"]["z"]["values"]
            steps = coords["axes"]["t"]["values"]
            steps = [step.replace("Z", "") for step in steps]
            steps = pd.to_datetime(steps)

            num = []
            datetime = []
            for coverage in self.covjson["coverages"]:
                num.append(coverage["mars:metadata"]["number"])
                datetime.append(coverage["mars:metadata"]["Forecast date"])

            nums = list(set(num))
            datetime = list(set(datetime))

            # Extract parameter values for the current domain
            for parameter in self.parameters:
                if len(param_values[parameter]) <= domain_idx:
                    param_values[parameter].append([])

                for i, num in enumerate(nums):
                    if len(param_values[parameter][domain_idx]) <= i:
                        param_values[parameter][domain_idx].append([])

                    for j, date in enumerate(datetime):
                        if len(param_values[parameter][domain_idx][i]) <= j:
                            param_values[parameter][domain_idx][i].append([])

                        for k, step in enumerate(steps):
                            for coverage in self.covjson["coverages"]:
                                # print(coverage["domain"])
                                # print(coverage["mars:metadata"]["number"], num)
                                # print(coverage["mars:metadata"]["Forecast date"], date)
                                # print(coverage["domain"]["axes"]["x"]["values"][0], x)
                                # print(coverage["domain"]["axes"]["y"]["values"][0], y)
                                # print(coverage["domain"]["axes"]["z"]["values"][0], z)
                                if (
                                    coverage["mars:metadata"]["number"] == num
                                    and coverage["mars:metadata"]["Forecast date"] == date
                                    and coverage["domain"]["axes"]["x"]["values"] == x
                                    and coverage["domain"]["axes"]["y"]["values"] == y
                                    and coverage["domain"]["axes"]["z"]["values"] == z
                                ):
                                    param_values[parameter][domain_idx][i][j] = coverage["ranges"][parameter]["values"]

            for parameter in self.parameters:
                param_coords = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }
                dataarray = xr.DataArray(
                    [[[param_values[parameter][domain_idx]]]],
                    dims=dims,
                    coords=param_coords,
                    name=f"{parameter}_domain_{domain_idx}",
                )

                dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
                dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
                dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
                dataarraydict[dataarray.attrs["long_name"]] = dataarray

            ds.append(xr.Dataset(dataarraydict))

        # Combine all DataArrays into a Dataset
        for mars_metadata in self.mars_metadata[0]:
            for dss in ds:
                if mars_metadata != "date" and mars_metadata != "step":
                    dss.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        if len(ds) == 1:
            return ds[0]

        return ds
