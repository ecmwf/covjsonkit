from datetime import datetime as dt
from datetime import timedelta

import xarray as xr

from .decoder import Decoder


class VerticalProfile(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        if "x" in self.covjson["coverages"][0]["domain"]["axes"]:
            self.x_name = "x"
        else:
            self.x_name = "latitude"
        if "y" in self.covjson["coverages"][0]["domain"]["axes"]:
            self.y_name = "y"
        else:
            self.y_name = "longitude"
        if "z" in self.covjson["coverages"][0]["domain"]["axes"]:
            self.z_name = "z"
        else:
            self.z_name = "levelist"

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
            x = domain["axes"][self.x_name]["values"][0]
            y = domain["axes"][self.y_name]["values"][0]
            t = domain["axes"]["t"]["values"]
            zs = domain["axes"][self.z_name]["values"]
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

    def to_geojson(self):
        features = []
        for coverage in self.covjson["coverages"]:
            lat = coverage["domain"]["axes"]["latitude"]["values"][0]
            lon = coverage["domain"]["axes"]["longitude"]["values"][0]
            levels = coverage["domain"]["axes"]["levelist"]["values"]
            datetimes = coverage["domain"]["axes"]["t"]["values"]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]
            values = {}
            for key in coverage["ranges"]:
                values[key] = coverage["ranges"][key]["values"]

            for idx, level in enumerate(levels):
                param_vals = {}
                for key in values.keys():
                    param_vals[key] = values[key][idx]
                param_vals["datetime"] = datetimes[0]
                if "mars:metadata" in coverage:
                    param_vals["mars:metadata"] = mars_metadata
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat, level]},
                        "properties": param_vals,
                    }
                )
        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    def to_xarray(self):
        dims = [
            "latitude",
            "longitude",
            "number",
            "datetime",
            "time",
            "levelist",
        ]
        ds = []

        # Get coordinates for all domains
        all_coords = self.get_domains()

        unique_coords = set()  # To track unique coordinate tuples
        unique_domains = []  # To store unique domains

        for domain in self.domains:
            # Extract coordinate values
            x = domain["axes"][self.x_name]["values"][0]
            y = domain["axes"][self.y_name]["values"][0]
            z = domain["axes"][self.z_name]["values"][0]

            # Create a unique identifier for the domain
            coord_tuple = (x, y, z)

            # Check if this coordinate combination is already seen
            if coord_tuple not in unique_coords:
                unique_coords.add(coord_tuple)  # Mark as seen
                unique_domains.append(domain)  # Add to unique domains

        all_coords = unique_domains
        param_values = {}

        # Initialize parameter values for all parameters
        for parameter in self.parameters:
            param_values[parameter] = []

        for domain_idx, coords in enumerate(all_coords):
            dataarraydict = {}

            # Get coordinates
            x = coords["axes"][self.x_name]["values"]
            y = coords["axes"][self.y_name]["values"]
            levelist = coords["axes"][self.z_name]["values"]
            steps = coords["axes"]["t"]["values"]

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
                            if len(param_values[parameter][domain_idx][i][j]) <= k:
                                param_values[parameter][domain_idx][i][j].append([])
                            for coverage in self.covjson["coverages"]:
                                new_step = (
                                    dt.fromisoformat(date.replace("Z", "")) + timedelta(hours=int(step))
                                ).isoformat() + "Z"
                                if (
                                    coverage["mars:metadata"]["number"] == num
                                    and coverage["mars:metadata"]["Forecast date"] == date
                                    and coverage["domain"]["axes"][self.x_name]["values"][0] == x[0]
                                    and coverage["domain"]["axes"][self.y_name]["values"][0] == y[0]
                                    and coverage["domain"]["axes"]["t"]["values"][0] == new_step
                                ):
                                    param_values[parameter][domain_idx][i][j][k] = coverage["ranges"][parameter][
                                        "values"
                                    ]

            for parameter in self.parameters:
                param_coords = {
                    "latitude": x,
                    "longitude": y,
                    "number": nums,
                    "datetime": datetime,
                    "time": steps,
                    "levelist": list(levelist),
                }
                dataarray = xr.DataArray(
                    [[param_values[parameter][domain_idx]]],
                    dims=dims,
                    coords=param_coords,
                    name=f"{parameter}_domain_{domain_idx}",
                )

                dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
                dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
                dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
                dataarraydict[dataarray.attrs["long_name"]] = dataarray

            ds.append(xr.Dataset(dataarraydict))

        for mars_metadata in self.mars_metadata[0]:
            for dss in ds:
                if mars_metadata != "date" and mars_metadata != "step":
                    dss.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        if len(ds) == 1:
            return ds[0]

        return ds
