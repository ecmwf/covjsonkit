import pandas as pd
import xarray as xr

from .decoder import Decoder


class Position(Decoder):
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
            x = domain["axes"][self.x_name]["values"][0]
            y = domain["axes"][self.y_name]["values"][0]
            z = domain["axes"][self.z_name]["values"][0]
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

    def to_geotiff(self):
        raise TypeError("Timeseries domain cannot be converted to GeoTIFF.")

    def to_geojson(self):
        features = []
        for coverage in self.covjson["coverages"]:
            lat = coverage["domain"]["axes"][self.x_name]["values"][0]
            lon = coverage["domain"]["axes"][self.y_name]["values"][0]
            z = coverage["domain"]["axes"][self.z_name]["values"][0]
            datetimes = coverage["domain"]["axes"]["t"]["values"]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]

            values = {}
            for key in coverage["ranges"]:
                values[key] = coverage["ranges"][key]["values"]

            for idx, datetime in enumerate(datetimes):
                param_vals = {}
                for key in values.keys():
                    param_vals[key] = values[key][idx]
                param_vals["datetime"] = datetime
                if "mars:metadata" in coverage:
                    param_vals["mars:metadata"] = mars_metadata
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat, z]},
                        "properties": param_vals,
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    # function to convert covjson to xarray dataset
    def to_xarray(self):
        dims = ["latitude", "longitude", "levelist", "number", "datetime", "t"]
        ds = []

        unique_coords = set()  # To track unique coordinate tuples
        unique_domains = []  # To store unique domains

        for domain in self.domains:
            # Extract coordinate values
            x = domain["axes"][self.x_name]["values"][0]
            y = domain["axes"][self.y_name]["values"][0]
            z = domain["axes"][self.z_name]["values"][0]
            t = tuple(domain["axes"]["t"]["values"])  # Use tuple for hashable type

            # Create a unique identifier for the domain
            coord_tuple = (x, y, z, t)

            # Check if this coordinate combination is already seen
            if coord_tuple not in unique_coords:
                unique_coords.add(coord_tuple)  # Mark as seen
                unique_domains.append(domain)  # Add to unique domains

        all_coords = unique_domains

        num = []
        datetime = []
        for coverage in self.covjson["coverages"]:
            num.append(coverage["mars:metadata"]["number"])
            datetime.append(coverage["mars:metadata"]["Forecast date"])
        nums = list(set(num))
        datetime = list(set(datetime))

        # Process each coordinate domain
        for coords in all_coords:
            dataarraydict = {}
            x = coords["axes"][self.x_name]["values"]
            y = coords["axes"][self.y_name]["values"]
            z = coords["axes"][self.z_name]["values"]
            steps = coords["axes"]["t"]["values"]
            steps = [step.replace("Z", "") for step in steps]
            steps = pd.to_datetime(steps)

            cov_idx_list = self._find_coverages(nums, datetime, x, y, z)

            coords = {
                "latitude": x,
                "longitude": y,
                "levelist": z,
                "number": nums,
                "datetime": datetime,
                "t": steps,
            }

            for parameter in self.parameters:
                param_values = [
                    [[] for _ in range(len(datetime))] for _ in range(len(nums))
                ]

                # Extract parameter values for the current domain
                for i, j, cov in cov_idx_list:
                    param_values[i][j] = cov["ranges"][parameter]["values"]

                long_name = self.get_parameter_metadata(parameter)["observedProperty"][
                    "id"
                ]

                if long_name == "t":
                    long_name = "T"  # Avoid collision with time dimension 't'

                attrs = {
                    "type": self.get_parameter_metadata(parameter)["type"],
                    "units": self.get_parameter_metadata(parameter)["unit"]["symbol"],
                    "long_name": long_name,
                }
                dataarraydict[long_name] = (
                    dims,
                    [[[param_values]]],
                    attrs,
                )

            ds.append(xr.Dataset(data_vars=dataarraydict, coords=coords))

        # Combine all DataArrays into a Dataset
        for mars_metadata in self.mars_metadata[0]:
            if mars_metadata != "date" and mars_metadata != "step":
                for dss in ds:
                    dss.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        if len(ds) == 1:
            return ds[0]

        return ds

    def _find_coverages(self, nums, datetime, x, y, z):
        """Find coverages matching domain parameters and return with indices."""
        result = []
        for i, num in enumerate(nums):
            for j, date in enumerate(datetime):
                for coverage in self.covjson["coverages"]:
                    if self._covers_domain(coverage, num, date, x, y, z):
                        result.append((i, j, coverage))
        return result

    def _covers_domain(self, coverage, num, date, x, y, z):
        """Check if coverage matches the given domain parameters."""
        return (
            coverage["mars:metadata"]["number"] == num
            and coverage["mars:metadata"]["Forecast date"] == date
            and coverage["domain"]["axes"][self.x_name]["values"] == x
            and coverage["domain"]["axes"][self.y_name]["values"] == y
            and coverage["domain"]["axes"][self.z_name]["values"] == z
        )
