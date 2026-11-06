import pandas as pd
import xarray as xr

from .decoder import Decoder, nulls_as_nan
from .TimeSeries import TimeSeries


class Position(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        first_axes = self.covjson["coverages"][0]["domain"]["axes"]
        # Backwards-compatible axis-name detection: read both spec-compliant
        # coverages (x/y/z) and legacy coverages (longitude/latitude/levelist).
        # Semantics are preserved: x == longitude, y == latitude.
        self.x_name = "x" if "x" in first_axes else "longitude"
        self.y_name = "y" if "y" in first_axes else "latitude"
        if "z" in first_axes:
            self.z_name = "z"
        elif "levelist" in first_axes:
            self.z_name = "levelist"
        else:
            self.z_name = None

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
            longitude = domain["axes"][self.x_name]["values"][0]
            latitude = domain["axes"][self.y_name]["values"][0]
            level = domain["axes"][self.z_name]["values"][0] if self.z_name else None
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
                    coords.append([longitude, latitude, level, fct, t, num])
                coord_dict[param].append(coords)
        return coord_dict

    def to_geopandas(self):
        pass

    def to_geotiff(self):
        raise TypeError("Timeseries domain cannot be converted to GeoTIFF.")

    def to_geojson(self):
        features = []
        for coverage in self.covjson["coverages"]:
            longitude = coverage["domain"]["axes"][self.x_name]["values"][0]
            latitude = coverage["domain"]["axes"][self.y_name]["values"][0]
            datetimes = coverage["domain"]["axes"]["t"]["values"]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]

            geom_coords = [longitude, latitude]
            if self.z_name:
                geom_coords.append(coverage["domain"]["axes"][self.z_name]["values"][0])

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
                        "geometry": {"type": "Point", "coordinates": geom_coords},
                        "properties": param_vals,
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    # function to convert covjson to xarray dataset
    @nulls_as_nan
    def to_xarray(self):
        if not all("Forecast date" in cov.get("mars:metadata", {}) for cov in self.covjson["coverages"]):
            # Reanalysis (efcl) coverages carry only valid times on ``t``.
            return TimeSeries._to_xarray_no_forecast_date(self)
        if self.z_name:
            dims = ["latitude", "longitude", "levelist", "number", "datetime", "t"]
        else:
            dims = ["latitude", "longitude", "number", "datetime", "t"]
        ds = []

        # One dataset per requested point: coverages are grouped by domain and label,
        # and points that snapped to the same grid point are kept apart.
        def point_key(coverage):
            axes = coverage["domain"]["axes"]
            return (
                axes[self.x_name]["values"][0],
                axes[self.y_name]["values"][0],
                axes[self.z_name]["values"][0] if self.z_name else None,
            )

        def slot_key(coverage):
            return (coverage["mars:metadata"]["number"], coverage["mars:metadata"]["Forecast date"])

        # Within each point, one dataset per distinct time axis (as before), filled from
        # that point's coverages only.
        datasets = []
        for group in self._point_groups(point_key, slot_key):
            seen_t = set()
            for coverage in group:
                t = tuple(coverage["domain"]["axes"]["t"]["values"])
                if t not in seen_t:
                    seen_t.add(t)
                    datasets.append((group, coverage["domain"]))

        num = []
        datetime = []
        for coverage in self.covjson["coverages"]:
            num.append(coverage["mars:metadata"]["number"])
            datetime.append(coverage["mars:metadata"]["Forecast date"])
        nums = list(set(num))
        datetime = list(set(datetime))

        # Process each requested point and time axis
        for group, coords in datasets:
            dataarraydict = {}
            longitude = coords["axes"][self.x_name]["values"]
            latitude = coords["axes"][self.y_name]["values"]
            steps = coords["axes"]["t"]["values"]
            steps = [step.replace("Z", "") for step in steps]
            steps = pd.to_datetime(steps)

            if self.z_name:
                z = coords["axes"][self.z_name]["values"]
                cov_idx_list = self._find_coverages(nums, datetime, longitude, latitude, z, group)
                coord_dict = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "levelist": z,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }
            else:
                cov_idx_list = self._find_coverages(nums, datetime, longitude, latitude, None, group)
                coord_dict = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }

            for parameter in self.parameters:
                param_values = [[[] for _ in range(len(datetime))] for _ in range(len(nums))]

                # Extract parameter values for the current domain
                for i, j, cov in cov_idx_list:
                    param_values[i][j] = cov["ranges"][parameter]["values"]

                long_name = self.get_parameter_metadata(parameter)["observedProperty"]["id"]

                if long_name == "t":
                    long_name = "T"  # Avoid collision with time dimension 't'

                attrs = {
                    "type": self.get_parameter_metadata(parameter)["type"],
                    "units": self.get_parameter_metadata(parameter)["unit"]["symbol"],
                    "long_name": long_name,
                }
                if self.z_name:
                    dataarraydict[long_name] = (dims, [[[param_values]]], attrs)
                else:
                    dataarraydict[long_name] = (dims, [[param_values]], attrs)

            dss = xr.Dataset(data_vars=dataarraydict, coords=coord_dict)
            dss.attrs.update(self._point_dataset_attrs(group))
            ds.append(dss)

        if len(ds) == 1:
            return ds[0]

        return ds

    def _find_coverages(self, nums, datetime, longitude, latitude, z, coverages=None):
        """Find coverages matching domain parameters and return with indices."""
        result = []
        for i, num in enumerate(nums):
            for j, date in enumerate(datetime):
                for coverage in coverages if coverages is not None else self.covjson["coverages"]:
                    if self._covers_domain(coverage, num, date, longitude, latitude, z):
                        result.append((i, j, coverage))
        return result

    def _covers_domain(self, coverage, num, date, longitude, latitude, z):
        """Check if coverage matches the given domain parameters."""
        axes = coverage["domain"]["axes"]
        match = (
            coverage["mars:metadata"]["number"] == num
            and coverage["mars:metadata"]["Forecast date"] == date
            and axes[self.x_name]["values"] == longitude
            and axes[self.y_name]["values"] == latitude
        )
        if match and self.z_name and z is not None:
            match = axes[self.z_name]["values"] == z
        return match
