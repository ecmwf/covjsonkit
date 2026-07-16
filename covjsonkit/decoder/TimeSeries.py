import pandas as pd
import xarray as xr

from .decoder import Decoder


class TimeSeries(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        first_axes = self.covjson["coverages"][0]["domain"]["axes"]
        self.x_name = "x"
        self.y_name = "y"
        self.z_name = "z" if "z" in first_axes else None

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
                    # coordinates.append([x, y, z, t])
                    coords.append([latitude, longitude, level, fct, t, num])
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
    def to_xarray(self):
        # Monthly-means fast path: coverages produced by from_polytope_month() pack all
        # time steps into a single coverage's t-axis and do NOT write "Forecast date"
        # into mars:metadata.  Detect this case and use a simpler (time, point) layout
        # that mirrors the Wkt/Frame/Shapefile decoders.
        has_forecast_date = any("Forecast date" in cov.get("mars:metadata", {}) for cov in self.covjson["coverages"])
        if not has_forecast_date:
            return self._to_xarray_no_forecast_date()

        if self.z_name:
            dims = ["latitude", "longitude", "levelist", "number", "datetime", "t"]
        else:
            dims = ["latitude", "longitude", "number", "datetime", "t"]
        ds = []

        all_coords = self.get_domains()

        unique_coords = set()
        unique_domains = []

        for domain in self.domains:
            longitude = domain["axes"][self.x_name]["values"][0]
            latitude = domain["axes"][self.y_name]["values"][0]
            t = tuple(domain["axes"]["t"]["values"])

            if self.z_name:
                z = domain["axes"][self.z_name]["values"][0]
                coord_tuple = (longitude, latitude, z, t)
            else:
                coord_tuple = (longitude, latitude, t)

            if coord_tuple not in unique_coords:
                unique_coords.add(coord_tuple)
                unique_domains.append(domain)

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
            longitude = coords["axes"][self.x_name]["values"]
            latitude = coords["axes"][self.y_name]["values"]
            steps = coords["axes"]["t"]["values"]
            steps = [step.replace("Z", "") for step in steps]
            steps = pd.to_datetime(steps)

            if self.z_name:
                z = coords["axes"][self.z_name]["values"]
                cov_idx_list = self._find_coverages(nums, datetime, longitude, latitude, z)
                coord_dict = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "levelist": z,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }
            else:
                cov_idx_list = self._find_coverages(nums, datetime, longitude, latitude, None)
                coord_dict = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }

            for parameter in self.parameters:
                param_values = [[[] for _ in range(len(datetime))] for _ in range(len(nums))]

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

            ds.append(xr.Dataset(data_vars=dataarraydict, coords=coord_dict))

        for mars_metadata in self.mars_metadata[0]:
            if mars_metadata != "date" and mars_metadata != "step":
                for dss in ds:
                    dss.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        if len(ds) == 1:
            return ds[0]

        return ds

    def _find_coverages(self, nums, datetime, longitude, latitude, z):
        result = []
        for i, num in enumerate(nums):
            for j, date in enumerate(datetime):
                for coverage in self.covjson["coverages"]:
                    if self._covers_domain(coverage, num, date, longitude, latitude, z):
                        result.append((i, j, coverage))
        return result

    def _covers_domain(self, coverage, num, date, longitude, latitude, z):
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

    def _to_xarray_no_forecast_date(self):
        """Monthly-means path: all time steps are packed into a single coverage's
        t-axis and there is no "Forecast date" in mars:metadata.
        """
        ds_list = []

        for coverage in self.covjson["coverages"]:
            domain = coverage["domain"]["axes"]
            longitude = domain[self.x_name]["values"]
            latitude = domain[self.y_name]["values"]

            steps = domain["t"]["values"]
            steps = [s.replace("Z", "") for s in steps]
            steps = pd.to_datetime(steps)

            dataarraydict = {}
            for parameter in self.parameters:
                values = coverage["ranges"][parameter]["values"]
                long_name = self.get_parameter_metadata(parameter)["observedProperty"]["id"]

                if long_name == "t":
                    long_name = "T"  # Avoid collision with time dimension 't'

                attrs = {
                    "type": self.get_parameter_metadata(parameter)["type"],
                    "units": self.get_parameter_metadata(parameter)["unit"]["symbol"],
                    "long_name": long_name,
                }

                dataarray = xr.DataArray(values, dims=["t"], coords={"t": steps}, attrs=attrs)

                dataarraydict[long_name] = dataarray

            coord_dict = dict(
                latitude=(["latitude"], latitude),
                longitude=(["longitude"], longitude),
            )
            if self.z_name:
                z = domain[self.z_name]["values"]
                coord_dict["levelist"] = (["levelist"], z)

            dss = xr.Dataset(dataarraydict, coords=coord_dict)

            # Attach MARS metadata (skip keys that vary per time step)
            mm = coverage.get("mars:metadata", {})
            for key, val in mm.items():
                dss.attrs[key] = val

            ds_list.append(dss)

        if len(ds_list) == 1:
            return ds_list[0]
        return ds_list
