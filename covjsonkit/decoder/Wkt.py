import xarray as xr

from .decoder import Decoder


class Wkt(Decoder):
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
            # values[parameter] = [
            #    value for sublist in values[parameter] for value in sublist
            # ]
        return values

    def get_coordinates(self):
        return self.domains[0]["axes"]

    def to_geopandas(self):
        pass

    def to_geotiff(self):
        pass

    def to_geojson(self):
        features = []
        for coverage in self.covjson["coverages"]:
            coords = coverage["domain"]["axes"]["composite"]["values"]
            datetime = coverage["domain"]["axes"]["t"]["values"][0]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]

            values = {}
            for key in coverage["ranges"]:
                values[key] = coverage["ranges"][key]["values"]

            for idx, lonlat in enumerate(coords):
                param_vals = {}
                for key in values.keys():
                    param_vals[key] = values[key][idx]
                param_vals["datetime"] = datetime
                if "mars:metadata" in coverage:
                    param_vals["mars:metadata"] = mars_metadata
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lonlat[1], lonlat[0], lonlat[2]]},
                        "properties": param_vals,
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    def to_xarray(self):
        """Convert a MultiPoint CoverageCollection to an xarray Dataset.

        Each coverage corresponds to one time step. The resulting Dataset has
        dimensions (time, points) so that all coverages are represented, not
        only the first one.
        """
        all_values = self.get_values()

        # Collect per-coverage spatial coordinates and timestamps.
        # All coverages share the same spatial points, but each has its own
        # timestamp and set of values.
        times = []
        x = []
        y = []

        for i, domain in enumerate(self.domains):
            times.append(domain["axes"]["t"]["values"][0])
            if i == 0:
                for coord in domain["axes"]["composite"]["values"]:
                    x.append(float(coord[0]))
                    y.append(float(coord[1]))

        n_times = len(times)
        n_points = len(x)

        dataarraydict = {}
        for parameter in self.parameters:
            # Shape: (time, points)
            data = all_values[parameter]  # list of n_times lists, each of length n_points
            dataarray = xr.DataArray(data, dims=["time", "points"])
            dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
            dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        ds = xr.Dataset(
            dataarraydict,
            coords=dict(
                time=(["time"], times),
                points=(["points"], list(range(n_points))),
                x=(["points"], x),
                y=(["points"], y),
            ),
        )

        # Attach MARS metadata from the first coverage as dataset attributes
        for key, val in self.mars_metadata[0].items():
            ds.attrs[key] = val

        return ds
