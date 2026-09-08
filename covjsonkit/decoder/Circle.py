import xarray as xr

from ..encoder.encoder import sort_step_values
from .decoder import Decoder


class Circle(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        # Backwards-compatible composite ordering: derive the position of the
        # longitude (x), latitude (y) and optional vertical (z) components from
        # the ``composite`` axis's ``coordinates`` labels. Reads both spec
        # coverages (x/y[/z], values [lon, lat[, level]]) and legacy coverages
        # (latitude/longitude/levelist, values [lat, lon[, level]]).
        labels = self.domains[0]["axes"]["composite"]["coordinates"]
        self.x_idx = self._label_index(labels, ("x", "longitude"))
        self.y_idx = self._label_index(labels, ("y", "latitude"))
        self.z_idx = self._label_index(labels, ("z", "levelist"), required=False)

    @staticmethod
    def _label_index(labels, names, required=True):
        for name in names:
            if name in labels:
                return labels.index(name)
        if required:
            raise ValueError(f"None of {names} found in composite coordinates {labels}")
        return None

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
        raise TypeError("Circle domain cannot be converted to GeoTIFF.")

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
                geom_coords = [lonlat[self.x_idx], lonlat[self.y_idx]]
                if self.z_idx is not None:
                    geom_coords.append(lonlat[self.z_idx])
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": geom_coords},
                        "properties": param_vals,
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    def to_xarray(self):
        dims = ["datetimes", "number", "steps", "points"]
        dataarraydict = {}

        # Get coordinates
        longitude = []
        latitude = []
        levelist = []
        datetimes = []
        for coord in self.get_coordinates()["composite"]["values"]:
            longitude.append(float(coord[self.x_idx]))
            latitude.append(float(coord[self.y_idx]))
            if self.z_idx is not None:
                levelist.append(float(coord[self.z_idx]))
        for datetime in self.get_coordinates()["t"]["values"]:
            datetimes.append(datetime)

        values = {}
        for parameter in self.parameters:
            values[parameter] = {}

        datetimes = []
        numbers = []
        steps = []
        for coverage in self.coverages:
            if "number" not in coverage["mars:metadata"]:
                coverage["mars:metadata"]["number"] = 0
            numbers.append(coverage["mars:metadata"]["number"])
            if "step" not in coverage["mars:metadata"]:
                coverage["mars:metadata"]["step"] = 0
            steps.append(coverage["mars:metadata"]["step"])
            datetimes.append(coverage["domain"]["axes"]["t"]["values"][0])
            for parameter in self.parameters:
                # values[parameter].append(coverage["ranges"][parameter]["values"])
                if coverage["domain"]["axes"]["t"]["values"][0] not in values[parameter]:
                    values[parameter][coverage["domain"]["axes"]["t"]["values"][0]] = {}
                if (
                    coverage["mars:metadata"]["number"]
                    not in values[parameter][coverage["domain"]["axes"]["t"]["values"][0]]
                ):
                    values[parameter][coverage["domain"]["axes"]["t"]["values"][0]][
                        coverage["mars:metadata"]["number"]
                    ] = {}
                values[parameter][coverage["domain"]["axes"]["t"]["values"][0]][coverage["mars:metadata"]["number"]][
                    coverage["mars:metadata"]["step"]
                ] = coverage["ranges"][parameter]["values"]

        datetimes = sorted(list(set(datetimes)))
        numbers = sorted(list(set(numbers)))
        steps = sort_step_values(list(set(steps)))

        new_values = {}
        for parameter in values.keys():
            new_values[parameter] = []
            for i, datetime in enumerate(datetimes):
                new_values[parameter].append([])
                for j, number in enumerate(numbers):
                    new_values[parameter][i].append([])
                    for k, step in enumerate(steps):
                        new_values[parameter][i][j].append(values[parameter][datetime][number][step])

        for parameter in self.parameters:
            dataarray = xr.DataArray(new_values[parameter], dims=dims)
            dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
            dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        coords_dict = dict(
            datetimes=(["datetimes"], datetimes),
            number=(["number"], numbers),
            steps=(["steps"], steps),
            points=(["points"], list(range(0, len(longitude)))),
            latitude=(["points"], latitude),
            longitude=(["points"], longitude),
        )
        if self.z_idx is not None:
            coords_dict["levelist"] = (["points"], levelist)

        ds = xr.Dataset(
            dataarraydict,
            coords=coords_dict,
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        # Add date attribute
        ds.attrs["date"] = self.get_coordinates()["t"]["values"][0]

        return ds
