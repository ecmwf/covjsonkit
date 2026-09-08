import xarray as xr

from ..encoder.encoder import sort_step_values
from .decoder import Decoder


class Path(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        # Backwards-compatible composite ordering for the Trajectory ``composite``
        # tuple. Both the spec-compliant output and the legacy output label the
        # spatial components ``x``/``y``, but with SWAPPED meaning:
        #   * spec:   values ordered [t, x=lon, y=lat(, z)] with split referencing
        #             (a GeographicCRS whose coordinates are exactly ["x", "y"]).
        #   * legacy: values ordered [t, lat, lon(, z)] with a single combined
        #             GeographicCRS whose coordinates are ["t", "x", "y", "z"].
        # Labels alone cannot disambiguate these, so the referencing structure is
        # used: a split GeographicCRS(["x", "y"]) marks spec-compliant output.
        self.t_idx, self.x_idx, self.y_idx, self.z_idx = self._composite_indices()

    def _composite_indices(self):
        labels = self.domains[0]["axes"]["composite"]["coordinates"]
        refs = self.covjson.get("referencing", [])
        is_spec = any(
            ref.get("system", {}).get("type") == "GeographicCRS" and set(ref.get("coordinates", [])) == {"x", "y"}
            for ref in refs
        )

        t_idx = labels.index("t") if "t" in labels else None
        z_idx = None
        for name in ("z", "levelist"):
            if name in labels:
                z_idx = labels.index(name)
                break

        if is_spec:
            # Spec order: label ``x`` holds longitude, label ``y`` holds latitude.
            x_idx = labels.index("x") if "x" in labels else labels.index("longitude")
            y_idx = labels.index("y") if "y" in labels else labels.index("latitude")
        else:
            # Legacy order: values are [t, lat, lon(, z)]; the mislabeled ``x``
            # component actually holds latitude and ``y`` holds longitude.
            lat_label = "x" if "x" in labels else "latitude"
            lon_label = "y" if "y" in labels else "longitude"
            y_idx = labels.index(lat_label)  # latitude
            x_idx = labels.index(lon_label)  # longitude

        return t_idx, x_idx, y_idx, z_idx

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
        return self.domains[0]["axes"]

    def to_geopandas(self):
        pass

    def to_geotiff(self):
        raise TypeError("Path domain cannot be converted to GeoTIFF.")

    def to_geojson(self):
        features = []
        for coverage in self.coverages:
            coords = coverage["domain"]["axes"]["composite"]["values"]
            values = {}
            for key in coverage["ranges"]:
                values[key] = coverage["ranges"][key]["values"]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]

            for idx, coord in enumerate(coords):
                param_vals = {}
                for key in values.keys():
                    param_vals[key] = values[key][idx]
                if "mars:metadata" in coverage:
                    param_vals["mars:metadata"] = mars_metadata
                if self.t_idx is not None:
                    param_vals["datetime"] = coord[self.t_idx]
                geom_coords = [coord[self.x_idx], coord[self.y_idx]]  # lon, lat
                if self.z_idx is not None:
                    geom_coords.append(coord[self.z_idx])
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": geom_coords,
                        },
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
        time = []
        for coord in self.get_coordinates()["composite"]["values"]:
            longitude.append(float(coord[self.x_idx]))
            latitude.append(float(coord[self.y_idx]))
            if self.z_idx is not None:
                levelist.append(float(coord[self.z_idx]))
            if self.t_idx is not None:
                time.append(coord[self.t_idx])

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
            datetimes.append(coverage["mars:metadata"]["Forecast date"])
            for parameter in self.parameters:
                # values[parameter].append(coverage["ranges"][parameter]["values"])
                if coverage["mars:metadata"]["Forecast date"] not in values[parameter]:
                    values[parameter][coverage["mars:metadata"]["Forecast date"]] = {}
                if (
                    coverage["mars:metadata"]["number"]
                    not in values[parameter][coverage["mars:metadata"]["Forecast date"]]
                ):
                    values[parameter][coverage["mars:metadata"]["Forecast date"]][
                        coverage["mars:metadata"]["number"]
                    ] = {}
                values[parameter][coverage["mars:metadata"]["Forecast date"]][coverage["mars:metadata"]["number"]][
                    coverage["mars:metadata"]["step"]
                ] = coverage["ranges"][parameter]["values"]

        datetimes = list(set(datetimes))
        numbers = list(set(numbers))
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

        ds = xr.Dataset(
            dataarraydict,
            coords=dict(
                datetimes=(["datetimes"], datetimes),
                number=(["number"], numbers),
                steps=(["steps"], steps),
                points=(["points"], list(range(0, len(longitude)))),
                latitude=(["points"], latitude),
                longitude=(["points"], longitude),
                **({"levelist": (["points"], levelist)} if self.z_idx is not None else {}),
                time=(["points"], time),
            ),
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        return ds
