import xarray as xr

from .decoder import Decoder


class Path(Decoder):
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
        return self.domains[0]["axes"]

    def to_geopandas(self):
        pass

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
                param_vals["datetime"] = coord[0]
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [[coord[1], coord[2], coord[3]]],  # lon, lat, z
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
        x = []
        y = []
        levelist = []
        time = []
        for coord in self.get_coordinates()["composite"]["values"]:
            x.append(float(coord[1]))
            y.append(float(coord[2]))
            levelist.append(float(coord[3]))
            time.append(coord[0])

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
        steps = list(set(steps))

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
                points=(["points"], list(range(0, len(x)))),
                latitude=(["points"], x),
                longitude=(["points"], y),
                levelist=(["points"], levelist),
                time=(["points"], time),
            ),
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        return ds
