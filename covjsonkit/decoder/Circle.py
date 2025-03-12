import xarray as xr

from .decoder import Decoder


class Circle(Decoder):
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

    def to_xarray(self):
        dims = ["datetimes", "number", "steps", "points"]
        dataarraydict = {}

        # Get coordinates
        x = []
        y = []
        z = []
        datetimes = []
        for coord in self.get_coordinates()["composite"]["values"]:
            x.append(float(coord[0]))
            y.append(float(coord[1]))
            z.append(float(coord[2]))
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
        steps = sorted(list(set(steps)))

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
                x=(["points"], x),
                y=(["points"], y),
                z=(["points"], z),
            ),
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        # Add date attribute
        ds.attrs["date"] = self.get_coordinates()["t"]["values"][0]

        return ds
