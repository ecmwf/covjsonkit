import xarray as xr

from .decoder import Decoder


class Frame(Decoder):
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

    def to_xarray(self):
        dims = ["points"]
        dataarraydict = {}

        # Get coordinates
        x = []
        y = []
        for coord in self.get_coordinates()["composite"]["values"]:
            x.append(float(coord[0]))
            y.append(float(coord[1]))

        # Get values
        for parameter in self.parameters:
            dataarray = xr.DataArray(self.get_values()[parameter][0], dims=dims)
            dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
            dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        ds = xr.Dataset(
            dataarraydict,
            coords=dict(points=(["points"], list(range(0, len(x)))), x=(["points"], x), y=(["points"], y)),
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        # Add date attribute
        ds.attrs["date"] = self.get_coordinates()["t"]["values"]

        return ds
