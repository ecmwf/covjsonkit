from .decoder import Decoder
import xarray as xr
import datetime as dt


class TimeSeries(Decoder):
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
            values[parameter] = [
                value for sublist in values[parameter] for value in sublist
            ]
        return values

    def get_coordinates(self):
        coordinates = []
        coord_dict = {}
        for param in self.parameters:
            coord_dict[param] = []
        # Get x,y,z,t coords and unpack t coords and match to x,y,z coords
        for domain in self.domains:
            x = domain["axes"]["x"]["values"][0]
            y = domain["axes"]["y"]["values"][0]
            z = domain["axes"]["z"]["values"][0]
            ts = domain["axes"]["t"]["values"]
            for param in self.parameters:
                for t in ts:
                    # Have to replicate these coords for each parameter
                    # coordinates.append([x, y, z, t])
                    coord_dict[param].append([x, y, z, t])
        return coord_dict

    def to_geopandas(self):
        pass

    # function to convert covjson to xarray dataset
    def to_xarray(self):
        dims = ["x", "y", "z", "t"]
        dataarraydict = {}

        # Get coordinates
        for parameter in self.parameters:
            param_values = [[[self.get_values()[parameter]]]]

            coords = self.get_coordinates()[parameter]
            x = [coords[0][0]]
            y = [coords[0][1]]
            z = [coords[0][2]]
            t = [
                dt.datetime.strptime(coord[3], "%Y-%m-%d %H:%M:%S") for coord in coords
            ]

            param_coords = {"x": x, "y": y, "z": z, "t": t}
            dataarray = xr.DataArray(
                param_values,
                dims=dims,
                coords=param_coords,
                name=parameter,
            )

            dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
            dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"][
                "symbol"
            ]
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)[
                "description"
            ]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        ds = xr.Dataset(dataarraydict)
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        return ds
