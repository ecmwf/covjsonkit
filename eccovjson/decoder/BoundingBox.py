import datetime as dt

import xarray as xr

from .decoder import Decoder


class BoundingBox(Decoder):
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
        """
        coord_dict = {}
        for param in self.parameters:
            coord_dict[param] = []
        # Get x,y,z,t coords and unpack t coords and match to x,y,z coords
        for ind, domain in enumerate(self.domains):
            t = domain["axes"]["t"]["values"][0]
            # num = self.mars_metadata[ind]["number"]
            fct = self.mars_metadata[ind]["date"]

            for param in self.parameters:
                coords = []
                for coord in domain["axes"]["composite"]["values"]:
                    x = coord[0]
                    y = coord[1]
                    z = coord[2]
                    coords.append([x, y, z, fct, t])
                coord_dict[param].append(coords)
        return coord_dict
        """
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
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["description"]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        ds = xr.Dataset(
            dataarraydict,
            coords=dict(points=(["points"], list(range(0, len(x)))), x=(["points"], x), y=(["points"], y)),
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        return ds
