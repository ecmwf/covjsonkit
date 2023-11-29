from .Decoder import Decoder


class VerticalProfile(Decoder):
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

    def get_coordinates(self):
        coordinates = []
        # Get x,y,z coords and unpack z coords and match to x,y coords
        for domain in self.domains:
            x = domain["axes"]["x"]["values"][0]
            y = domain["axes"]["y"]["values"][0]
            t = domain["axes"]["t"]["values"][0]
            zs = domain["axes"]["z"]["values"]
            for z in zs:
                # Have to replicate these coords for each parameter
                for _ in self.parameters:
                    coordinates.append([x, y, z, t])
        return coordinates

    def get_values(self):
        values = {}
        for parameter in self.parameters:
            values[parameter] = []
            for range in self.ranges:
                values[parameter].append(range[parameter]["values"])
        return values

    def to_geopandas(self):
        pass

    def to_xarray(self):
        pass
