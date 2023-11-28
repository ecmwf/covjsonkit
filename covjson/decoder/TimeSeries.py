from .Decoder import Decoder


class TimeSeries(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)

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
        pass
