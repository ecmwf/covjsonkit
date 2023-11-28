from .Decoder import Decoder


class TimeSeries(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)

    def get_domain(self):
        pass

    def get_ranges(self):
        pass
