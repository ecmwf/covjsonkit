from .Decoder import Decoder


class VerticalProfile(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)

    def get_domain(self):
        pass

    def get_ranges(self):
        pass
