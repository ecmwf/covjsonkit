class Coverage:
    def __init__(self, covjson):
        if isinstance(covjson, dict):
            print("Received Coverage")
            self.coverage = covjson

        self.type = self.coverage.pop("type")

        if self.type == "Coverage":
            print("Correct Type")
        elif self.type == "CoverageCollection":
            raise TypeError("Coverage class takes coverage not CoverageCollection")
