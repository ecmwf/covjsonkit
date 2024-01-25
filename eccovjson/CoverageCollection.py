class CoverageCollection:
    def __init__(self, covjson):
        if isinstance(covjson, dict):
            print("Received Coverage")
            self.coverage = covjson

        self.type = self.coverage["type"]

        if self.type == "CoverageCollection":
            print("Correct Type")
        elif self.type == "Coverage":
            raise TypeError("CoverageCollection class takes CoverageCollection not Coverage")

        self.coverages = self.coverage["coverages"]
