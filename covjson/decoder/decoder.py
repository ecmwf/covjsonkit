import os
import json
from ..Coverage import Coverage
from ..CoverageCollection import CoverageCollection


class Decoder:
    def __init__(self, covjson):
        # if python dictionary no need for loading, otherwise load json file
        if isinstance(covjson, dict):
            self.covjson = covjson
            print("Dictionary")
        elif isinstance(covjson, str):
            with open(covjson) as json_file:
                self.covjson = json.load(json_file)
            print("Not dictionary")
        else:
            raise TypeError("Covjson must be dictionary or covjson file")

        self.type = self.covjson["type"]

        if self.type == "Coverage":
            self.coverage = Coverage(self.covjson)
        elif self.type == "CoverageCollection":
            self.coverage = CoverageCollection(self.covjson)
        else:
            raise TypeError("Type must be Coverage or CoverageCollection")

    def get_type(self):
        return self.type
