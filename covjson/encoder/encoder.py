import os
import json
import xarray as xr
import datetime as dt
import geopandas as gpd
from abc import ABC, abstractmethod
from ..Coverage import Coverage
from ..CoverageCollection import CoverageCollection


class Encoder(ABC):
    def __init__(self, type):
        self.covjson = {}
        self.type = type
        self.covjson["type"] = self.type
        self.covjson["coverages"] = []
        if self.type == "Coverage":
            self.coverage = Coverage(self.covjson)
        elif self.type == "CoverageCollection":
            self.coverage = CoverageCollection(self.covjson)
        else:
            raise TypeError("Type must be Coverage or CoverageCollection")
