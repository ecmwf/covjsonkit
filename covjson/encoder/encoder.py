import os
import json
import xarray as xr
import datetime as dt
import geopandas as gpd
from abc import ABC, abstractmethod
from ..Coverage import Coverage
from ..CoverageCollection import CoverageCollection


class Encoder(ABC):
    def __init__(self, type, domaintype):
        self.covjson = {}

        self.type = type
        self.covjson["type"] = self.type
        self.covjson["domainType"] = domaintype
        self.covjson["coverages"] = []
        self.covjson["referencing"] = []
        self.covjson["parameters"] = {}

        if type == "Coverage":
            self.coverage = Coverage(self.covjson)
        elif type == "CoverageCollection":
            self.coverage = CoverageCollection(self.covjson)
        else:
            raise TypeError("Type must be Coverage or CoverageCollection")

        if domaintype != "PointSeries" and domaintype != "VerticalProfile":
            raise TypeError("DomainType must be PointSeries or VerticalProfile")

    def add_parameter(self, parameter, metadata):
        self.covjson["parameters"][parameter] = metadata

    def add_reference(self, reference):
        self.covjson["referencing"].append(reference)

    def add_coverage(self, coverage):
        pass
