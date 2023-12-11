import os
import json
import xarray as xr
import datetime as dt
import geopandas as gpd
from abc import ABC, abstractmethod
from eccovjson.Coverage import Coverage
from eccovjson.CoverageCollection import CoverageCollection


class Encoder(ABC):
    def __init__(self, type, domaintype):
        self.covjson = {}

        self.type = type
        self.parameters = []
        self.covjson["type"] = self.type
        self.covjson["domainType"] = domaintype
        self.covjson["coverages"] = []
        self.covjson["parameters"] = {}
        self.covjson["referencing"] = []

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
        self.parameters.append(parameter)

    def add_reference(self, reference):
        self.covjson["referencing"].append(reference)

    @abstractmethod
    def add_coverage(self, mars_metadata, coords, values):
        pass

    @abstractmethod
    def add_domain(self, coverage, domain):
        pass

    @abstractmethod
    def add_range(self, coverage, range):
        pass

    @abstractmethod
    def add_mars_metadata(self, coverage, metadata):
        pass

    @abstractmethod
    def from_xarray(self, dataset):
        pass

    @abstractmethod
    def from_polytope(self, result):
        pass
