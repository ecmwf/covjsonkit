from abc import ABC, abstractmethod

from eccovjson.Coverage import Coverage
from eccovjson.CoverageCollection import CoverageCollection
from eccovjson.param_db import get_param_from_db, get_unit_from_db


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

    def add_parameter(self, param):
        param_dict = get_param_from_db(param)
        unit = get_unit_from_db(param_dict["unit_id"])
        self.covjson["parameters"][param_dict["shortname"]] = {
            "type": "Parameter",
            "description": param_dict["description"],
            "unit": {"symbol": unit["name"]},
            "observedProperty": {
                "id": param_dict["shortname"],
                "label": {"en": param_dict["name"]},
            },
        }
        self.parameters.append(param)

    def add_reference(self, reference):
        self.covjson["referencing"].append(reference)

    def convert_param_id_to_param(self, paramid):
        try:
            param = int(paramid)
        except BaseException:
            return paramid
        param_dict = get_param_from_db(int(param))
        return param_dict["shortname"]

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
