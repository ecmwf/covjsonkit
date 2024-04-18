from abc import ABC, abstractmethod

from covjson_pydantic.coverage import CoverageCollection
from covjson_pydantic.domain import DomainType

# from eccovjson.CoverageCollection import CoverageCollection
from eccovjson.param_db import get_param_from_db, get_unit_from_db


class Encoder(ABC):
    def __init__(self, type, domaintype):
        self.covjson = {}

        self.type = type

        self.referencing = []

        domaintype = domaintype.lower()

        if domaintype == "pointseries":
            self.domaintype = DomainType.point_series
        elif domaintype == "multipoint":
            self.domaintype = DomainType.multi_point
        elif domaintype == "wkt":
            self.domaintype = DomainType.multi_point
        elif domaintype == "boundingbox":
            self.domaintype = DomainType.multi_point
        elif domaintype == "shapefile":
            self.domaintype = DomainType.multi_point
        elif domaintype == "frame":
            self.domaintype = DomainType.multi_point
        elif domaintype == "path":
            self.domaintype = DomainType.trajectory

        self.pydantic_coverage = CoverageCollection(
            type=type, coverages=[], domainType=self.domaintype, parameters={}, referencing=[]
        )
        # self.covjson = self.pydantic_coverage.model_dump_json(exclude_none=True)
        self.parameters = []
        # self.covjson["type"] = self.type
        # self.covjson["domainType"] = domaintype
        # self.covjson["coverages"] = []
        # self.covjson["parameters"] = {}
        # self.covjson["referencing"] = []

        # if type == "Coverage":
        #    self.coverage = Coverage(self.covjson)
        # elif type == "CoverageCollection":
        #    self.coverage = CoverageCollection(self.covjson)
        # else:
        #    raise TypeError("Type must be Coverage or CoverageCollection")

    def add_parameter(self, param):
        param_dict = get_param_from_db(param)
        unit = get_unit_from_db(param_dict["unit_id"])
        self.pydantic_coverage.parameters[param_dict["shortname"]] = {
            "type": "Parameter",
            "description": {"en": param_dict["description"]},
            "unit": {"symbol": unit["name"]},
            "observedProperty": {
                "id": param_dict["shortname"],
                "label": {"en": param_dict["name"]},
            },
        }
        self.parameters.append(param)

    def add_reference(self, reference):
        self.pydantic_coverage.referencing.append(reference)
        for ref in reference["coordinates"]:
            if ref not in self.referencing:
                self.referencing.append(ref)

    def convert_param_id_to_param(self, paramid):
        try:
            param = int(paramid)
        except BaseException:
            return paramid
        param_dict = get_param_from_db(int(param))
        return param_dict["shortname"]

    def get_json(self):
        self.covjson = self.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        return self.covjson

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
