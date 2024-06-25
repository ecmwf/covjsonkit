import json
from abc import ABC, abstractmethod

from covjson_pydantic.coverage import CoverageCollection
from covjson_pydantic.domain import DomainType
from covjson_pydantic.parameter import Parameter
from covjson_pydantic.reference_system import ReferenceSystemConnectionObject

from covjsonkit.param_db import get_param_from_db, get_unit_from_db, get_param_ids, get_params, get_units


class Encoder(ABC):
    def __init__(self, type, domaintype):
        self.covjson = {}
        self.covjson["type"] = "CoverageCollection"

        self.type = type

        self.referencing = []

        self.units = get_units()
        self.params = get_params()
        self.param_ids = get_param_ids()

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
        self.parameters = []

    def add_parameter(self, param):
        #param_dict = get_param_from_db(param)
        #unit = get_unit_from_db(param_dict["unit_id"])
        param_dict = self.params[str(param)]
        unit = self.units[str(param_dict["unit_id"])]
        parameter = {
            "type": "Parameter",
            "description": {"en": param_dict["description"]},
            "unit": {"symbol": unit["name"]},
            "observedProperty": {
                "id": param_dict["shortname"],
                "label": {"en": param_dict["name"]},
            },
        }
        self.pydantic_coverage.parameters[param_dict["shortname"]] = Parameter.model_validate_json(
            json.dumps(parameter)
        )
        self.parameters.append(param)

    def add_reference(self, reference):
        #self.pydantic_coverage.referencing.append(
        #    ReferenceSystemConnectionObject.model_validate_json(json.dumps(reference))
        #)
        #self.pydantic_coverage.referencing.append(reference)
        #for ref in reference["coordinates"]:
        #    if ref not in self.referencing:
                #self.referencing.append(ref)
        self.covjson['referencing'] = [reference]

    def convert_param_id_to_param(self, paramid):
        try:
            param = int(paramid)
        except BaseException:
            return paramid
        #param_dict = get_param_from_db(int(param))
        param_dict = self.params[str(param)]
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
