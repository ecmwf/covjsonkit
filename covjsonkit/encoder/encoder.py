from abc import ABC, abstractmethod

import orjson
from covjson_pydantic.coverage import CoverageCollection
from covjson_pydantic.domain import DomainType

from covjsonkit.param_db import get_param_ids, get_params, get_units


class Encoder(ABC):
    def __init__(self, type, domaintype):
        self.covjson = {}
        self.covjson["type"] = "CoverageCollection"

        self.type = type

        self.referencing = []

        self.units = get_units(self.type)
        self.params = get_params(self.type)
        self.param_ids = get_param_ids(self.type)

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
            self.domaintype = "Trajectory"

        # Trajectory not yet implemented in covjson-pydantic
        if self.domaintype != "Trajectory":
            self.pydantic_coverage = CoverageCollection(
                type="CoverageCollection", coverages=[], domainType=self.domaintype, parameters={}, referencing=[]
            )
        self.parameters = []

    def add_parameter(self, param):
        # param_dict = get_param_from_db(param)
        # unit = get_unit_from_db(param_dict["unit_id"])
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
        # self.pydantic_coverage.parameters[param_dict["shortname"]] = Parameter.model_validate_json(
        #    json.dumps(parameter)
        # )
        if "parameters" not in self.covjson:
            self.covjson["parameters"] = {}
            self.covjson["parameters"][param_dict["shortname"]] = parameter
        else:
            self.covjson["parameters"][param_dict["shortname"]] = parameter
        self.parameters.append(param)

    def add_reference(self, reference):
        # self.pydantic_coverage.referencing.append(
        #    ReferenceSystemConnectionObject.model_validate_json(json.dumps(reference))
        # )
        # self.pydantic_coverage.referencing.append(reference)
        # for ref in reference["coordinates"]:
        #    if ref not in self.referencing:
        # self.referencing.append(ref)
        self.covjson["referencing"] = [reference]

    def convert_param_id_to_param(self, paramid):
        try:
            param = int(paramid)
        except BaseException:
            return paramid
        # param_dict = get_param_from_db(int(param))
        param_dict = self.params[str(param)]
        return param_dict["shortname"]

    def get_json(self):
        # self.covjson = self.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        return orjson.dumps(self.covjson)

    def walk_tree(self, tree, lat, coords, mars_metadata, param, range_dict, number, step, dates, levels):
        if len(tree.children) != 0:
            # recurse while we are not a leaf
            for c in tree.children:
                if (
                    c.axis.name != "latitude"
                    and c.axis.name != "longitude"
                    and c.axis.name != "param"
                    and c.axis.name != "date"
                    and c.axis.name != "levelist"
                ):
                    mars_metadata[c.axis.name] = c.values[0]
                if c.axis.name == "latitude":
                    lat = c.values[0]
                if c.axis.name == "levelist":
                    levels = c.values
                    for date in range_dict.keys():
                        for level in levels:
                            if level not in range_dict[date]:
                                range_dict[date][level] = {}
                if c.axis.name == "param":
                    param = c.values
                    for date in range_dict.keys():
                        if range_dict[date] == {}:
                            range_dict[date] = {0: {}}
                        for level in levels:
                            if range_dict[date][level] == {}:
                                range_dict[date][level] = {0: {}}
                            for num in number:
                                for para in param:
                                    if para not in range_dict[date][level][num]:
                                        range_dict[date][level][num][para] = {}
                                        self.add_parameter(para)
                if c.axis.name == "date" or c.axis.name == "time":
                    dates = [str(date) + "Z" for date in c.values]
                    mars_metadata["Forecast date"] = str(c.values[0])
                    for date in dates:
                        coords[date] = {}
                        coords[date]["composite"] = []
                        coords[date]["t"] = [date]
                        if date not in range_dict:
                            range_dict[date] = {}
                if c.axis.name == "number":
                    number = c.values
                    for date in dates:
                        for level in levels:
                            if level not in range_dict[date]:
                                range_dict[date][level] = {}
                            for num in number:
                                range_dict[date][level][num] = {}
                if c.axis.name == "step":
                    step = c.values
                    for date in dates:
                        for level in levels:
                            for num in number:
                                for para in param:
                                    for s in step:
                                        range_dict[date][level][num][para][s] = []

                self.walk_tree(c, lat, coords, mars_metadata, param, range_dict, number, step, dates, levels)
        else:
            tree.values = [float(val) for val in tree.values]
            if all(val is None for val in tree.result):
                range_dict.pop(dates[0], None)
            else:
                tree.result = [float(val) if val is not None else val for val in tree.result]
                level_len = len(tree.result) / len(levels)
                num_len = level_len / len(number)
                para_len = num_len / len(param)
                step_len = para_len / len(step)

                for date in dates:
                    for val in tree.values:
                        coords[date]["composite"].append([lat, val])

                for l, level in enumerate(levels):  # noqa: E741
                    for i, num in enumerate(number):
                        for j, para in enumerate(param):
                            for k, s in enumerate(step):
                                range_dict[dates[0]][level][num][para][s].extend(
                                    tree.result[
                                        int(l * level_len)
                                        + int(i * num_len)
                                        + int(j * para_len)
                                        + int(k * step_len) : int(l * level_len)
                                        + int(i * num_len)
                                        + int(j * para_len)
                                        + int((k + 1) * step_len)
                                    ]
                                )

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
