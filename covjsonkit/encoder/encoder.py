from abc import ABC, abstractmethod

import orjson
import pandas as pd
from covjson_pydantic.coverage import CoverageCollection
from covjson_pydantic.domain import DomainType

from covjsonkit.param_db import get_param_ids, get_params, get_units


class Encoder(ABC):
    def __init__(self, type, domaintype):
        """
        Base class for encoding data into CovJSON format.

        The Encoder class provides functionality to initialize and manage the encoding
        process for various domain types, including point series, multi-point, vertical profiles,
        and trajectories. It handles parameters, referencing systems, and domain types, and
        supports conversion between parameter IDs and parameter names.

        Attributes:
            covjson (dict): The CovJSON representation being constructed.
            type (str): The type of data being encoded.
            referencing (list): A list of referencing systems used in the encoding.
            units (dict): Units associated with the parameters, retrieved from the database.
            params (dict): Parameters associated with the data type, retrieved from the database.
            param_ids (dict): Mapping of parameter names to their IDs.
            domaintype (str): The domain type of the data being encoded.
            pydantic_coverage (CoverageCollection): A Pydantic representation of the coverage collection.
            parameters (list): A list of parameters included in the encoding.

        Methods:
            add_parameter(param): Adds a parameter to the CovJSON representation.
            add_reference(reference): Adds a referencing system to the CovJSON representation.
            convert_param_id_to_param(paramid): Converts a parameter ID to its corresponding parameter name.
            convert_param_to_param_id(param): Converts a parameter name to its corresponding parameter ID.
            get_json(): Returns the CovJSON representation as a JSON string.
            walk_tree(tree, fields, coords, mars_metadata, range_dict): Processes a hierarchical tree structure
                to extract data and populate the CovJSON representation.
            walk_tree_step(tree, fields, coords, mars_metadata, range_dict): Processes a hierarchical tree structure
                with step-based data to extract and populate the CovJSON representation.

        Abstract Methods:
            add_coverage(mars_metadata, coords, values): Abstract method for adding coverage data.
            add_domain(coverage, domain): Abstract method for adding domain information.
            add_range(coverage, range): Abstract method for adding range information.
            add_mars_metadata(coverage, metadata): Abstract method for adding Mars metadata.
            from_xarray(dataset): Abstract method for encoding data from an xarray dataset.
            from_polytope(result): Abstract method for encoding data from a polytope result.
        """

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
        elif domaintype == "polygon":
            self.domaintype = DomainType.multi_point
        elif domaintype == "boundingbox":
            self.domaintype = DomainType.multi_point
        elif domaintype == "shapefile":
            self.domaintype = DomainType.multi_point
        elif domaintype == "frame":
            self.domaintype = DomainType.multi_point
        elif domaintype == "circle":
            self.domaintype = DomainType.multi_point
        elif domaintype == "verticalprofile":
            self.domaintype = DomainType.vertical_profile
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

    def convert_param_to_param_id(self, param):
        if isinstance(param, int):
            return param
        # param_dict = get_param_from_db(param)
        param_id = self.param_ids[param]
        return param_id

    def get_json(self):
        # self.covjson = self.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        return orjson.dumps(self.covjson)

    def walk_tree(self, tree, fields, coords, mars_metadata, range_dict):
        def create_composite_key(date, level, num, para, s):
            return (date, level, num, para, s)

        def handle_non_leaf_node(child):
            non_leaf_axes = ["latitude", "longitude", "param", "date"]
            if child.axis.name not in non_leaf_axes:
                mars_metadata[child.axis.name] = child.values[0]

        def handle_specific_axes(child):
            if child.axis.name == "latitude":
                return child.values[0]
            if child.axis.name == "levelist":
                return child.values
            if child.axis.name == "param":
                return child.values
            if child.axis.name in ["date", "time"]:
                dates = [f"{date}Z" for date in child.values]
                mars_metadata["Forecast date"] = str(child.values[0])
                for date in dates:
                    coords[date] = {}
                    coords[date]["composite"] = []
                    coords[date]["t"] = [date]
                return dates
            if child.axis.name == "number":
                return child.values
            if child.axis.name == "step":
                return child.values
            return None

        def calculate_index_bounds(level_len, num_len, para_len, step_len, l, i, j, k):  # noqa: E741
            start_index = int(l * level_len) + int(i * num_len) + int(j * para_len) + int(k * step_len)
            end_index = start_index + int(step_len)
            return start_index, end_index

        def append_composite_coords(dates, tree_values, lat, coords):
            # for date in dates:
            for value in tree_values:
                coords[dates]["composite"].append([lat, value])

        if len(tree.children) != 0:
            for child in tree.children:
                handle_non_leaf_node(child)
                result = handle_specific_axes(child)
                if result is not None:
                    if child.axis.name == "latitude":
                        fields["lat"] = result
                    elif child.axis.name == "levelist":
                        fields["levels"] = result
                        if "l" in fields:
                            fields["l"].extend(result)
                    elif child.axis.name == "param":
                        fields["param"] = result
                    elif child.axis.name in ["date", "time"]:
                        fields["dates"].extend(result)
                    elif child.axis.name == "number":
                        fields["number"] = result
                    elif child.axis.name == "step":
                        fields["step"] = result
                        if "s" in fields:
                            fields["s"].extend(result)

                self.walk_tree(child, fields, coords, mars_metadata, range_dict)
        else:
            tree.values = [float(val) for val in tree.values]
            if all(val is None for val in tree.result):
                fields["dates"] = fields["dates"][:-1]
                for date in fields["dates"]:
                    for level in fields["levels"]:
                        for num in fields["number"]:
                            for para in fields["param"]:
                                for s in fields["step"]:
                                    key = create_composite_key(date, level, num, para, s)
                                    if key in range_dict:
                                        del range_dict[key]
            else:
                tree.result = [float(val) if val is not None else val for val in tree.result]
                level_len = len(tree.result) / len(fields["levels"])
                num_len = level_len / len(fields["number"])
                para_len = num_len / len(fields["param"])
                step_len = para_len / len(fields["step"])

                append_composite_coords(fields["dates"][-1], tree.values, fields["lat"], coords)

                for l, level in enumerate(fields["levels"]):  # noqa: E741
                    for i, num in enumerate(fields["number"]):
                        for j, para in enumerate(fields["param"]):
                            for k, s in enumerate(fields["step"]):
                                start_index, end_index = calculate_index_bounds(
                                    level_len, num_len, para_len, step_len, l, i, j, k
                                )
                                key = create_composite_key(fields["dates"][-1], level, num, para, s)
                                if key not in range_dict:
                                    range_dict[key] = []
                                range_dict[key].extend(tree.result[start_index:end_index])

    def walk_tree_step(self, tree, fields, coords, mars_metadata, range_dict):
        def create_composite_key_step(date, level, num, para):
            return (date, level, num, para)

        def handle_non_leaf_node_step(child):
            non_leaf_axes = ["latitude", "longitude", "param", "date", "time"]
            if child.axis.name not in non_leaf_axes:
                mars_metadata[child.axis.name] = child.values[0]

        def handle_specific_axes_step(child):
            if child.axis.name == "latitude":
                return child.values[0]
            if child.axis.name == "levelist":
                return child.values
            if child.axis.name == "param":
                return child.values
            if child.axis.name in ["date"]:
                dates = [f"{date}Z" for date in child.values]
                # mars_metadata["Forecast date"] = str(child.values[0])
                # for date in dates:
                #    coords[date] = {}
                #    coords[date]["composite"] = []
                #    coords[date]["t"] = [date]
                return dates
            if child.axis.name == "number":
                return child.values
            if child.axis.name == "step":
                return child.values
            if child.axis.name == "time":
                for date in fields["dates"]:
                    coords[date] = {}
                    coords[date]["composite"] = []
                    coords[date]["t"] = []
                    for time in child.values:
                        datetime = pd.Timestamp(date) + time
                        coords[date]["t"].append(str(datetime).split("+")[0] + "Z")
                return child.values
            return None

        def calculate_index_bounds_step(level_len, num_len, para_len, step_len, l, i, j, k):  # noqa: E741
            start_index = int(l * level_len) + int(i * num_len) + int(j * para_len) + int(k * step_len)
            end_index = start_index + int(step_len)
            return start_index, end_index

        def append_composite_coords_step(dates, tree_values, lat, coords):
            # for date in dates:
            for value in tree_values:
                coords[dates]["composite"].append([lat, value])

        if len(tree.children) != 0:
            for child in tree.children:
                handle_non_leaf_node_step(child)
                result = handle_specific_axes_step(child)
                if result is not None:
                    if child.axis.name == "latitude":
                        fields["lat"] = result
                    elif child.axis.name == "levelist":
                        fields["levels"] = result
                        if "l" in fields:
                            fields["l"].extend(result)
                    elif child.axis.name == "param":
                        fields["param"] = result
                    elif child.axis.name in ["date"]:
                        fields["dates"].extend(result)
                    elif child.axis.name == "number":
                        fields["number"] = result
                    elif child.axis.name == "step":
                        fields["step"] = result
                        if "s" in fields:
                            fields["s"].extend(result)
                    elif child.axis.name == "time":
                        fields["times"].extend(result)

                self.walk_tree_step(child, fields, coords, mars_metadata, range_dict)
        else:
            tree.values = [float(val) for val in tree.values]
            if all(val is None for val in tree.result):
                fields["dates"] = fields["dates"][:-1]
                for date in fields["dates"]:
                    for level in fields["levels"]:
                        for num in fields["number"]:
                            for para in fields["param"]:
                                for s in fields["step"]:
                                    key = create_composite_key_step(date, level, num, para)
                                    if key in range_dict:
                                        del range_dict[key]
            else:
                tree.result = [float(val) if val is not None else val for val in tree.result]
                date_len = len(tree.result) / len(fields["dates"])
                level_len = date_len / len(fields["levels"])
                para_len = level_len / len(fields["param"])
                # time_len = para_len / len(fields["times"])
                # coords_len = len(tree.values)

                for date in fields["dates"]:
                    append_composite_coords_step(date, tree.values, fields["lat"], coords)
                """
                for ti, _ in enumerate(fields["times"]):
                    for d, date in enumerate(fields["dates"]):
                        for l, level in enumerate(fields["levels"]):  # noqa: E741
                            for i, num in enumerate(fields["number"]):
                                for j, para in enumerate(fields["param"]):
                                    # for k, t in enumerate(fields["times"]):
                                    # start_index, end_index = calculate_index_bounds_step(
                                    #    level_len, num_len, para_len, time_len, l, i, j, k
                                    # )
                                    key = create_composite_key_step(date, level, num, para)
                                    if key not in range_dict:
                                        range_dict[key] = []
                                    # range_dict[key].extend(tree.result[start_index:end_index])
                                    # print(d, date_len,j, para_len)
                                    # print(d*date_len+j*para_len)
                                    # print(int(d*date_len+j*para_len+len(fields["times"])))
                                    # print(tree.result[int(d*date_len+j*para_len+len)])
                                    #print(tree.result)
                                    range_dict[key].append(#tree.result
                                        tree.result[
                                            int(d * date_len + j * para_len + ti * time_len) : int(
                                                d * date_len + j * para_len + ti * time_len + len(tree.values)
                                            )
                                        ]
                                    )
                """
                for d, date in enumerate(fields["dates"]):
                    for l, level in enumerate(fields["levels"]):  # noqa: E741
                        for i, num in enumerate(fields["number"]):
                            for j, para in enumerate(fields["param"]):
                                # for k, t in enumerate(fields["times"]):
                                # start_index, end_index = calculate_index_bounds_step(
                                #    level_len, num_len, para_len, time_len, l, i, j, k
                                # )
                                key = create_composite_key_step(date, level, num, para)
                                if key not in range_dict:
                                    range_dict[key] = []
                                range_dict[key].append(  # tree.result
                                    tree.result[
                                        int(d * date_len + l * level_len + j * para_len) : int(
                                            d * date_len + l * level_len + j * para_len + len(fields["times"])
                                        )
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
