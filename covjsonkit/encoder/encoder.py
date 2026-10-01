from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import orjson
import pandas as pd
from covjson_pydantic.coverage import CoverageCollection
from covjson_pydantic.domain import DomainType

from covjsonkit.param_db import get_param_ids, get_params, get_units

try:
    # Polytope compacts unstructured-grid (e.g. ICON, Lambert LAM) leaves into a single
    # MergedTensorIndexNode holding axes=(lat_axis, lon_axis) and values=(lat, lon).
    from polytope_feature.datacube.tensor_index_tree import MergedTensorIndexNode
except ImportError:  # older polytope without merged nodes
    MergedTensorIndexNode = None


def is_merged_node(node) -> bool:
    """True if ``node`` is a polytope ``MergedTensorIndexNode`` (a compacted lat/lon leaf).

    Such nodes carry ``axes=(lat_axis, lon_axis)`` and ``values=(lat, lon)`` for a single
    spatial point, and are always leaves. Falls back to duck-typing (``.axes`` present)
    if the polytope import was unavailable.
    """
    if MergedTensorIndexNode is not None:
        return isinstance(node, MergedTensorIndexNode)
    return hasattr(node, "axes") and getattr(node, "axes", None) is not None


def timedelta_to_step_string(td: timedelta) -> str:
    """
    Convert a timedelta object to a step string in the format 'XhYm'.

    Args:
        td: timedelta object representing the step

    Returns:
        String in format 'Xh', 'Ym', or 'XhYm' depending on the timedelta value

    Examples:
        timedelta(hours=13) -> "13h"
        timedelta(hours=13, minutes=30) -> "13h30m"
        timedelta(minutes=30) -> "30m"
        timedelta(hours=0, minutes=0) -> "0h"
    """
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    if hours > 0 and minutes > 0:
        return f"{hours}h{minutes}m"
    elif hours > 0:
        return f"{hours}h"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return "0h"


def parse_step_string(step_str: str) -> float:
    """
    Parse a step string in format 'XhYm' and return total hours as float.

    Args:
        step_str: Step string in format 'Xh', 'Ym', or 'XhYm'

    Returns:
        Total hours as a float value

    Examples:
        parse_step_string("13h") -> 13.0
        parse_step_string("13h30m") -> 13.5
        parse_step_string("30m") -> 0.5
        parse_step_string("0h") -> 0.0
    """
    if isinstance(step_str, (int, float)):
        return float(step_str)

    step_str = str(step_str)
    hours = 0.0
    minutes = 0.0

    # Parse hours
    if "h" in step_str:
        parts = step_str.split("h")
        hours = float(parts[0])
        step_str = parts[1] if len(parts) > 1 else ""

    # Parse minutes
    if "m" in step_str and step_str:
        minutes = float(step_str.replace("m", ""))

    return hours + (minutes / 60.0)


def sort_step_values(steps: list) -> list:
    """
    Sort a list of step values that might be in various formats.

    Args:
        steps: List of step values (strings like "13h30m", integers, or floats)

    Returns:
        Sorted list of step values in their original format

    Examples:
        sort_step_values(["13h30m", "12h", "14h"]) -> ["12h", "13h30m", "14h"]
    """
    # Create tuples of (original_value, numeric_value) for sorting
    step_tuples = [(step, parse_step_string(step)) for step in steps]
    # Sort by numeric value
    step_tuples.sort(key=lambda x: x[1])
    # Return original values in sorted order
    return [step[0] for step in step_tuples]


def normalize_step_value(step):
    """
    Normalize a step value from various formats, preserving integers when possible.

    Handles:
    - timedelta objects (datetime.timedelta) - converts to int if whole hours, else string "XhYm"
    - numpy.timedelta64 objects - converts to int if whole hours, else string "XhYm"
    - integers - returns as-is (legacy compatibility)
    - floats - converts to int if whole hours, else string "XhYm"
    - strings - returns as-is if already in correct format

    Args:
        step: Step value in various formats

    Returns:
        Integer for whole hour values, string in 'XhYm' format for sub-hourly values

    Examples:
        normalize_step_value(timedelta(hours=13, minutes=30)) -> "13h30m"
        normalize_step_value(timedelta(hours=13)) -> 13
        normalize_step_value(13) -> 13
        normalize_step_value(13.5) -> "13h30m"
        normalize_step_value("13h30m") -> "13h30m"
    """
    # If it's already a string, return as-is (assumes it's already in correct format)
    if isinstance(step, str):
        return step

    # If it's a timedelta or pd.Timedelta, check if it has sub-hourly components
    if isinstance(step, (timedelta, pd.Timedelta)):
        if isinstance(step, pd.Timedelta):
            td = step.to_pytimedelta()
        else:
            td = step

        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        # If there are minutes, return string format
        if minutes > 0:
            return timedelta_to_step_string(td)
        # Otherwise return integer hours
        return hours

    # If it's a numpy timedelta64, convert to Python timedelta first
    if isinstance(step, np.timedelta64):
        td = pd.Timedelta(step).to_pytimedelta()
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if minutes > 0:
            return timedelta_to_step_string(td)
        return hours

    # If it's an integer, return as-is (legacy compatibility)
    if isinstance(step, (int, np.integer)):
        return int(step)

    # If it's a float, check if it has fractional hours
    if isinstance(step, (float, np.floating)):
        hours = int(step)
        total_minutes = int(step * 60)
        minutes = total_minutes % 60

        if minutes > 0:
            if hours > 0:
                return f"{hours}h{minutes}m"
            else:
                return f"{minutes}m"
        return hours

    # Fallback: convert to string
    return str(step)


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
        elif domaintype == "grid":
            self.domaintype = "Grid"
        elif domaintype == "position":
            self.domaintype = DomainType.point_series

        # Trajectory not yet implemented in covjson-pydantic
        if self.domaintype != "Trajectory" and self.domaintype != "Grid":
            self.pydantic_coverage = CoverageCollection(
                type="CoverageCollection", coverages=[], domainType=self.domaintype, parameters={}, referencing=[]
            )
        self.parameters = []

    def add_parameter(self, param):
        # param_dict = get_param_from_db(param)
        # unit = get_unit_from_db(param_dict["unit_id"])
        param_dict = self.params[str(param)]
        if isinstance(param_dict["unit_id"], str):
            unit = {"name": param_dict["unit_id"]}
        else:
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

    def walk_tree(
        self,
        tree,
        fields: dict[str, Any],
        coords: dict[str, dict[str, list]],
        mars_metadata: dict[str, Any],
        range_dict: dict[tuple, list],
        date_key: str = "date",
    ) -> None:
        """Walk the polytope result tree, extracting data into fields, coords, and range_dict.

        ``date_key`` controls which tree axis is treated as the time dimension
        (e.g. ``"date"`` for forecasts, ``"hdate"`` for hindcast/reforecast data).
        Any other axis with the default name falls through to ``mars_metadata``
        instead.  Regardless of ``date_key``, values are always stored under
        ``fields["dates"]``.
        """

        def create_composite_key(date, level, num, para, s):
            return (date, level, num, para, s)

        def handle_non_leaf_node(child):
            non_leaf_axes = ["latitude", "longitude", "param", date_key]
            if child.axis.name not in non_leaf_axes:
                val = child.values[0]
                if isinstance(val, np.datetime64):
                    val = str(val)
                elif isinstance(val, timedelta):
                    val = timedelta_to_step_string(val)
                elif child.axis.name == "step":
                    # Step is not a timedelta! Need to normalize it
                    val = normalize_step_value(val)
                mars_metadata[child.axis.name] = val

        def handle_specific_axes(child):
            if child.axis.name == "latitude":
                return child.values[0]
            if child.axis.name == "levelist":
                return child.values
            if child.axis.name == "param":
                return child.values
            if child.axis.name in [date_key, "time"]:
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

        def emit_leaf(lat, lon_values, result):
            """Emit one spatial leaf: append [lat, lon] composite coords and slice results.

            Shared by the legacy longitude-leaf path (``lat`` from the parent latitude
            node, ``lon_values`` = the leaf's list of longitudes) and the compacted
            ``MergedTensorIndexNode`` path (single point: ``lon_values`` = [lon]).
            """
            lon_values = [float(val) for val in lon_values]
            if all(val is None for val in result):
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
                result = [float(val) if val is not None else val for val in result]
                level_len = len(result) / len(fields["levels"])
                num_len = level_len / len(fields["number"])
                para_len = num_len / len(fields["param"])
                step_len = para_len / len(fields["step"])

                append_composite_coords(fields["dates"][-1], lon_values, lat, coords)

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
                                range_dict[key].extend(result[start_index:end_index])

        if len(tree.children) != 0:
            for child in tree.children:
                # Compacted unstructured leaf: values=(lat, lon), own result. Emit directly.
                if is_merged_node(child):
                    emit_leaf(child.values[0], [child.values[1]], child.result)
                    continue
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
                    elif child.axis.name in [date_key, "time"]:
                        fields["dates"].extend(result)
                    elif child.axis.name == "number":
                        fields["number"] = result
                    elif child.axis.name == "step":
                        fields["step"] = result
                        if "s" in fields:
                            fields["s"].extend(result)

                self.walk_tree(child, fields, coords, mars_metadata, range_dict, date_key=date_key)
        else:
            emit_leaf(fields["lat"], tree.values, tree.result)

    def walk_tree_reforecast(self, tree, fields, coords, mars_metadata, range_dict):
        """Walk the result tree for reforecast/reanalysis with an independent ``time`` axis.

        Unlike :meth:`walk_tree` (which, for the legacy merged representation,
        folds any ``time`` axis into the ``hdate`` time dimension), this walker
        treats ``hdate`` as the branching time axis and captures the separate
        ``time`` axis as a scalar time-of-day offset stored in
        ``fields["time_offset"]``. The valid-time for each hdate is then
        ``hdate + time_offset + step`` (computed downstream in the collapse step).

        For efcl the ``date`` and ``step`` axes are single-valued and ``time`` is a
        single timedelta, so the offset is a scalar. Backward compatibility with
        merged trees (no separate ``time`` node) is preserved: ``time_offset``
        simply stays absent and the timestamps reduce to ``hdate + step``.
        """
        date_key = "hdate"

        def create_composite_key(date, level, num, para, s):
            return (date, level, num, para, s)

        def handle_non_leaf_node(child):
            non_leaf_axes = ["latitude", "longitude", "param", date_key, "time"]
            if child.axis.name not in non_leaf_axes:
                val = child.values[0]
                if isinstance(val, np.datetime64):
                    val = str(val)
                elif isinstance(val, timedelta):
                    val = timedelta_to_step_string(val)
                elif child.axis.name == "step":
                    # Step is not a timedelta! Need to normalize it
                    val = normalize_step_value(val)
                mars_metadata[child.axis.name] = val

        def handle_specific_axes(child):
            if child.axis.name == "latitude":
                return child.values[0]
            if child.axis.name == "levelist":
                return child.values
            if child.axis.name == "param":
                return child.values
            if child.axis.name == date_key:
                dates = [f"{date}Z" for date in child.values]
                mars_metadata["Forecast date"] = str(child.values[0])
                for date in dates:
                    coords[date] = {}
                    coords[date]["composite"] = []
                    coords[date]["t"] = [date]
                return dates
            if child.axis.name == "time":
                # Independent time-of-day axis: capture as a scalar offset rather
                # than folding it into the hdate time dimension. efcl guarantees a
                # single time value; take the first if a span is ever returned.
                fields["time_offset"] = child.values[0]
                return None
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
            for value in tree_values:
                coords[dates]["composite"].append([lat, value])

        def emit_leaf(lat, lon_values, result):
            lon_values = [float(val) for val in lon_values]
            if all(val is None for val in result):
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
                result = [float(val) if val is not None else val for val in result]
                level_len = len(result) / len(fields["levels"])
                num_len = level_len / len(fields["number"])
                para_len = num_len / len(fields["param"])
                step_len = para_len / len(fields["step"])

                append_composite_coords(fields["dates"][-1], lon_values, lat, coords)

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
                                range_dict[key].extend(result[start_index:end_index])

        if len(tree.children) != 0:
            for child in tree.children:
                # Compacted unstructured leaf: values=(lat, lon), own result. Emit directly.
                if is_merged_node(child):
                    emit_leaf(child.values[0], [child.values[1]], child.result)
                    continue
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
                    elif child.axis.name == date_key:
                        fields["dates"].extend(result)
                    elif child.axis.name == "number":
                        fields["number"] = result
                    elif child.axis.name == "step":
                        fields["step"] = result
                        if "s" in fields:
                            fields["s"].extend(result)

                self.walk_tree_reforecast(child, fields, coords, mars_metadata, range_dict)
        else:
            emit_leaf(fields["lat"], tree.values, tree.result)

    def walk_tree_step(self, tree, fields, coords, mars_metadata, range_dict):
        def create_composite_key_step(date, level, num, para):
            return (date, level, num, para)

        def handle_non_leaf_node_step(child):
            non_leaf_axes = ["latitude", "longitude", "param", "date", "time"]
            if child.axis.name not in non_leaf_axes:
                val = child.values[0]
                if isinstance(val, np.datetime64):
                    val = str(val)
                elif isinstance(val, timedelta):
                    val = timedelta_to_step_string(val)
                elif child.axis.name == "step":
                    # Step is not a timedelta! Need to normalize it
                    val = normalize_step_value(val)
                mars_metadata[child.axis.name] = val

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

        def emit_leaf_step(lat, lon_values, result):
            """Emit one spatial leaf for the step walker (shared by legacy and merged)."""
            lon_values = [float(val) for val in lon_values]
            if all(val is None for val in result):
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
                result = [float(val) if val is not None else val for val in result]
                date_len = len(result) / len(fields["dates"])
                level_len = date_len / len(fields["levels"])
                para_len = level_len / len(fields["param"])

                for date in fields["dates"]:
                    append_composite_coords_step(date, lon_values, lat, coords)

                for d, date in enumerate(fields["dates"]):
                    for l, level in enumerate(fields["levels"]):  # noqa: E741
                        for i, num in enumerate(fields["number"]):
                            for j, para in enumerate(fields["param"]):
                                key = create_composite_key_step(date, level, num, para)
                                if key not in range_dict:
                                    range_dict[key] = []
                                range_dict[key].append(
                                    result[
                                        int(d * date_len + l * level_len + j * para_len) : int(
                                            d * date_len + l * level_len + j * para_len + len(fields["times"])
                                        )
                                    ]
                                )

        if len(tree.children) != 0:
            for child in tree.children:
                # Compacted unstructured leaf: values=(lat, lon), own result. Emit directly.
                if is_merged_node(child):
                    emit_leaf_step(child.values[0], [child.values[1]], child.result)
                    continue
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
            emit_leaf_step(fields["lat"], tree.values, tree.result)

    def walk_tree_month(self, tree, fields, coords, mars_metadata, range_dict, _ctx=None):
        """Walk the result tree for monthly-mean streams (e.g. clmn).

        These streams use ``year`` and ``month`` axes instead of ``date``/``time``/``step``.
        Each unique (year, month) pair is represented as an ISO-8601 date string
        ``"YYYY-MM"`` that plays the same role as ``date`` does in the standard
        step-based tree walker.

        Dates are accumulated in the order they are actually encountered during
        tree traversal, so the ordering correctly reflects the tree structure
        (e.g. month-major when the tree has month as the outer axis).
        """
        if _ctx is None:
            _ctx = {}

        def _year_month_key(year, month):
            return f"{int(year):04d}-{int(month):02d}"

        def _register_date_key(key):
            if key not in fields["dates"]:
                fields["dates"].append(key)
            if key not in coords:
                coords[key] = {"composite": [], "t": [key]}

        def handle_non_leaf_node_month(child):
            non_leaf_axes = ["latitude", "longitude", "param", "year", "month"]
            if child.axis.name not in non_leaf_axes:
                val = child.values[0]
                if isinstance(val, np.datetime64):
                    val = str(val)
                mars_metadata[child.axis.name] = val

        def handle_specific_axes_month(child):
            if child.axis.name == "latitude":
                return child.values[0]
            if child.axis.name == "levelist":
                return child.values
            if child.axis.name == "param":
                return child.values
            if child.axis.name == "year":
                return child.values
            if child.axis.name == "month":
                return child.values
            if child.axis.name == "number":
                return child.values
            return None

        def append_composite_coords_month(date_key, tree_values, lat):
            for value in tree_values:
                coords[date_key]["composite"].append([lat, value])

        def emit_leaf_month(lat, lon_values, result):
            """Emit one spatial leaf for the month walker (shared by legacy and merged).

            ``lat`` is the latitude for this leaf, ``lon_values`` the leaf's list of
            longitudes (length 1 for a compacted ``MergedTensorIndexNode``), and
            ``result`` its flat value array.
            """
            # Leaf node — ensure all (year, month) combinations from context are registered.
            ctx_years = _ctx.get("years", fields.get("years", []))
            ctx_months = _ctx.get("months", fields.get("months", []))
            for y in ctx_years:
                for m in ctx_months:
                    _register_date_key(_year_month_key(y, m))

            # Determine the dates in scope for this specific leaf. The loop order
            # must match the actual tree axis order (outermost axis first) so that
            # the flat result array is sliced correctly.
            # _axis_order records axes in the order they were encountered top-down;
            # the first entry is the outer axis at this leaf.
            if ctx_years and ctx_months:
                axis_order = _ctx.get("_axis_order", [])
                # Default: if year was seen before month in the tree, year is outer.
                year_is_outer = (
                    axis_order.index("year") < axis_order.index("month")
                    if ("year" in axis_order and "month" in axis_order)
                    else True
                )
                if year_is_outer:
                    leaf_dates = [_year_month_key(y, m) for y in ctx_years for m in ctx_months]
                else:
                    leaf_dates = [_year_month_key(y, m) for m in ctx_months for y in ctx_years]
            else:
                leaf_dates = fields["dates"]

            lon_values = [float(val) for val in lon_values]
            if all(val is None for val in result):
                # Remove date entries for this leaf that produced no data.
                for key in leaf_dates:
                    if key in fields["dates"]:
                        fields["dates"].remove(key)
                    for level in fields["levels"]:
                        for num in fields["number"]:
                            for para in fields["param"]:
                                rkey = (key, level, num, para)
                                if rkey in range_dict:
                                    del range_dict[rkey]
            else:
                result = [float(val) if val is not None else val for val in result]

                n_dates = len(leaf_dates)
                n_levels = len(fields["levels"])
                n_params = len(fields["param"])

                date_len = len(result) / n_dates if n_dates else len(result)
                level_len = date_len / n_levels if n_levels else date_len
                para_len = level_len / n_params if n_params else level_len

                # Append this leaf's longitude values to composite coords for
                # every date key in scope.
                for date in leaf_dates:
                    append_composite_coords_month(date, lon_values, lat)

                for d, date in enumerate(leaf_dates):
                    for l, level in enumerate(fields["levels"]):  # noqa: E741
                        for i, num in enumerate(fields["number"]):
                            for j, para in enumerate(fields["param"]):
                                key = (date, level, num, para)
                                if key not in range_dict:
                                    range_dict[key] = []
                                start = int(d * date_len + l * level_len + j * para_len)
                                end = int(start + len(lon_values))
                                range_dict[key].append(result[start:end])

        if len(tree.children) != 0:
            for child in tree.children:
                # Compacted unstructured leaf: values=(lat, lon), own result. Emit directly.
                if is_merged_node(child):
                    emit_leaf_month(child.values[0], [child.values[1]], child.result)
                    continue
                handle_non_leaf_node_month(child)
                result = handle_specific_axes_month(child)

                # Build a child context that inherits the current year/month
                child_ctx = dict(_ctx)

                if result is not None:
                    if child.axis.name == "latitude":
                        fields["lat"] = result
                    elif child.axis.name == "levelist":
                        fields["levels"] = result
                        if "l" in fields:
                            fields["l"].extend(result)
                    elif child.axis.name == "param":
                        fields["param"] = result
                    elif child.axis.name == "year":
                        fields["years"] = list(result) if fields.get("years") == [] else fields["years"]
                        child_ctx["years"] = result
                        child_ctx["_axis_order"] = _ctx.get("_axis_order", []) + ["year"]
                        # If month is already fixed in context, register dates now.
                        if "months" in _ctx:
                            for y in result:
                                for m in _ctx["months"]:
                                    _register_date_key(_year_month_key(y, m))
                    elif child.axis.name == "month":
                        fields["months"] = list(result) if fields.get("months") == [] else fields["months"]
                        child_ctx["months"] = result
                        # If year is already fixed in context, register dates now.
                        if "years" in _ctx:
                            for y in _ctx["years"]:
                                for m in result:
                                    _register_date_key(_year_month_key(y, m))
                        # Track that month is the inner axis relative to year
                        child_ctx["_axis_order"] = _ctx.get("_axis_order", []) + ["month"]
                    elif child.axis.name == "number":
                        fields["number"] = result

                self.walk_tree_month(child, fields, coords, mars_metadata, range_dict, _ctx=child_ctx)
        else:
            emit_leaf_month(fields["lat"], tree.values, tree.result)

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
    def from_polytope(self, result, date_key: str = "date") -> dict:
        pass

    @staticmethod
    def _reforecast_stringify(value):
        """Coerce datetime/timedelta-like values to strings for mars:metadata."""
        if isinstance(value, (pd.Timestamp, datetime)):
            # str() renders "YYYY-MM-DD HH:MM:SS"; emit ISO-8601 instead.
            return value.isoformat()
        if isinstance(value, (np.datetime64, np.timedelta64, timedelta)):
            return str(value)
        return value

    @staticmethod
    def _reforecast_timedelta(value) -> timedelta:
        """Coerce a ``time``/offset value to a :class:`datetime.timedelta`."""
        if value is None:
            return timedelta(0)
        if isinstance(value, timedelta):
            return value
        return pd.to_timedelta(value).to_pytimedelta()

    @staticmethod
    def _reforecast_step_timedelta(step) -> timedelta:
        """Coerce a normalized ``step`` value to a :class:`datetime.timedelta`."""
        if isinstance(step, timedelta):
            return step
        if isinstance(step, np.timedelta64):
            return pd.to_timedelta(step).to_pytimedelta()
        if isinstance(step, str):
            return timedelta(hours=parse_step_string(step))
        try:
            return timedelta(hours=float(step))
        except (TypeError, ValueError):
            return timedelta(0)

    @staticmethod
    def _reforecast_reference(rec):
        """Reference datetime for a record: ``hdate (or date) + time``, ISO ``...Z``."""
        hdate = rec.get("hdate", rec.get("date"))
        ref = pd.Timestamp(hdate) + Encoder._reforecast_timedelta(rec.get("time"))
        return ref

    @staticmethod
    def _tree_has_axis(tree, axis_name) -> bool:
        stack = [tree]
        while stack:
            node = stack.pop()
            for child in getattr(node, "children", []):
                if is_merged_node(child):
                    continue
                if getattr(getattr(child, "axis", None), "name", None) == axis_name:
                    return True
                stack.append(child)
        return False

    def _reforecast_records(self, result):
        """Flatten a reforecast result tree into per-value records.

        Every leaf value is expanded into a dict of ``{axis_name: value}`` for
        the full root-to-leaf path (latitude/longitude included), using the same
        ``itertools.product`` layout the compressed leaf ``result`` array follows.
        Handles both the compacted ``MergedTensorIndexNode`` lat/lon leaves and
        the classic nested latitude/longitude branches.
        """
        import itertools

        records = []

        def emit(full_path, flat_result):
            axis_names = [name for name, _ in full_path]
            axis_values = [values for _, values in full_path]
            for idx, combo in enumerate(itertools.product(*axis_values)):
                value = flat_result[idx]
                if value is None:
                    continue
                records.append(dict(zip(axis_names, combo)))
                records[-1]["__value__"] = value

        def recurse(node, path):
            children = node.children
            if len(children) == 0:
                emit(path, node.result)
                return
            for child in children:
                if is_merged_node(child):
                    lat, lon = child.values[0], child.values[1]
                    emit(
                        path + [("latitude", (lat,)), ("longitude", (lon,))],
                        child.result,
                    )
                    continue
                recurse(child, path + [(child.axis.name, tuple(child.values))])

        recurse(result, [])
        return records

    def from_polytope_reforecast(self, result) -> dict:
        """Encode reforecast/reanalysis data that uses ``"hdate"`` as the time axis.

        Two representations are supported:

        * **Legacy merged** trees (no independent ``time`` node): ``hdate`` is the
          branching time axis and each hdate/step produces its own coverage. This
          is delegated to :meth:`from_polytope` with ``date_key="hdate"``.
        * **Separate-datetime** trees (``class=ce``) where ``date``, ``hdate`` and
          ``time`` are independent axes: the reforecast reference datetime is
          ``hdate + time`` and one coverage is produced per
          ``(reference-datetime, step, number)`` combination, holding all spatial
          points in its ``composite`` axis.
        """
        if not self._tree_has_axis(result, "time"):
            return self.from_polytope(result, date_key="hdate")

        self.add_reference(
            {
                "coordinates": ["latitude", "longitude", "levelist"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        # Axes that should not leak into the per-coverage mars:metadata block.
        exclude_meta = {
            "latitude",
            "longitude",
            "hdate",
            "time",
            "step",
            "param",
            "number",
            "levelist",
        }

        def stringify(value):
            return self._reforecast_stringify(value)

        def to_timedelta(value):
            return self._reforecast_timedelta(value)

        coverages = {}
        coverage_order = []
        param_order = []

        for rec in self._reforecast_records(result):
            value = rec["__value__"]
            lat = float(rec["latitude"])
            lon = float(rec["longitude"])
            level = rec.get("levelist", 0)
            try:
                level = int(level)
            except (TypeError, ValueError):
                pass
            number = rec.get("number", 0)
            try:
                number = int(number)
            except (TypeError, ValueError):
                pass
            para = rec.get("param")
            step = normalize_step_value(rec.get("step", 0))
            hdate = rec.get("hdate", rec.get("date"))
            ref = pd.Timestamp(hdate) + to_timedelta(rec.get("time"))
            valid = ref + self._reforecast_step_timedelta(step)

            # One coverage per (reference, step, number); reference is
            # hdate + time for efcl and date + time (the run) for efas.
            key = (ref, step, number)
            if key not in coverages:
                meta = {}
                for name in rec:
                    if name == "__value__" or name in exclude_meta:
                        continue
                    meta[name] = stringify(rec[name])
                meta["number"] = number
                meta["step"] = step
                is_ce = rec.get("class") == "ce"
                if is_ce and rec.get("stream") == "efas":
                    # date + time is the run reference; drop the raw date so it
                    # doesn't duplicate "Forecast date".
                    meta.pop("date", None)
                    meta["Forecast date"] = ref.isoformat() + "Z"
                elif not (is_ce and rec.get("stream") == "efcl"):
                    # efcl coverages are delineated by their valid time, which
                    # the t-axis already carries, so they get no "Forecast date".
                    meta["Forecast date"] = ref.isoformat() + "Z"
                coverages[key] = {
                    "valid": valid.isoformat() + "Z",
                    "points": [],
                    "point_index": {},
                    "values": {},
                    "meta": meta,
                }
                coverage_order.append(key)

            cov = coverages[key]
            point = (lat, lon, level)
            if point not in cov["point_index"]:
                cov["point_index"][point] = len(cov["points"])
                cov["points"].append(point)
            if para not in param_order:
                param_order.append(para)
            cov["values"].setdefault(para, {})[point] = float(value)

        if not coverages:
            raise ValueError("No data was returned.")

        for para in param_order:
            self.add_parameter(para)

        # Emit date -> time -> step -> number regardless of tree axis order.
        coverage_order.sort(key=lambda k: (k[0], self._reforecast_step_timedelta(k[1]), k[2]))

        for key in coverage_order:
            cov = coverages[key]
            composite = [[lat, lon, level] for (lat, lon, level) in cov["points"]]
            coords = {"composite": composite, "t": [cov["valid"]]}
            val_dict = {}
            for para, point_vals in cov["values"].items():
                val_dict[para] = [point_vals[pt] for pt in cov["points"]]
            self.add_coverage(cov["meta"], coords, val_dict)

        return self.covjson
