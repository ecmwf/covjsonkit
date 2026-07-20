import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from .encoder import Encoder


class TimeSeries(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "PointSeries"
        self.covjson["coverages"] = []

    def add_coverage(self, mars_metadata, coords, values, include_z=False):
        new_coverage = {}
        new_coverage["mars:metadata"] = {}
        new_coverage["type"] = "Coverage"
        new_coverage["domain"] = {}
        new_coverage["ranges"] = {}
        self.add_mars_metadata(new_coverage, mars_metadata)
        self.add_domain(new_coverage, coords, include_z)
        self.add_range(new_coverage, values)
        self.covjson["coverages"].append(new_coverage)
        # cov = Coverage.model_validate_json(json.dumps(new_coverage))
        # self.pydantic_coverage.coverages.append(cov)

    def add_domain(self, coverage, coords, include_z=False):
        coverage["domain"]["type"] = "Domain"
        axes = {}
        axes["x"] = {"values": coords["longitude"]}
        axes["y"] = {"values": coords["latitude"]}
        if include_z:
            axes["z"] = {"values": coords["levelist"]}
        axes["t"] = {"values": coords["t"]}
        coverage["domain"]["axes"] = axes

    def add_range(self, coverage, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = [len(values[parameter])]
            coverage["ranges"][param]["axisNames"] = ["t"]
            coverage["ranges"][param]["values"] = values[
                parameter
            ]  # [values[parameter][val][0] for val in values[parameter].keys()]

    def _select_domain_type(self, points, include_z):
        is_multi_point = points > 1
        self.covjson["domainType"] = "MultiPointSeries" if is_multi_point else "PointSeries"
        self._set_references(include_z)
        return is_multi_point

    def _add_multi_point_coverage(self, mars_metadata, t_values, composite_values, values):
        coverage = {"mars:metadata": {}, "type": "Coverage", "domain": {"type": "Domain"}, "ranges": {}}
        self.add_mars_metadata(coverage, mars_metadata)
        coverage["domain"]["axes"] = {
            "t": {"values": t_values},
            "composite": {
                "dataType": "tuple",
                "coordinates": ["x", "y", "z"] if len(composite_values[0]) == 3 else ["x", "y"],
                "values": composite_values,
            },
        }
        for parameter, parameter_values in values.items():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {
                "type": "NdArray",
                "dataType": "float",
                "shape": [len(t_values), len(composite_values)],
                "axisNames": ["t", "composite"],
                "values": parameter_values,
            }
        self.covjson["coverages"].append(coverage)

    @staticmethod
    def _flatten_time_major(point_values):
        return [value for values_at_time in zip(*point_values) for value in values_at_time]

    @staticmethod
    def _composite_values(points, level, include_z):
        return [[longitude, latitude, level] if include_z else [longitude, latitude] for latitude, longitude in points]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def _set_references(self, include_z):
        refs = [
            {
                "coordinates": ["x", "y"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            },
            {
                "coordinates": ["t"],
                "system": {"type": "TemporalRS", "calendar": "Gregorian"},
            },
        ]
        if include_z:
            refs.append(
                {
                    "coordinates": ["z"],
                    "system": {"type": "VerticalCRS"},
                }
            )
        self.covjson["referencing"] = refs

    def from_xarray(self, datasets):
        """
        Converts an xarray dataset or a list of xarray datasets into an OGC CoverageJSON
        coverageCollection of type PointSeries.

        Args:
            datasets (Union[xarray.Dataset, List[xarray.Dataset]]): An xarray dataset or a list of xarray datasets.

        Returns:
            dict: The CoverageJSON representation of the coverage collection.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "PointSeries"
        self.covjson["coverages"] = []

        include_z = "levelist" in datasets[0].coords

        # Add reference system
        self._set_references(include_z)

        for data_var in datasets[0].data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        for dataset in datasets:

            # Process each "number" in the dataset
            for num in dataset["number"].values:
                dv_dict = {}
                for dv in dataset.data_vars:
                    dv_dict[dv] = dataset[dv].sel(number=num).values[0][0][0][0].tolist()

                mars_metadata = {}
                for metadata in dataset.attrs:
                    mars_metadata[metadata] = dataset.attrs[metadata]
                mars_metadata["number"] = int(num)

                self.add_coverage(
                    mars_metadata,
                    {
                        "latitude": [float(x) for x in dataset["latitude"].values],
                        "longitude": [float(x) for x in dataset["longitude"].values],
                        "levelist": [float(x) for x in dataset["levelist"].values] if include_z else None,
                        "t": [str(x) for x in dataset["t"].values],
                    },
                    dv_dict,
                    include_z=include_z,
                )

        return self.covjson

    def from_polytope(self, result, date_key: str = "date") -> dict:
        """Encode a polytope ``TensorIndexTree`` result into a PointSeries CoverageJSON collection.

        Args:
            result: The polytope ``TensorIndexTree`` containing the data to be converted.
            date_key: Tree axis name to treat as the time dimension
                (``"date"`` for forecasts, ``"hdate"`` for hindcast/reforecast).
        Returns:
            dict: The CoverageJSON representation of the coverage collection.
        """
        coords = {}
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["step"] = 0
        fields["dates"] = []
        fields["levels"] = [0]
        fields["has_level_axis"] = False

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)  # noqa: E501
        self.walk_tree(result, fields, coords, mars_metadata, range_dict, date_key=date_key)
        end = time.time()
        delta = end - start
        logging.debug("Tree walking ends at: %s", end)  # noqa: E501
        logging.debug("Tree walking takes: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coords creation: %s", start)  # noqa: E501

        include_z = fields["has_level_axis"]

        coordinates = {}

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        points = len(coords[fields["dates"][0]]["composite"])
        is_multi_point = self._select_domain_type(points, include_z)

        for date in fields["dates"]:
            coordinates[date] = []
            for i, point in enumerate(range(points)):
                coordinates[date].append(
                    {
                        "latitude": [coords[date]["composite"][i][0]],
                        "longitude": [coords[date]["composite"][i][1]],
                        "levelist": [levels[0]],
                    }
                )
                coordinates[date][i]["t"] = []
                for level in fields["levels"]:
                    for num in fields["number"]:
                        for para in fields["param"]:
                            for step in fields["step"]:
                                date_format = "%Y%m%dT%H%M%S"
                                new_date = pd.Timestamp(date).strftime(date_format)
                                start_time = datetime.strptime(new_date, date_format)
                                # add current date to list by converting it to iso format
                                if isinstance(step, timedelta):
                                    stamp = start_time + step
                                else:
                                    try:
                                        int(step)
                                    except ValueError:
                                        step = step[0]
                                    stamp = start_time + timedelta(hours=int(step))
                                coordinates[date][i]["t"].append(stamp.isoformat() + "Z")
                            break
                        break
                    break

        logging.debug("Coordinates created: %s", coordinates)  # noqa: E501

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        logging.debug("The points found were: %s", points)  # noqa: E501
        logging.debug("The fields retrieved were: %s", fields)  # noqa: E501
        logging.debug("The range_dict created was: %s", range_dict)  # noqa: E501

        if is_multi_point:
            for date in fields["dates"]:
                for level in fields["levels"]:
                    for num in fields["number"]:
                        val_dict = {}
                        for para in fields["param"]:
                            point_values = [
                                [range_dict[(date, level, num, para, step)][i] for step in fields["step"]]
                                for i in range(points)
                            ]
                            val_dict[para] = self._flatten_time_major(point_values)
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        mm["Forecast date"] = date
                        mm["levelist"] = level
                        del mm["step"]
                        self._add_multi_point_coverage(
                            mm,
                            coordinates[date][0]["t"],
                            self._composite_values(coords[date]["composite"], level, include_z),
                            val_dict,
                        )
        else:
            for i, point in enumerate(range(points)):
                for date in fields["dates"]:
                    for level in fields["levels"]:
                        for num in fields["number"]:
                            val_dict = {}
                            for para in fields["param"]:
                                val_dict[para] = []
                                for step in fields["step"]:
                                    key = (date, level, num, para, step)
                                    try:
                                        val_dict[para].append(range_dict[key][i])
                                    except IndexError:
                                        logging.debug(
                                            f"Index {i} out of range for key {key} in range_dict. "
                                            f"Available keys: {list(range_dict.keys())}"
                                        )
                                        raise IndexError(
                                            f"Key {key} not found in range_dict. "
                                            f"Please ensure all axes are compressed in config"
                                        )
                            mm = mars_metadata.copy()
                            mm["number"] = num
                            mm["Forecast date"] = date
                            mm["levelist"] = level
                            coordinates[date][i]["levelist"] = [level]
                            del mm["step"]
                            self.add_coverage(mm, coordinates[date][i], val_dict, include_z)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson

    def from_polytope_month(self, result):
        """Convert a Polytope result for monthly-mean streams (e.g. clmn) into CovJSON.

        These streams index time with ``year`` and ``month`` axes rather than
        ``date``/``time``/``step``.  Each (year, month) pair becomes a single
        timestep represented as the ISO-8601 string ``"YYYY-MM"``.
        """
        coords = {}
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["years"] = []
        fields["months"] = []
        fields["dates"] = []  # populated as "YYYY-MM" keys once both year and month are seen
        fields["levels"] = [0]
        fields["has_level_axis"] = False

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)
        self.walk_tree_month(result, fields, coords, mars_metadata, range_dict)
        end = time.time()
        logging.debug("Tree walking ends at: %s", end)
        logging.debug("Tree walking takes: %s", end - start)

        start = time.time()
        logging.debug("Coords creation: %s", start)

        include_z = fields["has_level_axis"]

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)

        # Use the first date key to discover how many spatial points were found.
        if not fields["dates"]:
            raise ValueError("No year/month data was found in the result tree.")
        first_date = fields["dates"][0]
        points = len(coords[first_date]["composite"])
        is_multi_point = self._select_domain_type(points, include_z)

        # Build coordinate structures: one entry per spatial point per date.
        coordinates = {}
        for date in fields["dates"]:
            coordinates[date] = []
            for i in range(points):
                coord_entry = {
                    "latitude": [coords[date]["composite"][i][0]],
                    "longitude": [coords[date]["composite"][i][1]],
                    "levelist": [fields["levels"][0]],
                    # "YYYY-MM" is the timestep; append a day so it is a valid
                    # ISO-8601 datetime string (first day of the month).
                    "t": [f"{date}-01T00:00:00Z"],
                }
                coordinates[date].append(coord_entry)

        end = time.time()
        logging.debug("Coords creation ends: %s", end)
        logging.debug("Coords creation takes: %s", end - start)

        start = time.time()
        logging.debug("Coverage creation: %s", start)

        logging.debug("The points found were: %s", points)
        logging.debug("The fields retrieved were: %s", fields)
        logging.debug("The range_dict created was: %s", range_dict)

        if is_multi_point:
            for level in fields["levels"]:
                for num in fields["number"]:
                    val_dict = {}
                    for para in fields["param"]:
                        point_values = [
                            [range_dict[(date, level, num, para)][i][0] for date in fields["dates"]]
                            for i in range(points)
                        ]
                        val_dict[para] = self._flatten_time_major(point_values)
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["levelist"] = level
                    self._add_multi_point_coverage(
                        mm,
                        [f"{date}-01T00:00:00Z" for date in fields["dates"]],
                        self._composite_values(coords[first_date]["composite"], level, include_z),
                        val_dict,
                    )
        else:
            for i in range(points):
                for j, level in enumerate(fields["levels"]):
                    for num in fields["number"]:
                        val_dict = {}
                        for para in fields["param"]:
                            val_dict[para] = []
                            for date in fields["dates"]:
                                key = (date, level, num, para)
                                try:
                                    val_dict[para].extend(range_dict[key][i])
                                except (KeyError, IndexError) as exc:
                                    logging.debug(
                                        "Key %s not found or index %s out of range in range_dict: %s",
                                        key,
                                        i,
                                        exc,
                                    )
                                    raise
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        mm["levelist"] = level
                        # Use all date keys as the time series for this coverage.
                        coord_entry = coordinates[first_date][i].copy()
                        coord_entry["levelist"] = [level]
                        coord_entry["t"] = [f"{date}-01T00:00:00Z" for date in fields["dates"]]
                        self.add_coverage(mm, coord_entry, val_dict, include_z)

        end = time.time()
        logging.debug("Coverage creation ends: %s", end)
        logging.debug("Coverage creation takes: %s", end - start)

        return self.covjson

    def from_polytope_step(self, result):
        coords = {}
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["step"] = [0]
        fields["dates"] = []
        fields["levels"] = [0]
        fields["times"] = []
        fields["has_level_axis"] = False

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)  # noqa: E501
        self.walk_tree_step(result, fields, coords, mars_metadata, range_dict)
        end = time.time()
        delta = end - start
        logging.debug("Tree walking ends at: %s", end)  # noqa: E501
        logging.debug("Tree walking takes: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coords creation: %s", start)  # noqa: E501

        include_z = fields["has_level_axis"]

        coordinates = {}

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        points = len(coords[fields["dates"][0]]["composite"])
        is_multi_point = self._select_domain_type(points, include_z)

        for step in fields["step"]:
            coordinates[fields["dates"][0]] = []
            for i, point in enumerate(range(points)):
                for j, level in enumerate(fields["levels"]):
                    coordinates[fields["dates"][0]].append(
                        {
                            "latitude": [coords[fields["dates"][0]]["composite"][i][0]],
                            "longitude": [coords[fields["dates"][0]]["composite"][i][1]],
                            "levelist": [level],
                        }
                    )
                    coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)]["t"] = []
                    for num in fields["number"]:
                        for para in fields["param"]:
                            for date in fields["dates"]:
                                for times in fields["times"]:
                                    # date_format = "%Y%m%dT%H%M%S"
                                    # new_date = pd.Timestamp(date).strftime(date_format)
                                    # start_time = datetime.strptime(new_date, date_format)
                                    # add current date to list by converting it to iso format
                                    # stamp = start_time + timedelta(hours=int(step))
                                    datetime = pd.Timestamp(date) + times
                                    coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)]["t"].append(
                                        str(datetime).split("+")[0] + "Z"
                                    )
                            break
                        break

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        if is_multi_point:
            for date in fields["dates"]:
                for level in fields["levels"]:
                    for num in fields["number"]:
                        val_dict = {}
                        for para in fields["param"]:
                            point_values = [range_dict[(date, level, num, para)][i] for i in range(points)]
                            val_dict[para] = self._flatten_time_major(point_values)
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        mm["Forecast date"] = date
                        self._add_multi_point_coverage(
                            mm,
                            [str(pd.Timestamp(date) + time).split("+")[0] + "Z" for time in fields["times"]],
                            self._composite_values(coords[date]["composite"], level, include_z),
                            val_dict,
                        )
        else:
            for i, point in enumerate(range(points)):
                for j, level in enumerate(fields["levels"]):
                    for num in fields["number"]:
                        val_dict = {}
                        for para in fields["param"]:
                            val_dict[para] = []
                            for date in fields["dates"]:
                                key = (date, level, num, para)
                                # for k, v in range_dict.items():
                                #    if k == key:
                                # val_dict[para].append(v[0])
                                val_dict[para].extend(range_dict[key][i])
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        mm["Forecast date"] = date
                        self.add_coverage(
                            mm,
                            coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)],
                            val_dict,
                            include_z,
                        )

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson
