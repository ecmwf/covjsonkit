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

    def add_coverage(self, mars_metadata, coords, values):
        new_coverage = {}
        new_coverage["mars:metadata"] = {}
        new_coverage["type"] = "Coverage"
        new_coverage["domain"] = {}
        new_coverage["ranges"] = {}
        self.add_mars_metadata(new_coverage, mars_metadata)
        self.add_domain(new_coverage, coords)
        self.add_range(new_coverage, values)
        self.covjson["coverages"].append(new_coverage)
        # cov = Coverage.model_validate_json(json.dumps(new_coverage))
        # self.pydantic_coverage.coverages.append(cov)

    def add_domain(self, coverage, coords):
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = {}
        coverage["domain"]["axes"]["x"] = {}
        coverage["domain"]["axes"]["y"] = {}
        coverage["domain"]["axes"]["z"] = {}
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["x"]["values"] = coords["x"]
        coverage["domain"]["axes"]["y"]["values"] = coords["y"]
        coverage["domain"]["axes"]["z"]["values"] = coords["z"]
        coverage["domain"]["axes"]["t"]["values"] = coords["t"]

    def add_range(self, coverage, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = [len(values[parameter])]
            coverage["ranges"][param]["axisNames"] = [str(param)]
            coverage["ranges"][param]["values"] = values[
                parameter
            ]  # [values[parameter][val][0] for val in values[parameter].keys()]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def from_xarray(self, dataset):
        for data_var in dataset.data_vars:
            self.add_parameter(data_var)

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        for num in dataset["number"].values:
            dv_dict = {}
            for dv in dataset.data_vars:
                dv_dict[dv] = list(dataset[dv].sel(number=num).values[0][0][0])
            mars_metadata = {}
            for metadata in dataset.attrs:
                mars_metadata[metadata] = dataset.attrs[metadata]
            mars_metadata["number"] = num
            self.add_coverage(
                mars_metadata,
                {
                    "x": list(dataset["x"].values),
                    "y": list(dataset["y"].values),
                    "z": list(dataset["z"].values),
                    "t": [str(x) for x in dataset["t"].values],
                },
                dv_dict,
            )
        return self.covjson

    def from_polytope(self, result):
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

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)  # noqa: E501
        self.walk_tree(result, fields, coords, mars_metadata, range_dict)
        end = time.time()
        delta = end - start
        logging.debug("Tree walking ends at: %s", end)  # noqa: E501
        logging.debug("Tree walking takes: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coords creation: %s", start)  # noqa: E501

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        coordinates = {}

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No parameters were returned, date requested may be out of range")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        for date in fields["dates"]:
            coordinates[date] = {
                "x": [coords[date]["composite"][0][0]],
                "y": [coords[date]["composite"][0][1]],
                "z": [levels[0]],
            }
            coordinates[date]["t"] = []
            for level in fields["levels"]:
                for num in fields["number"]:
                    for para in fields["param"]:
                        for step in fields["step"]:
                            date_format = "%Y%m%dT%H%M%S"
                            new_date = pd.Timestamp(date).strftime(date_format)
                            start_time = datetime.strptime(new_date, date_format)
                            # add current date to list by converting it to iso format
                            stamp = start_time + timedelta(hours=int(step))
                            coordinates[date]["t"].append(stamp.isoformat() + "Z")
                        break
                    break
                break

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        # logging.debug("The values returned from walking tree: %s", range_dict)  # noqa: E501
        # logging.debug("The coordinates returned from walking tree: %s", coordinates)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        for date in fields["dates"]:
            for level in fields["levels"]:
                for num in fields["number"]:
                    val_dict = {}
                    for para in fields["param"]:
                        val_dict[para] = []
                        for step in fields["step"]:
                            key = (date, level, num, para, step)
                            # for k, v in range_dict.items():
                            #    if k == key:
                            # val_dict[para].append(v[0])
                            val_dict[para].append(range_dict[key][0])
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["Forecast date"] = date
                    del mm["step"]
                    self.add_coverage(mm, coordinates[date], val_dict)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson

    def from_polytope_step(self, result):
        coords = {}
        mars_metadata = {}
        range_dict = {}
        lat = 0
        param = 0
        number = [0]
        step = 0
        levels = [0]
        dates = [0]

        self.walk_tree(result, lat, coords, mars_metadata, param, range_dict, number, step, dates, levels)

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        coordinates = {}
        for date in range_dict.keys():
            for num in range_dict[date].keys():
                for para in range_dict[date][num].keys():
                    for step in range_dict[date][num][para].keys():
                        for dt in range_dict.keys():
                            date_format = "%Y%m%dT%H%M%S"
                            new_date = pd.Timestamp(dt).strftime(date_format)
                            start_time = datetime.strptime(new_date, date_format)
                            # add current date to list by converting it to iso format
                            stamp = start_time + timedelta(hours=int(step))
                            if step not in coordinates:
                                coordinates[step] = {
                                    "x": [coords[date]["composite"][0][0]],
                                    "y": [coords[date]["composite"][0][1]],
                                    "z": "sfc",
                                }
                            if "t" not in coordinates[step]:
                                coordinates[step]["t"] = [stamp.isoformat() + "Z"]
                            else:
                                coordinates[step]["t"].append(stamp.isoformat() + "Z")
                    break
                break
            break

        for date in range_dict.keys():
            for num in range_dict[date].keys():
                for param in range_dict[date][num].keys():
                    step_dict = {}
                    for step in range_dict[date][num][param].keys():
                        if step not in step_dict:
                            step_dict[step] = []
                        step_dict[step].append(range_dict[date][num][param][step])
                    for step in range_dict[date][num][param].keys():
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        mm["Forecast date"] = date
                        mm["step"] = step
                        self.add_coverage(mm, coordinates[step], range_dict[date][num])

        return self.covjson
