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
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        points = len(coords[fields["dates"][0]]["composite"])

        for date in fields["dates"]:
            coordinates[date] = []
            for i, point in enumerate(range(points)):
                coordinates[date].append(
                    {
                        "x": [coords[date]["composite"][i][0]],
                        "y": [coords[date]["composite"][i][1]],
                        "z": [levels[0]],
                    }
                )
                # coordinates[date] = {
                #    "x": [coords[date]["composite"][0][0]],
                #    "y": [coords[date]["composite"][0][1]],
                #    "z": [levels[0]],
                # }
                coordinates[date][i]["t"] = []
                for level in fields["levels"]:
                    for num in fields["number"]:
                        for para in fields["param"]:
                            for step in fields["step"]:
                                date_format = "%Y%m%dT%H%M%S"
                                new_date = pd.Timestamp(date).strftime(date_format)
                                start_time = datetime.strptime(new_date, date_format)
                                # add current date to list by converting it to iso format
                                try:
                                    int(step)
                                except ValueError:
                                    step = step[0]
                                stamp = start_time + timedelta(hours=int(step))
                                coordinates[date][i]["t"].append(stamp.isoformat() + "Z")
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

        for i, point in enumerate(range(points)):
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
                                val_dict[para].append(range_dict[key][i])
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        mm["Forecast date"] = date
                        del mm["step"]
                        self.add_coverage(mm, coordinates[date][i], val_dict)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

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

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)  # noqa: E501
        self.walk_tree_step(result, fields, coords, mars_metadata, range_dict)
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
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        points = len(coords[fields["dates"][0]]["composite"])

        for step in fields["step"]:
            coordinates[fields["dates"][0]] = []
            for i, point in enumerate(range(points)):
                coordinates[fields["dates"][0]].append(
                    {
                        "x": [coords[fields["dates"][0]]["composite"][i][0]],
                        "y": [coords[fields["dates"][0]]["composite"][i][1]],
                        "z": [levels[0]],
                    }
                )
                coordinates[fields["dates"][0]][i]["t"] = []
                for level in fields["levels"]:
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
                                    coordinates[fields["dates"][0]][i]["t"].append(str(datetime).split("+")[0] + "Z")
                            break
                        break
                    break

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        for i, point in enumerate(range(points)):
            for level in fields["levels"]:
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
                    # del mm["step"]
                    print(val_dict)
                    self.add_coverage(mm, coordinates[fields["dates"][0]][i], val_dict)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson
