import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from .encoder import Encoder


class VerticalProfile(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "VerticalProfile"
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
        for parameter in self.parameters:
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = [len(values[parameter])]
            coverage["ranges"][param]["axisNames"] = ["z"]
            coverage["ranges"][param]["values"] = values[parameter]  # [values[parameter]]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def from_xarray(self, dataset):
        for parameter in dataset.data_vars:
            if parameter == "Temperature":
                self.add_parameter("t")
            elif parameter == "Pressure":
                self.add_parameter("p")

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
            self.add_coverage(
                {
                    "number": num,
                    "type": "forecast",
                    "step": 0,
                },
                {
                    "x": list(dataset["x"].values),
                    "y": list(dataset["y"].values),
                    "z": list(dataset["z"].values),
                    "t": [str(x) for x in dataset["t"].values],
                },
                {
                    "t": list(dataset["Temperature"].sel(number=num).values[0][0][0]),
                    "p": dataset["Pressure"].sel(number=num).values[0][0][0],
                },
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
        fields["step"] = [0]
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

        for date in fields["dates"]:
            coordinates[date] = {}
            for level in fields["levels"]:
                for num in fields["number"]:
                    for para in fields["param"]:
                        for step in fields["step"]:
                            date_format = "%Y%m%dT%H%M%S"
                            new_date = pd.Timestamp(date).strftime(date_format)
                            start_time = datetime.strptime(new_date, date_format)
                            # add current date to list by converting it to iso format
                            stamp = start_time + timedelta(hours=int(step))
                            coordinates[date][step] = {
                                "x": [coords[date]["composite"][0][0]],
                                "y": [coords[date]["composite"][0][1]],
                                "z": list(levels),
                            }
                            coordinates[date][step]["t"] = [stamp.isoformat() + "Z"]
                            # coordinates[date]["t"].append(stamp.isoformat() + "Z")
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
            for num in fields["number"]:
                val_dict = {}
                for step in fields["step"]:
                    val_dict[step] = {}
                    for para in fields["param"]:
                        val_dict[step][para] = []
                        for level in fields["levels"]:
                            key = (date, level, num, para, step)
                            # for k, v in range_dict.items():
                            #    if k == key:
                            # val_dict[para].append(v[0])
                            val_dict[step][para].append(range_dict[key][0])
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["Forecast date"] = date
                    mm["step"] = step
                    # del mm["step"]
                    self.add_coverage(mm, coordinates[date][step], val_dict[step])

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson
