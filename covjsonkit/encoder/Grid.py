import logging
import time

import pandas as pd

from .encoder import Encoder


class Grid(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "Grid"
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
        # self.pydantic_coverage.coverages.append(json.dumps(new_coverage))

    def add_domain(self, coverage, coords):
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = {}
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["t"]["values"] = coords["t"]
        coverage["domain"]["axes"]["latitude"] = {}
        coverage["domain"]["axes"]["latitude"]["values"] = coords["latitude"]
        coverage["domain"]["axes"]["longitude"] = {}
        coverage["domain"]["axes"]["longitude"]["values"] = coords["longitude"]
        coverage["domain"]["axes"]["levelist"] = {}
        coverage["domain"]["axes"]["levelist"]["values"] = coords["levelist"]

    def add_range(self, coverage, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = self.shp
            coverage["ranges"][param]["axisNames"] = ["t", "levelist", "latitude", "longitude"]
            coverage["ranges"][param]["values"] = values[parameter]  # [values[parameter]]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def add_if_not_close(self, my_list, number, threshold=0.01):
        if all(abs(number - x) > threshold for x in my_list):
            my_list.append(number)

    def from_xarray(self, dataset):
        """
        Converts an xarray dataset into a grid CoverageJSON format.
        """

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "PointSeries"
        self.covjson["coverages"] = []

        if "latitude" in dataset.coords:
            x_coord = "latitude"
        elif "x" in dataset.coords:
            x_coord = "x"
        if "longitude" in dataset.coords:
            y_coord = "longitude"
        elif "y" in dataset.coords:
            y_coord = "y"
        if "levelist" in dataset.coords:
            z_coord = "levelist"

        # Add reference system
        self.add_reference(
            {
                "coordinates": [x_coord, y_coord, z_coord],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        for data_var in dataset.data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        # Prepare coordinates
        coords = {
            "t": [str(x) for x in dataset["steps"].values],
            "latitude": dataset["latitude"].values.tolist(),
            "longitude": dataset["longitude"].values.tolist(),
            "levelist": dataset["levelist"].values.tolist(),
        }

        self.shp = [len(coords["t"]), len(coords["levelist"]), len(coords["latitude"]), len(coords["longitude"])]

        for datetime in dataset["datetimes"].values:
            for num in dataset["number"].values:
                for step in dataset["steps"].values:
                    dv_dict = {}
                    mars_metadata = {metadata: dataset.attrs[metadata] for metadata in dataset.attrs}
                    mars_metadata["number"] = int(num)
                    mars_metadata["step"] = int(step)
                    mars_metadata["Forecast date"] = str(datetime)
                    for dv in dataset.data_vars:
                        nested_list = dataset[dv].sel(datetimes=datetime, number=num, steps=step).values.tolist()
                        print(nested_list)
                        flattened_list = [item for sublist in nested_list for item in sublist]
                        flattened_list = [item for sublist in flattened_list for item in sublist]
                        print(flattened_list)
                        dv_dict[dv] = flattened_list

                    self.add_coverage(mars_metadata, coords, dv_dict)

        # Return the generated CoverageJSON
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

        self.walk_tree(result, fields, coords, mars_metadata, range_dict)

        logging.debug("The values returned from walking tree: %s", range_dict)  # noqa: E501
        logging.debug("The coordinates returned from walking tree: %s", coords)  # noqa: E501

        self.add_reference(
            {
                "coordinates": ["latitude", "longitude", "levelist"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        combined_dict = {}

        for date in fields["dates"]:
            if date not in combined_dict:
                combined_dict[date] = {}
            for level in fields["levels"]:
                for num in fields["number"]:
                    if num not in combined_dict[date]:
                        combined_dict[date][num] = {}
                    for para in fields["param"]:
                        if para not in combined_dict[date][num]:
                            combined_dict[date][num][para] = {}
                        # for s, value in range_dict[date][level][num][para].items():
                        for s in fields["step"]:
                            key = (date, level, num, para, s)
                            # for k, v in range_dict.items():
                            # if k == key:
                            if s not in combined_dict[date][num][para]:
                                combined_dict[date][num][para][s] = range_dict[key]
                            else:
                                # Cocatenate arrays
                                combined_dict[date][num][para][s] += range_dict[key]

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        logging.debug("The fields retrieved were: %s", fields)  # noqa: E501
        logging.debug("The range_dict created was: %s", range_dict)  # noqa: E501

        coordinates = {}
        coordinates["t"] = list(fields["step"])

        for date in coords.keys():
            coordinates[date] = {}
            coordinates[date]["t"] = list(fields["step"])
            coordinates[date]["levelist"] = list(fields["levels"])
            coordinates[date]["latitude"] = []
            coordinates[date]["longitude"] = []
            for cor in coords[date]["composite"]:
                self.add_if_not_close(coordinates[date]["latitude"], cor[0])
                self.add_if_not_close(coordinates[date]["longitude"], cor[1])
            coordinates[date]["latitude"] = list(coordinates[date]["latitude"])
            coordinates[date]["longitude"] = list(coordinates[date]["longitude"])

        self.shp = [
            len(coordinates[fields["dates"][0]]["t"]),
            len(coordinates[fields["dates"][0]]["levelist"]),
            len(coordinates[fields["dates"][0]]["latitude"]),
            len(coordinates[fields["dates"][0]]["longitude"]),
        ]

        for date in combined_dict.keys():
            for num in combined_dict[date].keys():
                val_dict = {}
                for para in combined_dict[date][num].keys():
                    val_dict[para] = []
                    for step in combined_dict[date][num][para].keys():
                        val_dict[para].extend(combined_dict[date][num][para][step])
                mm = mars_metadata.copy()
                mm["number"] = num
                mm["step"] = step
                mm["Forecast date"] = date
                self.add_coverage(mm, coordinates[date], val_dict)

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

        # points = len(coords[fields["dates"][0]]["composite"])

        coordinates = {}

        for date in coords.keys():
            for dt in coords[date]["t"]:
                coordinates[dt] = {}
                coordinates[dt]["composite"] = []
                coordinates[dt]["t"] = [dt]
                coord = coords[date]["composite"]
                for level in levels:
                    for cor in coord:
                        coordinates[dt]["composite"].append([cor[0], cor[1], level])

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        val_dict = {}
        for i, t in enumerate(fields["times"]):
            val_dict[t] = {}
            for level in fields["levels"]:
                for num in fields["number"]:
                    for date in fields["dates"]:
                        for para in fields["param"]:
                            val_dict[t][para] = []
                            key = (date, level, num, para)
                            vals = int(len(range_dict[key]) / len(fields["times"]))
                            # for val in range_dict[key]:
                            #    vals.append(val[i])
                            val_dict[t][para].extend(range_dict[key][i * vals : (i + 1) * vals])
                        for para in fields["param"]:
                            val_dict[t][para] = [item for sublist in val_dict[t][para] for item in sublist]
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        # mm["Forecast date"] = date
                        datetime = pd.Timestamp(date) + t
                        self.add_coverage(mm, coordinates[str(datetime).split("+")[0] + "Z"], val_dict[t])

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson
