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
        coverage["domain"]["axes"]["latitude"] = {}
        coverage["domain"]["axes"]["longitude"] = {}
        coverage["domain"]["axes"]["levelist"] = {}
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["latitude"]["values"] = coords["latitude"]
        coverage["domain"]["axes"]["longitude"]["values"] = coords["longitude"]
        coverage["domain"]["axes"]["levelist"]["values"] = coords["levelist"]
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

    def from_xarray(self, datasets):
        """
        Converts an xarray dataset or a list of xarray datasets into an OGC CoverageJSON
        coverageCollection of type PointSeries.

        Args:
            datasets (Union[xarray.Dataset, List[xarray.Dataset]]): An xarray dataset or a list of xarray datasets.

        Returns:
            dict: The CoverageJSON representation of the coverageCollection.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "PointSeries"
        self.covjson["coverages"] = []

        if "latitude" in datasets[0].coords:
            x_coord = "latitude"
        elif "x" in datasets[0].coords:
            x_coord = "x"
        if "longitude" in datasets[0].coords:
            y_coord = "longitude"
        elif "y" in datasets[0].coords:
            y_coord = "y"
        if "levelist" in datasets[0].coords:
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
                        "levelist": [float(x) for x in dataset["levelist"].values],
                        "t": [str(x) for x in dataset["t"].values],
                    },
                    dv_dict,
                )

        return self.covjson

    def from_polytope(self, result):
        """
        Converts a Polytope result into an OGC CoverageJSON coverageCollection of type PointSeries
        Args:
            result (dict): The Polytope result containing the data to be converted.
        Returns:
            dict: The CoverageJSON representation of the coverageCollection.
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
                "coordinates": ["latitude", "longitude", "levelist"],
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

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        points = len(coords[fields["dates"][0]]["composite"])

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
                    self.add_coverage(mm, coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)], val_dict)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson
