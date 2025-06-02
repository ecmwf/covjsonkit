import gc
import logging
import time

import pandas as pd

from .encoder import Encoder


class Wkt(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "MultiPoint"
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
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["t"]["values"] = coords["t"]
        coverage["domain"]["axes"]["composite"] = {}
        coverage["domain"]["axes"]["composite"]["dataType"] = "tuple"
        coverage["domain"]["axes"]["composite"]["coordinates"] = self.covjson["referencing"][0][
            "coordinates"
        ]  # self.pydantic_coverage.referencing[0].coordinates
        coverage["domain"]["axes"]["composite"]["values"] = coords["composite"]

    def add_range(self, coverage, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = [len(values[parameter])]
            coverage["ranges"][param]["axisNames"] = [str(param)]
            coverage["ranges"][param]["values"] = values[parameter]  # [values[parameter]]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def from_xarray(self, dataset):
        range_dicts = {}

        for data_var in dataset.data_vars:
            self.add_parameter(data_var)
            range_dicts[data_var] = dataset[data_var].values.tolist()

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        mars_metadata = {}

        for metadata in dataset.attrs:
            mars_metadata[metadata] = dataset.attrs[metadata]

        coords = {}
        coords["composite"] = []
        coords["t"] = dataset.attrs["date"]

        xy = zip(dataset.x.values, dataset.y.values)
        for x, y in xy:
            coords["composite"].append([x, y])

        self.add_coverage(mars_metadata, coords, range_dicts)
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
                "coordinates": ["x", "y", "z"],
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
                            #    if k == key:
                            if s not in combined_dict[date][num][para]:
                                combined_dict[date][num][para][s] = range_dict[key]
                            else:
                                # Cocatenate arrays
                                combined_dict[date][num][para][s] += range_dict[key]

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        for date in coords.keys():
            coord = coords[date]["composite"]
            coords[date]["composite"] = []
            for level in levels:
                for cor in coord:
                    coords[date]["composite"].append([cor[0], cor[1], level])

        for date in combined_dict.keys():
            for num in combined_dict[date].keys():
                val_dict = {}
                for step in combined_dict[date][num][self.parameters[0]].keys():
                    val_dict[step] = {}
                for para in combined_dict[date][num].keys():
                    for step in combined_dict[date][num][para].keys():
                        val_dict[step][para] = combined_dict[date][num][para][step]
                for step in val_dict.keys():
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["step"] = step
                    mm["Forecast date"] = date
                    self.add_coverage(mm, coords[date], val_dict[step])

        # self.add_coverage(mars_metadata, coords, range_dict)
        # return self.covjson
        # with open('data.json', 'w') as f:
        #    json.dump(self.covjson, f)
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

        # print(fields)
        # print("********")
        # print(coords)
        # print("********")
        # print(coordinates)
        # print("********")
        # print(range_dict)

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        for i, t in enumerate(fields["times"]):
            for level in fields["levels"]:
                for num in fields["number"]:
                    val_dict = {}
                    for date in fields["dates"]:
                        for para in fields["param"]:
                            val_dict[para] = []
                            key = (date, level, num, para)
                            vals = []
                            for val in range_dict[key]:
                                vals.append(val[i])
                            val_dict[para].extend(vals)
                        mm = mars_metadata.copy()
                        mm["number"] = num
                        # mm["Forecast date"] = date
                        datetime = pd.Timestamp(date) + t
                        self.add_coverage(mm, coordinates[str(datetime).split("+")[0] + "Z"], val_dict)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        del coordinates
        gc.collect()

        return self.covjson
