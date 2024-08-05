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
            coverage["ranges"][param]["values"] = [values[parameter][val][0] for val in values[parameter].keys()] #values[parameter]

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
        lat = 0
        param = 0
        number = [0]
        step = 0
        dates = [0]

        self.walk_tree(result, lat, coords, mars_metadata, param, range_dict, number, step, dates)
        # print(range_dict)

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        #val_dict = {}
        #for num in range_dict.keys():
        #    val_dict[num] = {}
        #    for para in range_dict[1].keys():
        #        val_dict[num][para] = []
        #    for para in range_dict[num].keys():
        #        # for step in range_dict[num][para].keys():
        #        for step in range_dict[num][para]:
        #            val_dict[num][para].extend(range_dict[num][para][step])
        #    mm = mars_metadata.copy()
        #    mm["number"] = num
        #    self.add_coverage(mm, coords, val_dict[num])

        for date in range_dict.keys():
            coordinates = {"x": [coords[date]['composite'][0][0]], "y": [coords[date]['composite'][0][1]], "z": 'sfc'}
            coordinates['t'] = []
            for num in range_dict[date].keys():
                for para in range_dict[date][num].keys():
                    for step in range_dict[date][num][para].keys():
                        date_format = "%Y%m%dT%H%M%S"
                        new_date = pd.Timestamp(date).strftime(date_format)
                        start_time = datetime.strptime(new_date, date_format)
                        # add current date to list by converting it to iso format
                        stamp = start_time + timedelta(hours=int(step))
                        coordinates["t"].append(stamp.isoformat() + "Z")
                    break

        for date in range_dict.keys():
            for num in range_dict[date].keys():
                self.add_coverage(mars_metadata, coordinates, range_dict[date][num])

        return self.covjson