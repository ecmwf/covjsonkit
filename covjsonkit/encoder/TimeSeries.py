import json
from datetime import datetime, timedelta

import pandas as pd
from covjson_pydantic.coverage import Coverage

from .encoder import Encoder


class TimeSeries(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "PointSeries"
        self.covjson['coverages'] = []

    def add_coverage(self, mars_metadata, coords, values):
        new_coverage = {}
        new_coverage["mars:metadata"] = {}
        new_coverage["type"] = "Coverage"
        new_coverage["domain"] = {}
        new_coverage["ranges"] = {}
        self.add_mars_metadata(new_coverage, mars_metadata)
        self.add_domain(new_coverage, coords)
        self.add_range(new_coverage, values)
        self.covjson['coverages'].append(new_coverage)
        #cov = Coverage.model_validate_json(json.dumps(new_coverage))
        #self.pydantic_coverage.coverages.append(cov)

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
            coverage["ranges"][param]["values"] = values[parameter]

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
        """
        ancestors = [val.get_ancestors() for val in result.leaves]
        values = [val.result for val in result.leaves]

        columns = []
        df_dict = {}
        # Create empty dataframe
        for feature in ancestors[0]:
            columns.append(str(feature).split("=")[0])
            df_dict[str(feature).split("=")[0]] = []

        # populate dataframe
        for ancestor in ancestors:
            for feature in ancestor:
                df_dict[str(feature).split("=")[0]].append(str(feature).split("=")[1])
        values = [val.result for val in result.leaves]
        df_dict["values"] = values
        df = pd.DataFrame(df_dict)

        params = df["param"].unique()

        for param in params:
            self.add_parameter(param)

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        steps = df["step"].unique()

        mars_metadata = {}
        mars_metadata["class"] = df["class"].unique()[0]
        mars_metadata["expver"] = df["expver"].unique()[0]
        mars_metadata["levtype"] = df["levtype"].unique()[0]
        mars_metadata["type"] = df["type"].unique()[0]
        mars_metadata["date"] = df["date"].unique()[0]
        mars_metadata["domain"] = df["domain"].unique()[0]
        mars_metadata["stream"] = df["stream"].unique()[0]

        coords = {}
        coords["x"] = list(df["latitude"].unique())
        coords["y"] = list(df["longitude"].unique())
        coords["z"] = ["sfc"]
        coords["t"] = []

        # convert step into datetime
        date_format = "%Y%m%dT%H%M%S"
        date = pd.Timestamp(mars_metadata["date"]).strftime(date_format)
        start_time = datetime.strptime(date, date_format)
        for step in steps:
            # add current date to list by converting it to iso format
            stamp = start_time + timedelta(hours=int(step))
            coords["t"].append(stamp.isoformat() + "Z")
            # increment start date by timedelta

        if "number" not in df.columns:
            new_metadata = mars_metadata.copy()
            range_dict = {}
            for param in params:
                df_param = df[df["param"] == param]
                range_dict[param] = df_param["values"].values.tolist()
            self.add_coverage(new_metadata, coords, range_dict)
        else:
            for number in df["number"].unique():
                new_metadata = mars_metadata.copy()
                new_metadata["number"] = number
                df_number = df[df["number"] == number]
                range_dict = {}
                for param in params:
                    df_param = df_number[df_number["param"] == param]
                    range_dict[param] = df_param["values"].values.tolist()
                self.add_coverage(new_metadata, coords, range_dict)

        return json.loads(self.get_json())
        """

        coords  = {}
        coords['x'] = []
        coords['y'] = []
        coords['z'] = []
        coords['t'] = []
        mars_metadata = {}
        range_dict = {}
        lat = 0
        param = 0
        number = 0
        step = 0
        long = 0

        self.func(result, lat, long, coords, mars_metadata, param, range_dict, number, step)
        #print(range_dict)

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        
        for param in range_dict[1].keys():
            self.add_parameter(param)

        for step in range_dict[1][self.parameters[0]].keys():
            date_format = "%Y%m%dT%H%M%S"
            date = pd.Timestamp(coords["t"][0]).strftime(date_format)
            start_time = datetime.strptime(date, date_format)
            # add current date to list by converting it to iso format
            stamp = start_time + timedelta(hours=int(step))
            coords["t"].append(stamp.isoformat() + "Z")

        val_dict = {}
        for num in range_dict.keys():
            val_dict[num] = {}
            for para in range_dict[1].keys():
                val_dict[num][para] = []
            for para in range_dict[num].keys():
                #for step in range_dict[num][para].keys():
                for step in range_dict[num][para]:
                    val_dict[num][para].extend(range_dict[num][para][step])
            mm = mars_metadata.copy()
            mm["number"] = num
            self.add_coverage(mm, coords, val_dict[num])
        
        return self.covjson  

    def func(self, tree, lat, long, coords, mars_metadata, param, range_dict, number, step):
        if len(tree.children) != 0:
        # recurse while we are not a leaf
            for c in tree.children:
                if c.axis.name != "latitude" and c.axis.name != "longitude" and c.axis.name != "param" and c.axis.name != "step" and c.axis.name != "date":
                    mars_metadata[c.axis.name] = c.values[0]
                if c.axis.name == "latitude":
                    lat = c.values[0]
                if c.axis.name == "param":
                    param = c.values
                    for num in range_dict:
                        for para in param:
                            range_dict[num][para] = {}
                if c.axis.name == "date":
                    coords['t'] = [str(c.values[0]) + "Z"]
                    mars_metadata[c.axis.name] = str(c.values[0]) + "Z"
                if c.axis.name == "number":
                    number = c.values
                    for num in number:
                        range_dict[num] = {}
                if c.axis.name == "step":
                    step = c.values
                    for num in number:
                        for para in param:
                            for s in step:
                                range_dict[num][para][s] = []

                self.func(c, lat, long, coords, mars_metadata, param, range_dict, number, step)
        else:
            vals = len(tree.values)
            tree.values = [float(val) for val in tree.values]
            tree.result = [float(val) for val in tree.result]
            num_intervals = int(len(tree.result)/len(number))
            para_intervals = int(num_intervals/len(param))

            coords['x'] = [lat]
            coords['y'] = [long]
            coords['z'] = ['sfc']

            for num in range_dict:
                for i, para in enumerate(range_dict[num]):
                    for s in range_dict[num][para]:
                        start = ((int(num)-1)*num_intervals)+(vals*int(s))+((i*para_intervals))
                        end = ((int(num)-1)*num_intervals)+((vals)*int(s+1))+((i)*(para_intervals))
                        range_dict[num][para][s].extend(tree.result[start:end])