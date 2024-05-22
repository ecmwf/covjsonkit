import json

import pandas as pd

from .encoder import Encoder


class VerticalProfile(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)

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
            coverage["ranges"][parameter] = {}
            coverage["ranges"][parameter]["type"] = "NdArray"
            coverage["ranges"][parameter]["dataType"] = "float"
            coverage["ranges"][parameter]["shape"] = [len(values[parameter])]
            coverage["ranges"][parameter]["axisNames"] = ["z"]
            coverage["ranges"][parameter]["values"] = values[parameter]  # [values[parameter]]

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
                    # "date": fc_time.values.astype("M8[ms]")
                    # .astype("O")
                    # .strftime("%m/%d/%Y"),
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
        coords["z"] = list(df["level"].unique())
        coords["t"] = list(df["date"].unique())

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
