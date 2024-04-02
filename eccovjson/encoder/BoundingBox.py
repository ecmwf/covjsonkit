import pandas as pd

from .encoder import Encoder


class BoundingBox(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "MultiPoint"

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
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["t"]["values"] = coords["t"]
        coverage["domain"]["axes"]["composite"] = {}
        coverage["domain"]["axes"]["composite"]["dataType"] = "tuple"
        coverage["domain"]["axes"]["composite"]["coordinates"] = self.covjson["referencing"][0]["coordinates"]
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
        mars_metadata["domain"] = df["domain"].unique()[0]
        mars_metadata["stream"] = df["stream"].unique()[0]

        range_dict = {}
        coords = {}
        coords["composite"] = []
        coords["t"] = df["date"].unique()[0]

        for param in params:
            df_param = df[df["param"] == param]
            range_dict[param] = df_param["values"].values.tolist()

        df_param = df[df["param"] == params[0]]
        for row in df_param.iterrows():
            coords["composite"].append([row[1]["latitude"], row[1]["longitude"]])

        self.add_coverage(mars_metadata, coords, range_dict)
        return self.covjson
