import json

import pandas as pd
from covjson_pydantic.coverage import Coverage

from .encoder import Encoder


class BoundingBox(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "MultiPoint"
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
        #self.pydantic_coverage.coverages.append(json.dumps(new_coverage))

    def add_domain(self, coverage, coords):
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = {}
        coverage["domain"]["axes"]["t"] = {}
        coverage["domain"]["axes"]["t"]["values"] = coords["t"]
        coverage["domain"]["axes"]["composite"] = {}
        coverage["domain"]["axes"]["composite"]["dataType"] = "tuple"
        coverage["domain"]["axes"]["composite"]["coordinates"] = self.covjson['referencing'][0]['coordinates'] #self.pydantic_coverage.referencing[0].coordinates
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
        """
        ancestors = []
        values = []
        for val in result.leaves:
            ancestors.append(val.get_ancestors())
            values.append(val.result)

        df_dict = {}
        # Create empty dataframe
        for feature in ancestors[0]:
            df_dict[feature.axis.name] = []

        # populate dataframe
        for ancestor in ancestors:
            for feature in ancestor:
                df_dict[feature.axis.name].append(feature.value)
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
        coords["t"] = [str(df["date"].unique()[0]) + "Z"]

        for param in params:
            df_param = df[df["param"] == param]
            range_dict[param] = df_param["values"].values.tolist()

        df_param = df[df["param"] == params[0]]
        lat = df_param["latitude"].values.tolist()
        long = df_param["longitude"].values.tolist()
        latlong = zip(lat, long)
        for lat, long in latlong:
            coords["composite"].append([lat, long])

        """
        coords  = {}
        coords['composite'] = []
        mars_metadata = {}
        range_dict = {}
        lat = 0
        param = 0
        number = 0

        self.func(result, lat, coords, mars_metadata, param, range_dict, number)

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

        for num in range_dict.keys():
            mars_metadata["number"] = num
            self.add_coverage(mars_metadata, coords, range_dict[num])

        #self.add_coverage(mars_metadata, coords, range_dict)
        return self.covjson


    def func(self, tree, lat, coords, mars_metadata, param, range_dict, number):
        if len(tree.children) != 0:
        # recurse while we are not a leaf
            for c in tree.children:
                if c.axis.name != "latitude" and c.axis.name != "longitude" and c.axis.name != "param" and c.axis.name != "date":
                    mars_metadata[c.axis.name] = c.values[0]
                if c.axis.name == "latitude":
                    lat = c.values[0]
                if c.axis.name == "param":
                    param = c.values
                    for key in range_dict:
                        for para in param:
                            range_dict[key][para] = []
                if c.axis.name == "date":
                    coords['t'] = [str(c.values[0]) + "Z"]
                if c.axis.name == "number":
                    number = c.values
                    for num in number:
                        range_dict[num] = {}

                self.func(c, lat, coords, mars_metadata, param, range_dict, number)
        else:
            vals = len(tree.values)
            intervals = int(len(tree.result)/len(number))
            for val in tree.values:
                coords['composite'].append([lat, val])
            for key in range_dict:
                for i, para in enumerate(param):
                    range_dict[key][para].extend(tree.result[((int(key)-1)*intervals)+((i*vals)):((int(key)-1)*intervals)+((i+1)*(vals))])

    """       
    def func(self, tree, lat, coords, mars_metadata, param, range_dict, number):
        if len(tree.children) != 0:
        # recurse while we are not a leaf
            for c in tree.children:
                if c.axis.name != "latitude" and c.axis.name != "longitude" and c.axis.name != "param" and c.axis.name != "date":
                    mars_metadata[c.axis.name] = c.values[0]
                if c.axis.name == "latitude":
                    lat = c.values[0]
                if c.axis.name == "param":
                    param = c.values
                    for para in param:
                        range_dict[para] = []
                if c.axis.name == "date":
                    coords['t'] = [str(c.values[0]) + "Z"]

                self.func(c, lat, coords, mars_metadata, param, range_dict)
        else:
            vals = len(tree.values)
            for val in tree.values:
                coords['composite'].append([lat, val])
            for i, para in enumerate(param):
                range_dict[para].extend(tree.result[(i)*vals:(i+1)*vals])
    """