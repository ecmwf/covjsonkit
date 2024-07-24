import json
import orjson

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

        coords  = {}
        #coords['composite'] = []
        mars_metadata = {}
        range_dict = {0:{}}
        lat = 0
        param = 0
        number = [0]
        step = 0
        dates = [0]

        self.func(result, lat, coords, mars_metadata, param, range_dict, number, step, dates)
        print(coords)
        print(range_dict)

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        
        #for para in range_dict[1][dates[0]].keys():
        #    self.add_parameter(para)               


        for num in range_dict.keys():
            for date in range_dict[num].keys():
                val_dict = {}
                for step in range_dict[0][date][self.parameters[0]].keys():
                    val_dict[step] = {}
                for para in range_dict[num][date].keys():
                    for step in range_dict[num][date][para].keys():
                        val_dict[step][para] = range_dict[num][date][para][step]
                for step in val_dict.keys():
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm['step'] = step
                    self.add_coverage(mm, coords[date], val_dict[step])

        #self.add_coverage(mars_metadata, coords, range_dict)
        #return self.covjson
        #with open('data.json', 'w') as f:
        #    json.dump(self.covjson, f)
        return self.covjson


    def func(self, tree, lat, coords, mars_metadata, param, range_dict, number, step, dates):
        if len(tree.children) != 0:
        # recurse while we are not a leaf
            for c in tree.children:
                print(c.axis.name)
                if c.axis.name != "latitude" and c.axis.name != "longitude" and c.axis.name != "param" and c.axis.name != "date":
                    mars_metadata[c.axis.name] = c.values[0]
                if c.axis.name == "latitude":
                    lat = c.values[0]
                if c.axis.name == "param":
                    param = c.values
                    print(range_dict)
                    for num in range_dict.keys():
                        for date in dates:
                            for para in param:
                                range_dict[num][date][para] = {}
                                self.add_parameter(para)
                if c.axis.name == "date":
                    dates = [str(date)+ "Z" for date in c.values]
                    for date in dates:
                        coords[date] = {}
                        coords[date]['composite'] = []
                        coords[date]['t'] = [date]
                    for num in range_dict.keys():
                        for date in dates:
                            range_dict[num][date] = {}
                if c.axis.name == "number":
                    number = c.values
                    for num in number:
                        range_dict[num] = {}
                if c.axis.name == "step":
                    step = c.values
                    for num in number:
                        for date in dates:
                            for para in param:
                                for s in step:
                                    range_dict[num][date][para][s] = []

                self.func(c, lat, coords, mars_metadata, param, range_dict, number, step, dates)
        else:
            vals = len(tree.values)
            tree.values = [float(val) for val in tree.values]
            tree.result = [float(val) for val in tree.result]
            num_intervals = int(len(tree.result)/len(number))
            #para_intervals = int(num_intervals/len(param))
            num_len = len(tree.result)/len(number)
            para_len = len(tree.result)/len(param)
            step_len = para_len/len(step)

            for date in dates:
                for val in tree.values:
                    coords[date]['composite'].append([lat, val])
            
            #print(lat)
            #print(number)
            #print(dates)
            #print(param)
            #print(step)
            #print(tree.values)
            print(para_len)
            print(step_len)
            print(tree.result)

            for i, para in enumerate(param):
                for j, s in enumerate(step):
                    range_dict[number[0]][dates[0]][para][s].extend(tree.result[int(i*para_len)+ int((j)*step_len): int(i*para_len) + int((j+1)*step_len)])


            #for i, num in enumerate(range_dict):
            #    for j, date in enumerate(dates):
            #        for k, para in enumerate(param):
            #            for s in step:
            #                start = ((int(num)-1)*num_intervals)+(vals*int(s))+((i*para_intervals))
            #                end = ((int(num)-1)*num_intervals)+((vals)*int(s+1))+((i)*(para_intervals))
            #                range_dict[num][date][para][s].extend(tree.result[start:end])
            #        for s in range_dict[num][para]:
            #            start = ((int(num)-1)*num_intervals)+(vals*int(s))+((i*para_intervals))
            #            end = ((int(num)-1)*num_intervals)+((vals)*int(s+1))+((i)*(para_intervals))
            #            range_dict[num][para][s].extend(tree.result[start:end])