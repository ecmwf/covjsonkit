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
        coords = {}
        # coords['x'] = []
        # coords['y'] = []
        # coords['z'] = []
        # coords['t'] = []
        mars_metadata = {}
        range_dict = {}
        lat = 0
        param = 0
        # number = 0
        step = 0
        long = 0
        levels = 0
        dates = 0

        self.func(
            result,
            lat,
            long,
            coords,
            mars_metadata,
            param,
            range_dict,
            step,
            levels,
            dates,
        )

        self.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        for date in range_dict.keys():
            for param in range_dict[date].keys():
                # self.coord_length = len(range_dict[date][param])
                self.add_parameter(param)
            break

        for date in range_dict.keys():
            self.add_coverage(mars_metadata, coords[date], range_dict[date])

        # return json.loads(self.get_json())
        return self.covjson

    def func(
        self,
        tree,
        lat,
        long,
        coords,
        mars_metadata,
        param,
        range_dict,
        step,
        levels,
        dates,
    ):
        if len(tree.children) != 0:
            # recurse while we are not a leaf
            for c in tree.children:
                if (
                    c.axis.name != "latitude"
                    and c.axis.name != "longitude"
                    and c.axis.name != "param"
                    and c.axis.name != "step"
                    and c.axis.name != "date"
                    and c.axis.name != "levelist"
                ):
                    mars_metadata[c.axis.name] = c.values[0]
                if c.axis.name == "latitude":
                    lat = c.values[0]
                if c.axis.name == "param":
                    param = c.values
                    for date in dates:
                        for para in param:
                            range_dict[date][para] = []
                if c.axis.name == "date":
                    dates = [str(date) + "Z" for date in c.values]
                    for date in dates:
                        coords[date] = {}
                        range_dict[date] = {}
                    mars_metadata[c.axis.name] = str(c.values[0]) + "Z"
                if c.axis.name == "levelist":
                    levels = c.values

                self.func(
                    c,
                    lat,
                    long,
                    coords,
                    mars_metadata,
                    param,
                    range_dict,
                    step,
                    levels,
                    dates,
                )
        else:
            tree.values = [float(val) for val in tree.values]
            tree.result = [float(val) for val in tree.result]
            # num_intervals = int(len(tree.result)/len(number))
            # para_intervals = int(num_intervals/len(param))
            len_paras = len(param) * len(levels)
            len_levels = len(param)

            for date in dates:

                coords[date]["x"] = [lat]
                coords[date]["y"] = [long]
                coords[date]["z"] = list(levels)
                coords[date]["t"] = date

            for i, date in enumerate(dates):
                for j, level in enumerate(list(levels)):
                    for k, para in enumerate(param):
                        range_dict[date][para].append(tree.result[i * len_paras + j * len_levels + k])
