from .encoder import Encoder


class Path(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "Trajectory"
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
                "coordinates": ["t", "x", "y", "z"],
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

        xyt = zip(dataset.t.values, dataset.x.values, dataset.y.values)
        for t, x, y in xyt:
            coords["composite"].append([t, x, y])

        self.add_coverage(mars_metadata, coords, range_dicts)
        return self.covjson

    def from_polytope(self, result):

        coords = {}
        # coords['composite'] = []
        mars_metadata = {}
        range_dict = {}
        lat = 0
        param = 0
        number = [0]
        step = 0
        dates = [0]

        self.walk_tree(result, lat, coords, mars_metadata, param, range_dict, number, step, dates)

        self.add_reference(
            {
                "coordinates": ["t", "x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        for date in range_dict.keys():
            new_coords = []
            for val in coords[date]["composite"]:
                val.insert(0, date)
                new_coords.append(val)
            coords[date]["composite"] = new_coords

            for num in range_dict[date].keys():
                val_dict = {}
                for step in range_dict[date][num][self.parameters[0]].keys():
                    val_dict[step] = {}
                for para in range_dict[date][num].keys():
                    for step in range_dict[date][num][para].keys():
                        val_dict[step][para] = range_dict[date][num][para][step]
                for step in val_dict.keys():
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["step"] = step
                    self.add_coverage(mm, coords[date], val_dict[step])

        # self.add_coverage(mars_metadata, coords, range_dict)
        return self.covjson
