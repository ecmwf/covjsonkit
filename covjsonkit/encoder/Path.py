import logging

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
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["step"] = 0
        fields["dates"] = []
        fields["levels"] = [0]
        fields["s"] = []
        fields["l"] = []

        self.walk_tree(result, fields, coords, mars_metadata, range_dict)

        if len(fields["l"]) == 0:
            fields["l"] = [0]

        logging.debug("The values returned from walking tree: %s", range_dict)  # noqa: E501
        logging.debug("The coordinates returned from walking tree: %s", coords)  # noqa: E501
        logging.debug("The fields: %s", fields)

        self.add_reference(
            {
                "coordinates": ["t", "x", "y", "z"],
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
            for level in fields["l"]:
                for num in fields["number"]:
                    if num not in combined_dict[date]:
                        combined_dict[date][num] = {}
                    for para in fields["param"]:
                        if para not in combined_dict[date][num]:
                            combined_dict[date][num][para] = {}
                        # for s, value in range_dict[date][level][num][para].items():
                        for s in fields["s"]:
                            key = (date, level, num, para, s)
                            # for k, v in range_dict.items():
                            # if k == key:
                            if s not in combined_dict[date][num][para]:
                                if key in range_dict:
                                    combined_dict[date][num][para][s] = range_dict[key]
                                # combined_dict[date][num][para][s] = range_dict[key]
                            else:
                                # Cocatenate arrays
                                if key in range_dict:
                                    combined_dict[date][num][para][s] += range_dict[key]
                        # for s in fields["s"]:

        logging.debug("The values returned from combined dicts: %s", combined_dict)  # noqa: E501

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No parameters were returned, date requested may be out of range")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        for date in coords.keys():
            coord = coords[date]["composite"]
            coords[date]["composite"] = []
            for level in levels:
                start = 0
                for i, s in enumerate(fields["s"]):
                    end = start + len(coord) / len(fields["s"])
                    for cor in coord[int(start) : int(end)]:
                        if len(fields["l"]) == 1:
                            coords[date]["composite"].append([s, cor[0], cor[1], fields["l"][0]])
                        elif len(fields["l"]) == 0:
                            coords[date]["composite"].append([s, cor[0], cor[1], level])
                        else:
                            coords[date]["composite"].append([s, cor[0], cor[1], fields["l"][i]])
                    start = end
        logging.debug("The coordinates returned from walking tree: %s", coords)  # noqa: E501

        for date in combined_dict.keys():
            for num in combined_dict[date].keys():
                val_dict = {}
                # for step in combined_dict[date][num][self.parameters[0]].keys():
                #    val_dict[step] = {}
                for para in combined_dict[date][num].keys():
                    if para not in val_dict:
                        val_dict[para] = []
                    for step in combined_dict[date][num][para].keys():
                        val_dict[para].extend(combined_dict[date][num][para][step])
                # for step in val_dict.keys():
                mm = mars_metadata.copy()
                mm["number"] = num
                # mm["step"] = step
                # temp = []
                # for coord in coords[date]["composite"]:
                #    temp.append([step] + coord)
                # coords[date]["composite"] = temp
                mm["Forecast date"] = date
                self.add_coverage(mm, coords[date], val_dict)

        return self.covjson
