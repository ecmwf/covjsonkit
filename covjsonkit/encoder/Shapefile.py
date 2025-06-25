import logging

from .encoder import Encoder


class Shapefile(Encoder):
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
        """
        Converts an xarray dataset into a MultiPoint CoverageJSON format.
        """

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "PointSeries"
        self.covjson["coverages"] = []

        if "latitude" in dataset.coords:
            x_coord = "latitude"
        elif "x" in dataset.coords:
            x_coord = "x"
        if "longitude" in dataset.coords:
            y_coord = "longitude"
        elif "y" in dataset.coords:
            y_coord = "y"
        if "levelist" in dataset.coords:
            z_coord = "levelist"

        # Add reference system
        self.add_reference(
            {
                "coordinates": [x_coord, y_coord, z_coord],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        for data_var in dataset.data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        # Prepare coordinates
        coords = {
            "composite": [],
            "dataType": "tuple",
            "t": [str(x) for x in dataset["datetimes"].values],
        }

        for point in dataset["points"].values:
            coords["composite"].append(
                [
                    float(dataset.isel(points=point).longitude.values),
                    float(dataset.isel(points=point).latitude.values),
                    float(dataset.isel(points=point).levelist.values),
                ]
            )

        for datetime in dataset["datetimes"].values:
            for num in dataset["number"].values:
                for step in dataset["steps"].values:
                    dv_dict = {}
                    mars_metadata = {metadata: dataset.attrs[metadata] for metadata in dataset.attrs}
                    mars_metadata["number"] = int(num)
                    mars_metadata["step"] = int(step)
                    mars_metadata["Forecast date"] = str(datetime)
                    for dv in dataset.data_vars:
                        dv_dict[dv] = dataset[dv].sel(number=num, steps=step, datetimes=datetime).values.tolist()

                    self.add_coverage(mars_metadata, coords, dv_dict)

        # Return the generated CoverageJSON
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
                            for k, v in range_dict.items():
                                if k == key:
                                    if s not in combined_dict[date][num][para]:
                                        combined_dict[date][num][para][s] = v
                                    else:
                                        # Cocatenate arrays
                                        combined_dict[date][num][para][s] += v

        levels = fields["levels"]
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

        return self.covjson
