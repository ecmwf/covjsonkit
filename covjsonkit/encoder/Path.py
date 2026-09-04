import logging

from .encoder import Encoder, normalize_step_value


class Path(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "Trajectory"
        self.covjson["coverages"] = []

    def add_coverage(self, mars_metadata, coords, values, include_z=False):
        new_coverage = {}
        new_coverage["mars:metadata"] = {}
        new_coverage["type"] = "Coverage"
        new_coverage["domain"] = {}
        new_coverage["ranges"] = {}
        self.add_mars_metadata(new_coverage, mars_metadata)
        self.add_domain(new_coverage, coords, include_z)
        self.add_range(new_coverage, values)
        self.covjson["coverages"].append(new_coverage)

    def add_domain(self, coverage, coords, include_z=False):
        # OGC-compliant Trajectory domain: a single ``composite`` axis whose
        # tuples are ordered [t, x, y] (== [time, longitude, latitude]) at the
        # surface, or [t, x, y, z] (== [time, longitude, latitude, level]) when a
        # vertical axis is present. Time is embedded in the composite tuple (a
        # Trajectory has no separate ``t`` axis).
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = {}
        coverage["domain"]["axes"]["composite"] = {
            "dataType": "tuple",
            "coordinates": ["t", "x", "y", "z"] if include_z else ["t", "x", "y"],
            "values": coords["composite"],
        }

    def add_range(self, coverage, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = [len(values[parameter])]
            coverage["ranges"][param]["axisNames"] = ["composite"]
            coverage["ranges"][param]["values"] = values[parameter]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def _set_references(self, include_z):
        # Trajectory referencing is split into GeographicCRS (x, y), a TemporalRS
        # (t) and an optional VerticalCRS (z) so each coordinate carries the
        # correct coordinate reference system.
        refs = [
            {
                "coordinates": ["x", "y"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            },
            {
                "coordinates": ["t"],
                "system": {"type": "TemporalRS", "calendar": "Gregorian"},
            },
        ]
        if include_z:
            refs.append(
                {
                    "coordinates": ["z"],
                    "system": {"type": "VerticalCRS"},
                }
            )
        self.covjson["referencing"] = refs

    def from_xarray(self, dataset):
        """
        Converts an xarray dataset into a Trajectory CoverageJSON collection.
        """

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "Trajectory"
        self.covjson["coverages"] = []

        include_z = "levelist" in dataset.coords

        # Add reference system (split GeographicCRS / TemporalRS / VerticalCRS)
        self._set_references(include_z)

        for data_var in dataset.data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        # Prepare coordinates: composite tuples ordered [t, x, y(, z)].
        coords = {"composite": []}

        for point in dataset["points"].values:
            t = float(dataset.isel(points=point).time.values)
            lon = float(dataset.isel(points=point).longitude.values)
            lat = float(dataset.isel(points=point).latitude.values)
            if include_z:
                level = float(dataset.isel(points=point).levelist.values)
                coords["composite"].append([t, lon, lat, level])
            else:
                coords["composite"].append([t, lon, lat])

        for datetime in dataset["datetimes"].values:
            for num in dataset["number"].values:
                for step in dataset["steps"].values:
                    dv_dict = {}
                    mars_metadata = {metadata: dataset.attrs[metadata] for metadata in dataset.attrs}
                    mars_metadata["number"] = int(num)
                    mars_metadata["step"] = normalize_step_value(step)
                    mars_metadata["Forecast date"] = str(datetime)
                    for dv in dataset.data_vars:
                        dv_dict[dv] = dataset[dv].sel(number=num, steps=step, datetimes=datetime).values.tolist()

                    self.add_coverage(mars_metadata, coords, dv_dict, include_z)

        # Return the generated CoverageJSON
        return self.covjson

    def from_polytope(self, result, date_key: str = "date") -> dict:
        """Encode a polytope ``TensorIndexTree`` result into a Trajectory (Path) CoverageJSON collection."""
        coords = {}
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["step"] = [0]
        fields["dates"] = []
        fields["levels"] = [0]
        fields["s"] = []
        fields["l"] = []
        fields["has_level_axis"] = False

        self.walk_tree(result, fields, coords, mars_metadata, range_dict, date_key=date_key)

        if len(fields["l"]) == 0:
            fields["l"] = [0]

        if len(fields["s"]) == 0:
            fields["s"] = [0]

        logging.debug("The values returned from walking tree: %s", range_dict)  # noqa: E501
        logging.debug("The coordinates returned from walking tree: %s", coords)  # noqa: E501
        logging.debug("The fields: %s", fields)

        include_z = fields["has_level_axis"]
        self._set_references(include_z)

        # Build trajectory composite tuples ordered [t, x, y] (== [step, lon,
        # lat]) at the surface, or [t, x, y, z] (== [step, lon, lat, level]) when
        # a vertical axis is present. walk_tree stores each spatial point as
        # [lat, lon].
        for date in coords.keys():
            coord = coords[date]["composite"]
            coords[date]["composite"] = []
            start = 0
            for level in set(fields["l"]):
                for s in set(fields["s"]):
                    if (date, level, fields["number"][0], fields["param"][0], s) in range_dict:
                        cor_len = len(range_dict[(date, level, fields["number"][0], fields["param"][0], s)])
                        end = start + cor_len
                        # Normalize the step value before it is placed into the
                        # composite (t) coordinate so that timedelta/Timedelta
                        # values are JSON serialisable. The original ``s`` is kept
                        # for the range_dict lookups above.
                        s_norm = normalize_step_value(s)
                        if include_z:
                            if len(fields["levels"]) != 1:
                                for lev in fields["levels"]:
                                    for cor in coord[int(start) : int(end)]:
                                        coords[date]["composite"].append([s_norm, cor[1], cor[0], lev])
                            else:
                                for cor in coord[int(start) : int(end)]:
                                    coords[date]["composite"].append([s_norm, cor[1], cor[0], level])
                        else:
                            for cor in coord[int(start) : int(end)]:
                                coords[date]["composite"].append([s_norm, cor[1], cor[0]])
                        start = end
        logging.debug("The coordinates returned from walking tree: %s", coords)  # noqa: E501

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
                        for s in set(fields["s"]):
                            key = (date, level, num, para, s)
                            if s not in combined_dict[date][num][para]:
                                if key in range_dict:
                                    combined_dict[date][num][para][s] = range_dict[key]
                            else:
                                # Concatenate arrays
                                if key in range_dict:
                                    combined_dict[date][num][para][s] += range_dict[key]

        logging.debug("The values returned from combined dicts: %s", combined_dict)  # noqa: E501

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        logging.debug("The fields retrieved were: %s", fields)  # noqa: E501
        logging.debug("The range_dict created was: %s", range_dict)  # noqa: E501

        for date in combined_dict.keys():
            for num in combined_dict[date].keys():
                val_dict = {}
                for para in combined_dict[date][num].keys():
                    if para not in val_dict:
                        val_dict[para] = []
                    for step in combined_dict[date][num][para].keys():
                        val_dict[para].extend(combined_dict[date][num][para][step])
                mm = mars_metadata.copy()
                mm["number"] = num
                mm["Forecast date"] = date
                if "levelist" in mm:
                    del mm["levelist"]
                self.add_coverage(mm, coords[date], val_dict, include_z)

        return self.covjson

    def from_polytope_month(self, result):
        coords = {}
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["years"] = []
        fields["months"] = []
        fields["dates"] = []
        fields["levels"] = [0]
        fields["l"] = []
        fields["has_level_axis"] = False

        self.walk_tree_month(result, fields, coords, mars_metadata, range_dict)

        if len(fields["l"]) == 0:
            fields["l"] = [0]

        logging.debug("The values returned from walking tree: %s", range_dict)
        logging.debug("The coordinates returned from walking tree: %s", coords)
        logging.debug("The fields: %s", fields)

        include_z = fields["has_level_axis"]
        self._set_references(include_z)

        # Build trajectory composite tuples: [month_key, x, y(, z)].
        # Each (year, month) key acts as the "t" element of the tuple.
        for date in coords.keys():
            coord = coords[date]["composite"]
            coords[date]["composite"] = []
            start_idx = 0
            for level in set(fields["l"]):
                if (date, level, fields["number"][0], fields["param"][0]) in range_dict:
                    cor_len = len(range_dict[(date, level, fields["number"][0], fields["param"][0])])
                    end_idx = start_idx + cor_len
                    if include_z:
                        if len(fields["levels"]) != 1:
                            for lev in fields["levels"]:
                                for cor in coord[int(start_idx) : int(end_idx)]:
                                    coords[date]["composite"].append([date, cor[1], cor[0], lev])
                        else:
                            for cor in coord[int(start_idx) : int(end_idx)]:
                                coords[date]["composite"].append([date, cor[1], cor[0], level])
                    else:
                        for cor in coord[int(start_idx) : int(end_idx)]:
                            coords[date]["composite"].append([date, cor[1], cor[0]])
                    start_idx = end_idx

        logging.debug("The coordinates returned from walking tree: %s", coords)

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)
        logging.debug("The fields retrieved were: %s", fields)
        logging.debug("The range_dict created was: %s", range_dict)

        for date in fields["dates"]:
            for num in fields["number"]:
                val_dict = {}
                for para in fields["param"]:
                    if para not in val_dict:
                        val_dict[para] = []
                    for level in fields["l"]:
                        key = (date, level, num, para)
                        if key in range_dict:
                            vals = range_dict[key]
                            val_dict[para].extend([item for sublist in vals for item in sublist])
                mm = mars_metadata.copy()
                mm["number"] = num
                if "levelist" in mm:
                    del mm["levelist"]
                self.add_coverage(mm, coords[date], val_dict, include_z)

        return self.covjson
