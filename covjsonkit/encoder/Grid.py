import logging
import time

from .encoder import Encoder, is_reanalysis, normalize_step_value, valid_time


class Grid(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "Grid"
        self.covjson["coverages"] = []

    def add_coverage(self, mars_metadata, coords, values, include_z=False):
        new_coverage = {}
        new_coverage["mars:metadata"] = {}
        new_coverage["type"] = "Coverage"
        new_coverage["domain"] = {}
        new_coverage["ranges"] = {}
        self.add_mars_metadata(new_coverage, mars_metadata)
        self.add_domain(new_coverage, coords, include_z)
        self.add_range(new_coverage, values, include_z)
        self.covjson["coverages"].append(new_coverage)

    def add_domain(self, coverage, coords, include_z=False):
        # OGC-compliant Grid domain: regularly-sampled named axes ``x``
        # (longitude), ``y`` (latitude), ``t`` (time) and an optional ``z``
        # (level). ``z`` is omitted at the surface. Coordinate values may be
        # provided under the spec keys (x/y/z) or the legacy keys
        # (longitude/latitude/levelist).
        axes = {"t": {"values": coords["t"]}}
        if include_z:
            axes["z"] = {"values": coords.get("z", coords.get("levelist"))}
        axes["y"] = {"values": coords.get("y", coords.get("latitude"))}
        axes["x"] = {"values": coords.get("x", coords.get("longitude"))}
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = axes

    def add_range(self, coverage, values, include_z=False):
        axis_names = ["t", "z", "y", "x"] if include_z else ["t", "y", "x"]
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            coverage["ranges"][param]["shape"] = self.shp
            coverage["ranges"][param]["axisNames"] = axis_names
            coverage["ranges"][param]["values"] = values[parameter]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def add_if_not_close(self, my_list, number, threshold=0.01):
        if all(abs(number - x) > threshold for x in my_list):
            my_list.append(number)

    def _set_references(self, include_z):
        # Grid referencing is split into GeographicCRS (x, y), a TemporalRS (t)
        # and an optional VerticalCRS (z) so each coordinate carries the correct
        # coordinate reference system.
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
        Converts an xarray dataset into a Grid CoverageJSON collection.
        """

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "Grid"
        self.covjson["coverages"] = []

        include_z = "levelist" in dataset.coords

        # Add reference system (split GeographicCRS / TemporalRS / VerticalCRS)
        self._set_references(include_z)

        for data_var in dataset.data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        # Prepare shared spatial coordinates (spec-named axes: x=lon, y=lat,
        # z=level). Each coverage carries a single time step, so the ``t`` axis
        # and range shape use a t-dimension of 1 (set per coverage below).
        y_vals = dataset["latitude"].values.tolist()
        x_vals = dataset["longitude"].values.tolist()
        z_vals = dataset["levelist"].values.tolist() if include_z else None
        if include_z:
            self.shp = [1, len(z_vals), len(y_vals), len(x_vals)]
        else:
            self.shp = [1, len(y_vals), len(x_vals)]

        for datetime in dataset["datetimes"].values:
            for num in dataset["number"].values:
                for step in dataset["steps"].values:
                    dv_dict = {}
                    mars_metadata = {metadata: dataset.attrs[metadata] for metadata in dataset.attrs}
                    mars_metadata["number"] = int(num)
                    mars_metadata["step"] = normalize_step_value(step)
                    mars_metadata["Forecast date"] = str(datetime)
                    for dv in dataset.data_vars:
                        # Flatten in axis order z, y, x (or y, x at the surface).
                        # ravel() handles both the 3D (level) and 2D (surface)
                        # cases without assuming a level dimension.
                        arr = dataset[dv].sel(datetimes=datetime, number=num, steps=step).values
                        dv_dict[dv] = arr.ravel().tolist()

                    coords = {"t": [str(step)], "y": y_vals, "x": x_vals}
                    if include_z:
                        coords["z"] = z_vals
                    self.add_coverage(mars_metadata, coords, dv_dict, include_z)

        # Return the generated CoverageJSON
        return self.covjson

    def from_polytope(self, result, date_key: str = "date") -> dict:
        """Encode a polytope ``TensorIndexTree`` result into a Grid CoverageJSON collection."""
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
        fields["has_level_axis"] = False

        self.walk_tree(result, fields, coords, mars_metadata, range_dict, date_key=date_key)
        logging.debug("The values returned from walking tree: %s", range_dict)  # noqa: E501
        logging.debug("The coordinates returned from walking tree: %s", coords)  # noqa: E501

        include_z = fields["has_level_axis"]
        self._set_references(include_z)

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
                        for s in fields["step"]:
                            key = (date, level, num, para, s)
                            if s not in combined_dict[date][num][para]:
                                combined_dict[date][num][para][s] = range_dict[key]
                            else:
                                # Concatenate arrays
                                combined_dict[date][num][para][s] += range_dict[key]

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        logging.debug("The fields retrieved were: %s", fields)  # noqa: E501
        logging.debug("The range_dict created was: %s", range_dict)  # noqa: E501

        # Build named grid axes: y (latitude) and x (longitude) are the unique
        # sorted coordinate values; walk_tree stores each point as [lat, lon].
        coordinates = {}
        for date in coords.keys():
            coordinates[date] = {}
            # ``t`` axis holds the valid-time (forecast date + step) for each step.
            coordinates[date]["t"] = [valid_time(date, s) for s in fields["step"]]
            if include_z:
                coordinates[date]["z"] = list(fields["levels"])
            coordinates[date]["y"] = []
            coordinates[date]["x"] = []
            for cor in coords[date]["composite"]:
                self.add_if_not_close(coordinates[date]["y"], cor[0])  # latitude
                self.add_if_not_close(coordinates[date]["x"], cor[1])  # longitude

        first_date = fields["dates"][0]
        if include_z:
            self.shp = [
                len(coordinates[first_date]["t"]),
                len(coordinates[first_date]["z"]),
                len(coordinates[first_date]["y"]),
                len(coordinates[first_date]["x"]),
            ]
        else:
            self.shp = [
                len(coordinates[first_date]["t"]),
                len(coordinates[first_date]["y"]),
                len(coordinates[first_date]["x"]),
            ]

        reanalysis = is_reanalysis(mars_metadata, date_key)

        for date in combined_dict.keys():
            for num in combined_dict[date].keys():
                val_dict = {}
                for para in combined_dict[date][num].keys():
                    val_dict[para] = []
                    for step in combined_dict[date][num][para].keys():
                        val_dict[para].extend(combined_dict[date][num][para][step])
                mm = mars_metadata.copy()
                mm["number"] = num
                if reanalysis:
                    mm.pop("Forecast date", None)
                    mm.pop("step", None)
                else:
                    mm["step"] = normalize_step_value(step)
                    mm["Forecast date"] = date
                self.add_coverage(mm, coordinates[date], val_dict, include_z)

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
        fields["has_level_axis"] = False

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)
        self.walk_tree_month(result, fields, coords, mars_metadata, range_dict)
        end = time.time()
        logging.debug("Tree walking ends at: %s", end)
        logging.debug("Tree walking takes: %s", end - start)

        start = time.time()
        logging.debug("Coords creation: %s", start)

        include_z = fields["has_level_axis"]
        self._set_references(include_z)

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)
        logging.debug("The fields retrieved were: %s", fields)
        logging.debug("The range_dict created was: %s", range_dict)

        # Build named grid axes, one entry per date key ("YYYY-MM").
        # The t axis holds the "first of month" ISO timestamps for all dates.
        coordinates = {}
        for date in fields["dates"]:
            coordinates[date] = {}
            coordinates[date]["t"] = [f"{date}-01T00:00:00Z"]
            if include_z:
                coordinates[date]["z"] = list(fields["levels"])
            coordinates[date]["y"] = []
            coordinates[date]["x"] = []
            for cor in coords.get(date, {}).get("composite", []):
                self.add_if_not_close(coordinates[date]["y"], cor[0])  # latitude
                self.add_if_not_close(coordinates[date]["x"], cor[1])  # longitude

        if fields["dates"]:
            first_date = fields["dates"][0]
            if include_z:
                self.shp = [
                    len(coordinates[first_date]["t"]),
                    len(coordinates[first_date]["z"]),
                    len(coordinates[first_date]["y"]),
                    len(coordinates[first_date]["x"]),
                ]
            else:
                self.shp = [
                    len(coordinates[first_date]["t"]),
                    len(coordinates[first_date]["y"]),
                    len(coordinates[first_date]["x"]),
                ]

        end = time.time()
        logging.debug("Coords creation: %s", end)
        logging.debug("Coords creation takes: %s", end - start)

        start = time.time()
        logging.debug("Coverage creation: %s", start)

        for num in fields["number"]:
            val_dict = {}
            for para in fields["param"]:
                val_dict[para] = []
                for date in fields["dates"]:
                    for level in fields["levels"]:
                        key = (date, level, num, para)
                        vals = range_dict.get(key, [[]])
                        val_dict[para].extend([item for sublist in vals for item in sublist])
            mm = mars_metadata.copy()
            mm["number"] = num
            if fields["dates"]:
                self.add_coverage(mm, coordinates[fields["dates"][0]], val_dict, include_z)

        end = time.time()
        logging.debug("Coverage creation: %s", end)
        logging.debug("Coverage creation takes: %s", end - start)

        return self.covjson
