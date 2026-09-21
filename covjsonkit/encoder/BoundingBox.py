import logging
import time

from .encoder import Encoder, is_reanalysis, normalize_step_value, valid_time


class BoundingBox(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        self.covjson["domainType"] = "MultiPoint"
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
        # OGC-compliant MultiPoint domain: a single ``composite`` axis whose
        # tuples are ordered [x, y] (== [longitude, latitude]) at the surface,
        # or [x, y, z] (== [longitude, latitude, level]) when a vertical axis
        # is present. ``t`` carries a single time value per coverage.
        coverage["domain"]["type"] = "Domain"
        coverage["domain"]["axes"] = {}
        coverage["domain"]["axes"]["t"] = {"values": coords["t"]}
        coverage["domain"]["axes"]["composite"] = {
            "dataType": "tuple",
            "coordinates": ["x", "y", "z"] if include_z else ["x", "y"],
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
        # MultiPoint referencing is split into GeographicCRS (x, y), an optional
        # VerticalCRS (z) and a TemporalRS (t) so each coordinate carries the
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
        Converts an xarray dataset into a MultiPoint CoverageJSON collection.
        """

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "MultiPoint"
        self.covjson["coverages"] = []

        include_z = "levelist" in dataset.coords

        # Add reference system (split GeographicCRS / TemporalRS / VerticalCRS)
        self._set_references(include_z)

        for data_var in dataset.data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        # Prepare coordinates: composite tuples ordered [x, y(, z)].
        coords = {
            "composite": [],
            "t": [str(x) for x in dataset["datetimes"].values],
        }

        for point in dataset["points"].values:
            lon = float(dataset.isel(points=point).longitude.values)
            lat = float(dataset.isel(points=point).latitude.values)
            if include_z:
                level = float(dataset.isel(points=point).levelist.values)
                coords["composite"].append([lon, lat, level])
            else:
                coords["composite"].append([lon, lat])

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
        """Encode a polytope ``TensorIndexTree`` result into a MultiPoint (BoundingBox) CoverageJSON collection."""
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

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        logging.debug("The fields retrieved were: %s", fields)  # noqa: E501
        logging.debug("The range_dict created was: %s", range_dict)  # noqa: E501

        # Build composite tuples ordered [x, y] (== [lon, lat]) at the surface,
        # or [x, y, z] (== [lon, lat, level]) when a vertical axis is present.
        # walk_tree stores each spatial point as [lat, lon].
        for date in coords.keys():
            coord = coords[date]["composite"]
            coords[date]["composite"] = []
            for level in levels:
                for cor in coord:
                    if include_z:
                        coords[date]["composite"].append([cor[1], cor[0], level])
                    else:
                        coords[date]["composite"].append([cor[1], cor[0]])

        reanalysis = is_reanalysis(mars_metadata, date_key)

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
                    mm["step"] = normalize_step_value(step)
                    # ``t`` is the valid-time (forecast date + step offset).
                    cov_coords = dict(coords[date])
                    cov_coords["t"] = [valid_time(date, step)]
                    if reanalysis:
                        # Reanalysis (class=ce, stream=efcl): expose only the
                        # valid-time; drop the scalar forecast-date metadata.
                        mm.pop("Forecast date", None)
                        mm.pop("step", None)
                    else:
                        mm["Forecast date"] = date
                    self.add_coverage(mm, cov_coords, val_dict[step], include_z)

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

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)

        # Build per-date composite coordinates, expanding each spatial point
        # across all levels when a vertical axis is present.
        coordinates = {}
        for date in fields["dates"]:
            t_val = f"{date}-01T00:00:00Z"
            coordinates[date] = {}
            coordinates[date]["composite"] = []
            coordinates[date]["t"] = [t_val]
            coord = coords.get(date, {}).get("composite", [])
            for level in levels:
                for cor in coord:
                    if include_z:
                        coordinates[date]["composite"].append([cor[1], cor[0], level])
                    else:
                        coordinates[date]["composite"].append([cor[1], cor[0]])

        end = time.time()
        logging.debug("Coords creation: %s", end)
        logging.debug("Coords creation takes: %s", end - start)

        start = time.time()
        logging.debug("Coverage creation: %s", start)

        for num in fields["number"]:
            for date in fields["dates"]:
                val_dict = {}
                for para in fields["param"]:
                    val_dict[para] = []
                    for level in fields["levels"]:
                        key = (date, level, num, para)
                        vals = range_dict.get(key, [[]])
                        val_dict[para].extend([item for sublist in vals for item in sublist])
                mm = mars_metadata.copy()
                mm["number"] = num
                mm["Forecast date"] = f"{date}-01T00:00:00Z"
                self.add_coverage(mm, coordinates[date], val_dict, include_z)

        end = time.time()
        logging.debug("Coverage creation: %s", end)
        logging.debug("Coverage creation takes: %s", end - start)

        return self.covjson
