import logging

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
import xarray as xr
from scipy.spatial import cKDTree

from ..encoder.encoder import sort_step_values
from .decoder import Decoder

logger = logging.getLogger(__name__)

# Default grid metadata for ECMWF oper data (TCo1279 → O1280 reduced Gaussian).
# Used when mars:grid is absent from the CoverageJSON.  Will be replaced once
# polytope exposes real grid information.
_OPER_GRID_DEFAULTS = {
    "gridType": "reduced_gg",
    "N": 1280,
}


class BoundingBox(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()

    def get_domains(self):
        domains = []
        for coverage in self.coverage.coverages:
            domains.append(coverage["domain"])
        return domains

    def get_ranges(self):
        ranges = []
        for coverage in self.coverage.coverages:
            ranges.append(coverage["ranges"])
        return ranges

    def get_values(self):
        values = {}
        for parameter in self.parameters:
            values[parameter] = []
            for range in self.ranges:
                values[parameter].append(range[parameter]["values"])
            # values[parameter] = [
            #    value for sublist in values[parameter] for value in sublist
            # ]
        return values

    def get_coordinates(self):
        return self.domains[0]["axes"]

    def to_geopandas(self):
        pass

    def to_geotiff(self, output_file="multipoint", resolution=0.01):
        if rasterio is None:
            raise ImportError("Please install 'rasterio' to use this feature: pip install covjsonkit[geo]")
        coords = self.covjson["coverages"][0]["domain"]["axes"]["composite"]["values"]
        x = [c[1] for c in coords]  # longitude
        y = [c[0] for c in coords]  # latitude
        # z = [c[2] for c in coords]  # height/time/etc (not used yet)

        # Define grid
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        # Notice: meshgrid with indexing="ij"
        ny = int(np.ceil((y_max - y_min) / resolution))
        nx = int(np.ceil((x_max - x_min) / resolution))

        grid_y, grid_x = np.meshgrid(
            np.linspace(y_max, y_min, ny),  # from north to south
            np.linspace(x_min, x_max, nx),  # from west to east
            indexing="ij",  # row = y, col = x
        )

        # Nearest-neighbor interpolation
        points = np.column_stack([x, y])
        tree = cKDTree(points)

        # Loop through each parameter in ranges
        for param, param_data in self.covjson["coverages"][0]["ranges"].items():
            values = param_data["values"]

            # Interpolate values onto the grid
            _, idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
            grid_values = np.array(values)[idx].reshape((ny, nx))

            # Define transform (upper-left corner, pixel size)
            transform = from_origin(x_min, y_max, resolution, resolution)

            # Write GeoTIFF for the current parameter
            output_path = f"{output_file}_{param}.tif"
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=ny,
                width=nx,
                count=1,
                dtype=grid_values.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(grid_values, 1)
                dst.set_band_description(1, param)

    def to_geojson(self):
        features = []
        for coverage in self.covjson["coverages"]:
            coords = coverage["domain"]["axes"]["composite"]["values"]
            datetime = coverage["domain"]["axes"]["t"]["values"][0]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]

            values = {}
            for key in coverage["ranges"]:
                values[key] = coverage["ranges"][key]["values"]

            for idx, lonlat in enumerate(coords):
                param_vals = {}
                for key in values.keys():
                    param_vals[key] = values[key][idx]
                param_vals["datetime"] = datetime
                if "mars:metadata" in coverage:
                    param_vals["mars:metadata"] = mars_metadata
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lonlat[1], lonlat[0], lonlat[2]]},
                        "properties": param_vals,
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    def to_xarray(self):
        dims = ["datetimes", "number", "steps", "points"]
        dataarraydict = {}

        # Get coordinates
        x = []
        y = []
        z = []
        datetimes = []
        for coord in self.get_coordinates()["composite"]["values"]:
            x.append(float(coord[0]))
            y.append(float(coord[1]))
            z.append(float(coord[2]))
        for datetime in self.get_coordinates()["t"]["values"]:
            datetimes.append(datetime)

        values = {}
        for parameter in self.parameters:
            values[parameter] = {}

        datetimes = []
        numbers = []
        steps = []
        for coverage in self.coverages:
            if "number" not in coverage["mars:metadata"]:
                coverage["mars:metadata"]["number"] = 0
            numbers.append(coverage["mars:metadata"]["number"])
            if "step" not in coverage["mars:metadata"]:
                coverage["mars:metadata"]["step"] = 0
            steps.append(coverage["mars:metadata"]["step"])
            datetimes.append(coverage["domain"]["axes"]["t"]["values"][0])
            for parameter in self.parameters:
                # values[parameter].append(coverage["ranges"][parameter]["values"])
                if coverage["domain"]["axes"]["t"]["values"][0] not in values[parameter]:
                    values[parameter][coverage["domain"]["axes"]["t"]["values"][0]] = {}
                if (
                    coverage["mars:metadata"]["number"]
                    not in values[parameter][coverage["domain"]["axes"]["t"]["values"][0]]
                ):
                    values[parameter][coverage["domain"]["axes"]["t"]["values"][0]][
                        coverage["mars:metadata"]["number"]
                    ] = {}
                values[parameter][coverage["domain"]["axes"]["t"]["values"][0]][coverage["mars:metadata"]["number"]][
                    coverage["mars:metadata"]["step"]
                ] = coverage["ranges"][parameter]["values"]

        datetimes = sorted(list(set(datetimes)))
        numbers = sorted(list(set(numbers)))
        steps = sort_step_values(list(set(steps)))

        new_values = {}
        for parameter in values.keys():
            new_values[parameter] = []
            for i, datetime in enumerate(datetimes):
                new_values[parameter].append([])
                for j, number in enumerate(numbers):
                    new_values[parameter][i].append([])
                    for k, step in enumerate(steps):
                        new_values[parameter][i][j].append(values[parameter][datetime][number][step])

        for parameter in self.parameters:
            dataarray = xr.DataArray(new_values[parameter], dims=dims)
            dataarray.attrs["type"] = self.get_parameter_metadata(parameter)["type"]
            dataarray.attrs["units"] = self.get_parameter_metadata(parameter)["unit"]["symbol"]
            dataarray.attrs["long_name"] = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
            dataarraydict[dataarray.attrs["long_name"]] = dataarray

        ds = xr.Dataset(
            dataarraydict,
            coords=dict(
                datetimes=(["datetimes"], datetimes),
                number=(["number"], numbers),
                steps=(["steps"], steps),
                points=(["points"], list(range(0, len(x)))),
                latitude=(["points"], x),
                longitude=(["points"], y),
                levelist=(["points"], z),
            ),
        )
        for mars_metadata in self.mars_metadata[0]:
            ds.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        # Add date attribute
        ds.attrs["date"] = self.get_coordinates()["t"]["values"][0]

        return ds

    # ------------------------------------------------------------------
    # GRIB export
    # ------------------------------------------------------------------

    def to_grib(self, output_path="output.grib", backend="auto"):
        """Convert the CoverageJSON to a multi-message GRIB file.

        Produces one GRIB message per field — i.e. for each unique
        combination of (parameter, step, number, date/time) found in
        the coverage collection.  This mirrors the output of a standard
        MARS ``retrieve`` with an ``area`` keyword.

        Args:
            output_path: Filesystem path for the output GRIB file.
            backend: GRIB encoding backend to use.  One of ``"auto"``
                (try pymars2grib first, fall back to eccodes),
                ``"mars2grib"``, or ``"eccodes"``.

        Returns:
            The *output_path* that was written.

        Raises:
            ImportError: If no suitable GRIB backend is available.
        """
        from .grib_backends import get_backend

        grib_backend = get_backend(backend)

        messages = []

        for coverage in self.coverages:
            mars_metadata = coverage.get("mars:metadata", {})
            grid_metadata = coverage.get("mars:grid", {})

            mars_dict = self._build_mars_dict(mars_metadata, coverage)
            misc_dict = self._build_misc_dict(grid_metadata, coverage)

            # Compute sort order for N→S, W→E point ordering (MARS convention)
            coords = coverage["domain"]["axes"]["composite"]["values"]
            sort_idx = self._nswe_sort_indices(coords)

            # One GRIB message per parameter
            for param_shortname in self.parameters:
                values = coverage["ranges"][param_shortname]["values"]

                # Reorder values to N→S, W→E
                sorted_values = [values[i] for i in sort_idx]

                field_mars = {**mars_dict, "param": self._shortname_to_param_id(param_shortname)}

                msg_bytes = grib_backend.encode_message(sorted_values, field_mars, misc_dict)
                messages.append(msg_bytes)

        with open(output_path, "wb") as fh:
            for msg in messages:
                fh.write(msg)

        logger.info("Wrote %d GRIB message(s) to %s", len(messages), output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private helpers for to_grib
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mars_dict(mars_metadata, coverage):
        """Normalise ``mars:metadata`` into the dict expected by GRIB backends.

        Handles the ``Forecast date`` ISO-8601 string that polytope-mars
        puts on each coverage, splitting it into separate ``date`` and
        ``time`` keys.
        """
        mars = {}

        # Direct MARS keys
        for key in ("class", "stream", "type", "expver", "levtype", "domain"):
            if key in mars_metadata:
                mars[key] = mars_metadata[key]

        # Date / time
        forecast_date = mars_metadata.get("Forecast date", "")
        if forecast_date:
            # "2025-06-23T00:00:00Z" → date=20250623, time=0000
            dt_str = str(forecast_date).replace("Z", "")
            if "T" in dt_str:
                date_part, time_part = dt_str.split("T", 1)
            else:
                date_part = dt_str
                time_part = "0000"
            mars["date"] = date_part.replace("-", "")
            mars["time"] = time_part.replace(":", "")[:4]
        elif "date" in mars_metadata:
            mars["date"] = str(mars_metadata["date"])
            if "time" in mars_metadata:
                mars["time"] = str(mars_metadata["time"])

        # Step
        if "step" in mars_metadata:
            mars["step"] = str(mars_metadata["step"])

        # Ensemble number
        if "number" in mars_metadata:
            mars["number"] = str(mars_metadata["number"])

        # Level — use the z-coordinate from the first composite point
        coords = coverage["domain"]["axes"]["composite"]["values"]
        if coords and len(coords[0]) > 2:
            level = coords[0][2]
            if level != 0:
                mars["levelist"] = str(int(level))

        return mars

    @staticmethod
    def _build_misc_dict(grid_metadata, coverage):
        """Build the ``misc`` dict with grid geometry for the GRIB backend.

        When ``mars:grid`` is not present on the coverage (i.e. polytope
        does not yet expose grid info), oper defaults are applied so that
        the pipeline can be tested end-to-end.
        """
        misc = {}

        if grid_metadata:
            misc.update(grid_metadata)

        # Apply oper defaults when grid info is missing
        if "gridType" not in misc:
            logger.warning(
                "No gridType in mars:grid metadata — assuming ECMWF oper defaults "
                "(reduced_gg N1280).  This will be replaced when polytope provides "
                "real grid information."
            )
            for key, default in _OPER_GRID_DEFAULTS.items():
                misc.setdefault(key, default)

        # Compute area from coordinates if not already present
        if "area" not in misc:
            coords = coverage["domain"]["axes"]["composite"]["values"]
            if coords:
                lats = [c[0] for c in coords]
                lons = [c[1] for c in coords]
                misc["area"] = [max(lats), min(lons), min(lats), max(lons)]  # N/W/S/E

        # Compute Nj (number of latitude rows) and pl (points per row)
        # from the coordinates if not already provided.
        if "Nj" not in misc or "pl" not in misc:
            coords = coverage["domain"]["axes"]["composite"]["values"]
            if coords:
                from collections import Counter

                lat_counts = Counter(round(c[0], 9) for c in coords)
                # Sort latitudes N→S
                sorted_lats = sorted(lat_counts.keys(), reverse=True)
                misc.setdefault("Nj", len(sorted_lats))
                misc.setdefault("pl", [lat_counts[lat] for lat in sorted_lats])

        return misc

    @staticmethod
    def _nswe_sort_indices(coords):
        """Return indices that sort composite coords into N→S, W→E order.

        MARS GRIB convention: first grid point is the north-west corner,
        scanning west→east within each latitude row, rows ordered north→south.
        """
        # coords is a list of [lat, lon, level] tuples
        # Sort by latitude descending (N→S), then longitude ascending (W→E)
        indexed = list(enumerate(coords))
        indexed.sort(key=lambda item: (-item[1][0], item[1][1]))
        return [i for i, _ in indexed]

    def _shortname_to_param_id(self, shortname):
        """Map a parameter shortname (e.g. ``'2t'``) to its numeric param ID."""
        from covjsonkit.param_db import get_param_id_from_db

        try:
            return str(get_param_id_from_db(shortname))
        except (KeyError, Exception):
            # If the shortname is not in the DB, pass it through as-is
            return shortname
