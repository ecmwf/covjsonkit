import json
from datetime import datetime
from importlib import resources as importlib_resources

import numpy as np

try:
    from eccodes import (
        codes_get_message,
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
    )
except ImportError:
    codes_grib_new_from_samples = None

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
import xarray as xr
from scipy.spatial import cKDTree

from .decoder import Decoder


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

    def _require_eccodes(self):
        if codes_grib_new_from_samples is None:
            raise ImportError("Please install 'eccodes' to use to_grib(): pip install eccodes")

    def _load_paramid_map(self):
        # Adjust this if paramid.json has a different structure
        with importlib_resources.open_text("covjsonkit.data.ecmwf", "param_id.json") as f:
            return json.load(f)

    def _paramid_for_parameter(self, parameter, mars_metadata):
        if mars_metadata and "param" in mars_metadata:
            return int(mars_metadata["param"])
        mapping = self._load_paramid_map()
        if parameter in mapping:
            return int(mapping[parameter])
        raise KeyError(f"Unable to resolve paramId for parameter '{parameter}'")

    def _parse_forecast_datetime(self, value):
        # Expected ISO format, e.g. "2026-02-25T00:00:00Z"
        if isinstance(value, str):
            value = value.replace("Z", "+00:00")
            return datetime.fromisoformat(value)
        return None

    def _set_mars_metadata(self, handle, mars_metadata, level_value=None):
        if not mars_metadata:
            return

        # Date/time
        dt = self._parse_forecast_datetime(mars_metadata.get("Forecast date"))
        if dt is not None:
            codes_set(handle, "dataDate", int(dt.strftime("%Y%m%d")))
            codes_set(handle, "dataTime", int(dt.strftime("%H%M")))

        # Step
        step = mars_metadata.get("step", 0)
        codes_set(handle, "forecastTime", int(step))

        # Ensemble number
        # if "number" in mars_metadata:
        #    codes_set(handle, "perturbationNumber", int(mars_metadata["number"]))

        # Level type mapping (minimal)
        levtype_map = {
            "sfc": "surface",
            "pl": "isobaricInhPa",
            "ml": "hybrid",
        }
        if "levtype" in mars_metadata:
            codes_set(handle, "typeOfLevel", levtype_map.get(mars_metadata["levtype"], "surface"))

        if level_value is not None:
            codes_set(handle, "level", int(level_value))

    def to_grib(self, path=None, resolution=0.1, sample="regular_ll_sfc_grib2"):
        """
        Convert BoundingBox CoverageJSON to GRIB2 (regular lat/lon grid via interpolation).

        Args:
            path (str|None): output path; if None, return bytes.
            resolution (float): grid resolution in degrees.
            sample (str): ecCodes GRIB2 sample name (default: regular_ll_sfc_grib2).

        Returns:
            bytes or str: GRIB bytes or output path.
        """
        self._require_eccodes()

        messages = []
        for coverage in self.coverages:
            coords = coverage["domain"]["axes"]["composite"]["values"]
            src_lats = [float(c[0]) for c in coords]
            src_lons = [float(c[1]) for c in coords]
            levels = [c[2] for c in coords] if len(coords[0]) > 2 else None
            level_value = levels[0] if levels else None

            # Define regular grid bounds from points
            lat_min, lat_max = min(src_lats), max(src_lats)
            lon_min, lon_max = min(src_lons), max(src_lons)

            # Build grid (north -> south)
            ny = int(np.ceil((lat_max - lat_min) / resolution)) + 1
            nx = int(np.ceil((lon_max - lon_min) / resolution)) + 1
            grid_lats = np.linspace(lat_max, lat_min, ny)
            grid_lons = np.linspace(lon_min, lon_max, nx)

            grid_lat2d, grid_lon2d = np.meshgrid(grid_lats, grid_lons, indexing="ij")

            # Nearest neighbour interpolation
            points = np.column_stack([src_lons, src_lats])
            tree = cKDTree(points)

            mars_metadata = coverage.get("mars:metadata", {})

            for param_name, param_range in coverage["ranges"].items():
                values = np.array(param_range["values"], dtype=float)

                # Interpolate to grid
                _, idx = tree.query(np.column_stack([grid_lon2d.ravel(), grid_lat2d.ravel()]))
                grid_values = values[idx].reshape((ny, nx))

                handle = codes_grib_new_from_samples(sample)

                # Regular lat/lon grid definition
                codes_set(handle, "Ni", nx)
                codes_set(handle, "Nj", ny)
                codes_set(handle, "latitudeOfFirstGridPointInDegrees", float(lat_max))
                codes_set(handle, "longitudeOfFirstGridPointInDegrees", float(lon_min))
                codes_set(handle, "latitudeOfLastGridPointInDegrees", float(lat_min))
                codes_set(handle, "longitudeOfLastGridPointInDegrees", float(lon_max))
                codes_set(handle, "iDirectionIncrementInDegrees", float(resolution))
                codes_set(handle, "jDirectionIncrementInDegrees", float(resolution))
                codes_set(handle, "jScansPositively", 0)  # north->south
                codes_set(handle, "iScansNegatively", 0)

                param_id = self._paramid_for_parameter(param_name, mars_metadata)
                codes_set(handle, "paramId", int(param_id))

                self._set_mars_metadata(handle, mars_metadata, level_value=level_value)

                # Values in scanning order (C order matches ij meshgrid above)
                codes_set_values(handle, grid_values.ravel(order="C"))

                messages.append(codes_get_message(handle))
                codes_release(handle)

        grib_bytes = b"".join(messages)

        if path is not None:
            with open(path, "wb") as f:
                f.write(grib_bytes)
            return path

        return grib_bytes

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
        for dt in self.get_coordinates()["t"]["values"]:
            datetimes.append(dt)

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
        steps = sorted(list(set(steps)))

        new_values = {}
        for parameter in values.keys():
            new_values[parameter] = []
            for i, dt in enumerate(datetimes):
                new_values[parameter].append([])
                for j, number in enumerate(numbers):
                    new_values[parameter][i].append([])
                    for k, step in enumerate(steps):
                        new_values[parameter][i][j].append(values[parameter][dt][number][step])

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
