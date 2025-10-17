import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None
import xarray as xr
from scipy.spatial import cKDTree

from .decoder import Decoder


class Grid(Decoder):
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
        """
        Convert a Grid domainType CoverageJSON CoverageCollection into an xarray.Dataset.

        Dimensions:
        - time (Forecast date)
        - number (ensemble member)
        - step ('t' axis)
        - level ('levelist')
        - latitude
        - longitude
        """
        if self.covjson["type"] != "CoverageCollection":
            raise ValueError("Expected CoverageCollection as root object")

        parameters = self.covjson.get("parameters", {})

        # Collect metadata for unique coords
        if "mars:metadata" not in self.covjson["coverages"][0]:
            times = [0]
        else:
            times = sorted({cov["mars:metadata"]["Forecast date"] for cov in self.covjson["coverages"]})
        if "mars:metadata" not in self.covjson["coverages"][0]:
            numbers = [0]
        else:
            numbers = sorted({cov["mars:metadata"].get("number", 0) for cov in self.covjson["coverages"]})

        # Initialize coords from first coverage
        first_cov = self.covjson["coverages"][0]
        domain = first_cov["domain"]["axes"]

        if "latitude" in domain:
            x_coords = "latitude"
        else:
            x_coords = "x"
        if "longitude" in domain:
            y_coords = "longitude"
        else:
            y_coords = "y"
        if "levelist" in domain:
            z_coords = "levelist"
        else:
            z_coords = "z"

        steps = np.array(domain.get("t", {}).get("values", [0]))
        levels = np.array(domain.get(z_coords, {}).get("values", [0]))
        lat = np.array(domain[x_coords]["values"])
        lon = np.array(domain[y_coords]["values"])

        # Prepare arrays for each parameter
        data_arrays = {
            pname: np.full(
                (len(times), len(numbers), len(steps), len(levels), len(lat), len(lon)),
                np.nan,
                dtype=float,
            )
            for pname in first_cov["ranges"].keys()
        }

        # Fill arrays
        for coverage in self.covjson["coverages"]:
            if "mars:metadata" not in coverage:
                md = {"Forecast date": 0, "number": 0}
            else:
                md = coverage["mars:metadata"]
            t_idx = times.index(md["Forecast date"])
            n_idx = numbers.index(md.get("number", 0))

            for pname, prange in coverage["ranges"].items():
                arr = np.array(prange["values"]).reshape(prange["shape"])
                data_arrays[pname][t_idx, n_idx, :, :, :, :] = arr

        # Build xarray Dataset
        xr_vars = {
            pname: (
                ["datetimes", "number", "steps", "levelist", "latitude", "longitude"],
                arr,
            )
            for pname, arr in data_arrays.items()
        }

        ds = xr.Dataset(
            xr_vars,
            coords={
                "datetimes": ("datetimes", np.array(times)),
                "number": ("number", np.array(numbers)),
                "steps": ("steps", steps),
                "levelist": ("levelist", levels),
                "latitude": ("latitude", lat),
                "longitude": ("longitude", lon),
            },
        )

        # Attach parameter metadata
        for pname, param in parameters.items():
            for k, v in param.items():
                ds[pname].attrs[k] = v

        return ds
