import numpy as np

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
        steps = sorted(list(set(steps)))

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
