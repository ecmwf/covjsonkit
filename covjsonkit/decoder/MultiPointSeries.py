import numpy as np
import pandas as pd
import xarray as xr

from .decoder import Decoder


class MultiPointSeries(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        self._validate_coverages()

    def get_domains(self):
        return [coverage["domain"] for coverage in self.coverage.coverages]

    def get_ranges(self):
        return [coverage["ranges"] for coverage in self.coverage.coverages]

    def get_values(self):
        return {parameter: [ranges[parameter]["values"] for ranges in self.ranges] for parameter in self.parameters}

    def get_coordinates(self):
        coordinates = {parameter: [] for parameter in self.parameters}
        for coverage in self.coverages:
            axes = coverage["domain"]["axes"]
            metadata = coverage.get("mars:metadata", {})
            forecast_date = metadata.get("Forecast date", axes["t"]["values"][0])
            number = metadata.get("number", 0)
            rows = []
            for timestamp in axes["t"]["values"]:
                for point in axes["composite"]["values"]:
                    level = point[2] if len(point) == 3 else None
                    rows.append([point[1], point[0], level, forecast_date, timestamp, number])
            for parameter in self.parameters:
                coordinates[parameter].append(rows)
        return coordinates

    def to_geopandas(self):
        pass

    def to_geotiff(self):
        raise TypeError("MultiPointSeries domain cannot be converted to GeoTIFF.")

    def to_geojson(self):
        features = []
        for coverage in self.coverages:
            axes = coverage["domain"]["axes"]
            points = axes["composite"]["values"]
            point_count = len(points)
            metadata = coverage.get("mars:metadata")

            for time_index, timestamp in enumerate(axes["t"]["values"]):
                for point_index, point in enumerate(points):
                    value_index = time_index * point_count + point_index
                    properties = {
                        parameter: coverage["ranges"][parameter]["values"][value_index] for parameter in self.parameters
                    }
                    properties["datetime"] = timestamp
                    if metadata is not None:
                        properties["mars:metadata"] = metadata
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": point},
                            "properties": properties,
                        }
                    )
        return {"type": "FeatureCollection", "features": features}

    def to_xarray(self):
        datasets = []
        for coverages in self._group_by_composite_domain():
            datasets.append(self._to_xarray_group(coverages))
        return datasets[0] if len(datasets) == 1 else datasets

    def _validate_coverages(self):
        for coverage in self.coverages:
            try:
                axes = coverage["domain"]["axes"]
                times = axes["t"]["values"]
                composite = axes["composite"]
                coordinate_names = composite["coordinates"]
                points = composite["values"]
            except KeyError as exc:
                raise ValueError("MultiPointSeries requires t and composite domain axes.") from exc

            if composite.get("dataType") != "tuple":
                raise ValueError("MultiPointSeries composite axis must have tuple dataType.")
            if coordinate_names not in (["x", "y"], ["x", "y", "z"]):
                raise ValueError("MultiPointSeries composite coordinates must be x/y or x/y/z.")
            if not isinstance(times, list) or not isinstance(points, list):
                raise ValueError("MultiPointSeries t and composite values must be lists.")
            if not times or not points:
                raise ValueError("MultiPointSeries t and composite values must not be empty.")
            if any(not isinstance(point, list) or len(point) != len(coordinate_names) for point in points):
                raise ValueError("MultiPointSeries composite tuple width does not match its coordinates.")

            try:
                ranges = coverage["ranges"]
            except KeyError as exc:
                raise ValueError("MultiPointSeries coverage requires ranges.") from exc
            if set(ranges) != set(self.parameters):
                raise ValueError("MultiPointSeries ranges must match the collection parameters.")
            expected_shape = [len(times), len(points)]
            for range_data in ranges.values():
                if range_data.get("axisNames") != ["t", "composite"]:
                    raise ValueError("MultiPointSeries range axisNames must be ['t', 'composite'].")
                if range_data.get("shape") != expected_shape:
                    raise ValueError(
                        f"MultiPointSeries range shape must be {expected_shape}, got {range_data.get('shape')}."
                    )
                values = range_data.get("values")
                if not isinstance(values, list) or len(values) != len(times) * len(points):
                    raise ValueError("MultiPointSeries range value count does not match its shape.")

    def _group_by_composite_domain(self):
        groups = []
        indices = {}
        for coverage in self.coverages:
            composite = coverage["domain"]["axes"]["composite"]
            key = (tuple(composite["coordinates"]), tuple(tuple(point) for point in composite["values"]))
            if key not in indices:
                indices[key] = len(groups)
                groups.append([])
            groups[indices[key]].append(coverage)
        return groups

    def _to_xarray_group(self, coverages):
        first_axes = coverages[0]["domain"]["axes"]
        point_count = len(first_axes["composite"]["values"])
        time_count = len(first_axes["t"]["values"])
        number_values = self._ordered_metadata_values(coverages, "number", 0)
        forecast_dates = self._ordered_metadata_values(coverages, "Forecast date", first_axes["t"]["values"][0])
        values_by_parameter = {
            parameter: np.full(
                (len(number_values), len(forecast_dates), time_count, point_count),
                np.nan,
            )
            for parameter in self.parameters
        }

        valid_times = {}
        for coverage in coverages:
            axes = coverage["domain"]["axes"]
            if len(axes["t"]["values"]) != time_count:
                raise ValueError("MultiPointSeries coverages sharing a composite domain need equal t axis lengths.")
            metadata = coverage.get("mars:metadata", {})
            number = metadata.get("number", 0)
            forecast_date = metadata.get("Forecast date", axes["t"]["values"][0])
            number_index = number_values.index(number)
            forecast_index = forecast_dates.index(forecast_date)
            if forecast_date in valid_times and valid_times[forecast_date] != axes["t"]["values"]:
                raise ValueError("MultiPointSeries coverages sharing a forecast date need identical t axes.")
            valid_times[forecast_date] = axes["t"]["values"]
            for parameter in self.parameters:
                values_by_parameter[parameter][number_index, forecast_index] = np.asarray(
                    coverage["ranges"][parameter]["values"]
                ).reshape(time_count, point_count)

        points = first_axes["composite"]["values"]
        coordinates = {
            "number": number_values,
            "datetime": pd.to_datetime([date.replace("Z", "") for date in forecast_dates]),
            "t": pd.to_datetime([timestamp.replace("Z", "") for timestamp in first_axes["t"]["values"]]),
            "points": list(range(point_count)),
            "longitude": ("points", [point[0] for point in points]),
            "latitude": ("points", [point[1] for point in points]),
            "valid_time": (
                ("datetime", "t"),
                [
                    pd.to_datetime([timestamp.replace("Z", "") for timestamp in valid_times[forecast_date]])
                    for forecast_date in forecast_dates
                ],
            ),
        }
        if len(points[0]) == 3:
            coordinates["levelist"] = ("points", [point[2] for point in points])

        data_vars = {}
        for parameter, values in values_by_parameter.items():
            metadata = self.get_parameter_metadata(parameter)
            name = metadata["observedProperty"]["id"]
            if name == "t":
                name = "T"
            data_vars[name] = (
                ("number", "datetime", "t", "points"),
                values,
                {
                    "type": metadata["type"],
                    "units": metadata["unit"]["symbol"],
                    "long_name": name,
                },
            )

        dataset = xr.Dataset(data_vars=data_vars, coords=coordinates)
        for key, value in coverages[0].get("mars:metadata", {}).items():
            if key not in {"number", "Forecast date"}:
                dataset.attrs[key] = value
        return dataset

    @staticmethod
    def _ordered_metadata_values(coverages, key, default):
        values = []
        for coverage in coverages:
            value = coverage.get("mars:metadata", {}).get(key, default)
            if value not in values:
                values.append(value)
        return values
