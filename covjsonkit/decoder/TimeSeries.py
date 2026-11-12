import numpy as np
import pandas as pd
import xarray as xr

from .decoder import Decoder


class TimeSeries(Decoder):
    def __init__(self, covjson):
        super().__init__(covjson)
        self.domains = self.get_domains()
        self.ranges = self.get_ranges()
        first_axes = self.covjson["coverages"][0]["domain"]["axes"]
        # MultiPointSeries coverages carry a ``composite`` (x/y tuple) axis and a
        # shared ``t`` axis. Detect this to switch to the combined decode paths.
        self.multipoint = self.covjson.get("domainType") == "MultiPointSeries" or "composite" in first_axes
        # Backwards-compatible axis-name detection: read both spec-compliant
        # coverages (x/y/z) and legacy coverages (longitude/latitude/levelist).
        # Semantics are preserved: x == longitude, y == latitude.
        self.x_name = "x" if "x" in first_axes else "longitude"
        self.y_name = "y" if "y" in first_axes else "latitude"
        if "z" in first_axes:
            self.z_name = "z"
        elif "levelist" in first_axes:
            self.z_name = "levelist"
        else:
            self.z_name = None

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
        return values

    def get_coordinates(self):
        coord_dict = {}
        for param in self.parameters:
            coord_dict[param] = []
        # Get x,y,z,t coords and unpack t coords and match to x,y,z coords
        for ind, domain in enumerate(self.domains):
            if self.multipoint:
                self._append_multipoint_coordinates(coord_dict, domain, ind)
                continue
            longitude = domain["axes"][self.x_name]["values"][0]
            latitude = domain["axes"][self.y_name]["values"][0]
            level = domain["axes"][self.z_name]["values"][0] if self.z_name else None
            fct = domain["axes"]["t"]["values"][0]
            ts = domain["axes"]["t"]["values"]
            if "number" in self.mars_metadata[ind]:
                num = self.mars_metadata[ind]["number"]
            else:
                num = 0
            for param in self.parameters:
                coords = []
                for t in ts:
                    # Have to replicate these coords for each parameter
                    # coordinates.append([x, y, z, t])
                    coords.append([latitude, longitude, level, fct, t, num])
                coord_dict[param].append(coords)
        return coord_dict

    def _append_multipoint_coordinates(self, coord_dict, domain, ind):
        """MultiPointSeries branch of ``get_coordinates``.

        Emits ``[lat, lon, level, fct, t, number]`` rows in the same order as the
        range values (``t`` outer, composite point inner).
        """
        axes = domain["axes"]
        lons, lats, levels = self._composite_components(axes)
        ts = axes["t"]["values"]
        fct = ts[0]
        num = self.mars_metadata[ind]["number"] if "number" in self.mars_metadata[ind] else 0
        for param in self.parameters:
            coords = []
            for t in ts:
                for lon, lat, level in zip(lons, lats, levels):
                    coords.append([lat, lon, level, fct, t, num])
            coord_dict[param].append(coords)

    @staticmethod
    def _composite_components(axes):
        """Split a ``composite`` axis into ``(lons, lats, levels)`` lists.

        Components are located through ``composite["coordinates"]`` (``x``, ``y``
        and optionally ``z``), so both 2- and 3-tuples are handled. ``levels`` is a
        list of ``None`` when the tuples carry no vertical component and there is no
        separate ``z`` axis (older output carried the level as a separate axis; it
        is honoured as a fallback).
        """
        composite = axes["composite"]
        labels = list(composite.get("coordinates", ["x", "y"]))
        values = composite["values"]

        def position(names):
            for name in names:
                if name in labels:
                    return labels.index(name)
            return None

        ix = position(("x", "longitude"))
        iy = position(("y", "latitude"))
        iz = position(("z", "levelist"))
        ix = 0 if ix is None else ix
        iy = 1 if iy is None else iy
        lons = [pt[ix] for pt in values]
        lats = [pt[iy] for pt in values]
        if iz is not None:
            levels = [pt[iz] for pt in values]
        elif "z" in axes:
            levels = [axes["z"]["values"][0]] * len(values)
        elif "levelist" in axes:
            levels = [axes["levelist"]["values"][0]] * len(values)
        else:
            levels = [None] * len(values)
        return lons, lats, levels

    def to_geopandas(self):
        pass

    def to_geotiff(self):
        raise TypeError("Timeseries domain cannot be converted to GeoTIFF.")

    def to_geojson(self):
        if self.multipoint:
            return self._to_geojson_multipoint()
        features = []
        for coverage in self.covjson["coverages"]:
            longitude = coverage["domain"]["axes"][self.x_name]["values"][0]
            latitude = coverage["domain"]["axes"][self.y_name]["values"][0]
            datetimes = coverage["domain"]["axes"]["t"]["values"]
            if "mars:metadata" in coverage:
                mars_metadata = coverage["mars:metadata"]

            geom_coords = [longitude, latitude]
            if self.z_name:
                geom_coords.append(coverage["domain"]["axes"][self.z_name]["values"][0])

            values = {}
            for key in coverage["ranges"]:
                values[key] = coverage["ranges"][key]["values"]

            for idx, datetime in enumerate(datetimes):
                param_vals = {}
                for key in values.keys():
                    param_vals[key] = values[key][idx]
                param_vals["datetime"] = datetime
                if "mars:metadata" in coverage:
                    param_vals["mars:metadata"] = mars_metadata
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": geom_coords},
                        "properties": param_vals,
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return geojson

    # function to convert covjson to xarray dataset
    def to_xarray(self):
        if self.multipoint:
            return self._to_xarray_multipoint()
        # Monthly-means fast path: coverages produced by from_polytope_month() pack all
        # time steps into a single coverage's t-axis and do NOT write "Forecast date"
        # into mars:metadata.  Detect this case and use a simpler (time, point) layout
        # that mirrors the Wkt/Frame/Shapefile decoders.
        has_forecast_date = any("Forecast date" in cov.get("mars:metadata", {}) for cov in self.covjson["coverages"])
        if not has_forecast_date:
            return self._to_xarray_no_forecast_date()

        if self.z_name:
            dims = ["latitude", "longitude", "levelist", "number", "datetime", "t"]
        else:
            dims = ["latitude", "longitude", "number", "datetime", "t"]
        ds = []

        all_coords = self.get_domains()

        unique_coords = set()
        unique_domains = []

        for domain in self.domains:
            longitude = domain["axes"][self.x_name]["values"][0]
            latitude = domain["axes"][self.y_name]["values"][0]
            t = tuple(domain["axes"]["t"]["values"])

            if self.z_name:
                z = domain["axes"][self.z_name]["values"][0]
                coord_tuple = (longitude, latitude, z, t)
            else:
                coord_tuple = (longitude, latitude, t)

            if coord_tuple not in unique_coords:
                unique_coords.add(coord_tuple)
                unique_domains.append(domain)

        all_coords = unique_domains

        num = []
        datetime = []
        for coverage in self.covjson["coverages"]:
            num.append(coverage["mars:metadata"]["number"])
            datetime.append(coverage["mars:metadata"]["Forecast date"])
        nums = list(set(num))
        datetime = list(set(datetime))

        # Process each coordinate domain
        for coords in all_coords:
            dataarraydict = {}
            longitude = coords["axes"][self.x_name]["values"]
            latitude = coords["axes"][self.y_name]["values"]
            steps = coords["axes"]["t"]["values"]
            steps = [step.replace("Z", "") for step in steps]
            steps = pd.to_datetime(steps)

            if self.z_name:
                z = coords["axes"][self.z_name]["values"]
                cov_idx_list = self._find_coverages(nums, datetime, longitude, latitude, z)
                coord_dict = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "levelist": z,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }
            else:
                cov_idx_list = self._find_coverages(nums, datetime, longitude, latitude, None)
                coord_dict = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "number": nums,
                    "datetime": datetime,
                    "t": steps,
                }

            for parameter in self.parameters:
                param_values = [[[] for _ in range(len(datetime))] for _ in range(len(nums))]

                for i, j, cov in cov_idx_list:
                    param_values[i][j] = cov["ranges"][parameter]["values"]

                long_name = self.get_parameter_metadata(parameter)["observedProperty"]["id"]

                if long_name == "t":
                    long_name = "T"  # Avoid collision with time dimension 't'

                attrs = {
                    "type": self.get_parameter_metadata(parameter)["type"],
                    "units": self.get_parameter_metadata(parameter)["unit"]["symbol"],
                    "long_name": long_name,
                }
                if self.z_name:
                    dataarraydict[long_name] = (dims, [[[param_values]]], attrs)
                else:
                    dataarraydict[long_name] = (dims, [[param_values]], attrs)

            ds.append(xr.Dataset(data_vars=dataarraydict, coords=coord_dict))

        for mars_metadata in self.mars_metadata[0]:
            if mars_metadata != "date" and mars_metadata != "step":
                for dss in ds:
                    dss.attrs[mars_metadata] = self.mars_metadata[0][mars_metadata]

        if len(ds) == 1:
            return ds[0]

        return ds

    def _find_coverages(self, nums, datetime, longitude, latitude, z):
        result = []
        for i, num in enumerate(nums):
            for j, date in enumerate(datetime):
                for coverage in self.covjson["coverages"]:
                    if self._covers_domain(coverage, num, date, longitude, latitude, z):
                        result.append((i, j, coverage))
        return result

    def _covers_domain(self, coverage, num, date, longitude, latitude, z):
        axes = coverage["domain"]["axes"]
        match = (
            coverage["mars:metadata"]["number"] == num
            and coverage["mars:metadata"]["Forecast date"] == date
            and axes[self.x_name]["values"] == longitude
            and axes[self.y_name]["values"] == latitude
        )
        if match and self.z_name and z is not None:
            match = axes[self.z_name]["values"] == z
        return match

    def _to_xarray_no_forecast_date(self):
        """Monthly-means path: all time steps are packed into a single coverage's
        t-axis and there is no "Forecast date" in mars:metadata.
        """
        ds_list = []

        for coverage in self.covjson["coverages"]:
            domain = coverage["domain"]["axes"]
            longitude = domain[self.x_name]["values"]
            latitude = domain[self.y_name]["values"]

            steps = domain["t"]["values"]
            steps = [s.replace("Z", "") for s in steps]
            steps = pd.to_datetime(steps)

            dataarraydict = {}
            for parameter in self.parameters:
                values = coverage["ranges"][parameter]["values"]
                long_name = self.get_parameter_metadata(parameter)["observedProperty"]["id"]

                if long_name == "t":
                    long_name = "T"  # Avoid collision with time dimension 't'

                attrs = {
                    "type": self.get_parameter_metadata(parameter)["type"],
                    "units": self.get_parameter_metadata(parameter)["unit"]["symbol"],
                    "long_name": long_name,
                }

                dataarray = xr.DataArray(values, dims=["t"], coords={"t": steps}, attrs=attrs)

                dataarraydict[long_name] = dataarray

            coord_dict = dict(
                latitude=(["latitude"], latitude),
                longitude=(["longitude"], longitude),
            )
            if self.z_name:
                z = domain[self.z_name]["values"]
                coord_dict["levelist"] = (["levelist"], z)

            dss = xr.Dataset(dataarraydict, coords=coord_dict)

            # Attach MARS metadata (skip keys that vary per time step)
            mm = coverage.get("mars:metadata", {})
            for key, val in mm.items():
                dss.attrs[key] = val

            ds_list.append(dss)

        if len(ds_list) == 1:
            return ds_list[0]
        return ds_list

    def _to_xarray_multipoint(self):
        """MultiPointSeries → one xarray Dataset per coverage.

        Each coverage has a shared ``t`` axis and a ``composite`` (x/y[/z]) axis whose
        values are row-major ``[t, composite]`` in the range.

        * Surface (no z): dims ``(t, points)`` with ``latitude``/``longitude`` as
          non-dimension coordinates along ``points``.
        * Several levels forming a full points × levels grid (the encoder's
          point-major, then level layout, or any permutation of it): dims
          ``(t, points, levelist)`` with ``levelist`` a real dimension coordinate.
        * A single level: dims ``(t, points)`` and a scalar ``levelist`` coordinate.
        * Anything else (ragged, duplicate cells): dims ``(t, points)`` with
          ``latitude``/``longitude``/``levelist`` coordinates per composite entry.

        Multiple coverages (e.g. per ensemble number) return a list of Datasets.
        """
        ds_list = []

        for coverage in self.covjson["coverages"]:
            axes = coverage["domain"]["axes"]

            steps = axes["t"]["values"]
            steps = [s.replace("Z", "") if isinstance(s, str) else s for s in steps]
            steps = pd.to_datetime(steps)

            # composite tuples are [x, y(, z)] = [lon, lat(, level)] per spec; the
            # component positions come from composite["coordinates"].
            longitude, latitude, levels = self._composite_components(axes)
            n_t = len(steps)
            n_c = len(longitude)
            has_levels = any(level is not None for level in levels)
            grid = self._points_levels_grid(longitude, latitude, levels) if has_levels else None

            dataarraydict = {}
            for parameter in self.parameters:
                flat = coverage["ranges"][parameter]["values"]
                # Reshape row-major [t, composite] -> nested [t][composite].
                arr = [flat[j * n_c : (j + 1) * n_c] for j in range(n_t)]

                long_name = self.get_parameter_metadata(parameter)["observedProperty"]["id"]
                if long_name == "t":
                    long_name = "T"  # Avoid collision with time dimension 't'

                attrs = {
                    "type": self.get_parameter_metadata(parameter)["type"],
                    "units": self.get_parameter_metadata(parameter)["unit"]["symbol"],
                    "long_name": long_name,
                }
                if grid is not None:
                    pt_idx, lv_idx, n_pts, n_lv, _, _, _ = grid
                    cube = np.full((n_t, n_pts, n_lv), np.nan)
                    for k in range(n_c):
                        for ti in range(n_t):
                            val = arr[ti][k]
                            cube[ti, pt_idx[k], lv_idx[k]] = np.nan if val is None else val
                    dataarraydict[long_name] = (["t", "points", "levelist"], cube, attrs)
                else:
                    dataarraydict[long_name] = (["t", "points"], arr, attrs)

            if grid is not None:
                _, _, _, _, pt_lon, pt_lat, lv_values = grid
                coord_dict = {
                    "t": steps,
                    "latitude": ("points", pt_lat),
                    "longitude": ("points", pt_lon),
                    "levelist": lv_values,
                }
            else:
                coord_dict = {
                    "t": steps,
                    "latitude": ("points", latitude),
                    "longitude": ("points", longitude),
                }
                if has_levels:
                    if len(set(levels)) == 1:
                        # One level for the whole coverage: scalar coordinate.
                        coord_dict["levelist"] = levels[0]
                    else:
                        coord_dict["levelist"] = ("points", levels)

            dss = xr.Dataset(data_vars=dataarraydict, coords=coord_dict)

            mm = coverage.get("mars:metadata", {})
            for key, val in mm.items():
                dss.attrs[key] = val

            ds_list.append(dss)

        if len(ds_list) == 1:
            return ds_list[0]
        return ds_list

    @staticmethod
    def _points_levels_grid(longitude, latitude, levels):
        """Return the points × levels grid layout of a composite, or ``None``.

        A grid needs more than one level, unique ``(lon, lat, level)`` entries and a
        complete cross product of the distinct points with the distinct levels. Result:
        ``(point_idx, level_idx, n_points, n_levels, point_lon, point_lat, level_values)``
        where ``point_idx[k]``/``level_idx[k]`` locate composite entry ``k`` in the grid.
        """
        point_ids, level_ids = {}, {}
        pt_idx, lv_idx = [], []
        for lon, lat, level in zip(longitude, latitude, levels):
            pt_idx.append(point_ids.setdefault((lon, lat), len(point_ids)))
            lv_idx.append(level_ids.setdefault(level, len(level_ids)))
        n_pts, n_lv = len(point_ids), len(level_ids)
        if n_lv < 2 or len(set(zip(pt_idx, lv_idx))) != len(pt_idx) or len(pt_idx) != n_pts * n_lv:
            return None
        pt_lon = [lon for lon, _ in point_ids]
        pt_lat = [lat for _, lat in point_ids]
        return pt_idx, lv_idx, n_pts, n_lv, pt_lon, pt_lat, list(level_ids)

    def _to_geojson_multipoint(self):
        """MultiPointSeries → one Point feature per (point × time)."""
        features = []
        for coverage in self.covjson["coverages"]:
            axes = coverage["domain"]["axes"]
            datetimes = axes["t"]["values"]
            lons, lats, levels = self._composite_components(axes)
            n_pts = len(lons)
            mars_metadata = coverage.get("mars:metadata")

            values = {key: coverage["ranges"][key]["values"] for key in coverage["ranges"]}

            for p, (lon, lat, level) in enumerate(zip(lons, lats, levels)):
                geom_coords = [lon, lat]
                if level is not None:
                    geom_coords.append(level)
                for t_idx, datetime in enumerate(datetimes):
                    param_vals = {}
                    for key in values:
                        param_vals[key] = values[key][t_idx * n_pts + p]
                    param_vals["datetime"] = datetime
                    if mars_metadata is not None:
                        param_vals["mars:metadata"] = mars_metadata
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": geom_coords},
                            "properties": param_vals,
                        }
                    )

        return {"type": "FeatureCollection", "features": features}
