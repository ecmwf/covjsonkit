import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from .encoder import (
    Encoder,
    anoffset_timedelta,
    iso_utc,
    range_shape,
    reference_datetime,
    set_forecast_date,
    valid_datetime,
    valid_time,
)

_MISSING = object()


def _resolve_missing(flat, include_z, first_error):
    """Finish a flat ``[t, point, level]`` value list that may contain ``_MISSING`` cells.

    Without levels (surface) a missing cell is an error, exactly as before. With
    levels a missing (point, level) cell becomes ``None`` (``null``) as long as some
    cell of the list is present; a list with no data at all still raises.
    """
    if first_error is None:
        return flat
    if not include_z or all(v is _MISSING for v in flat):
        raise first_error
    return [None if v is _MISSING else v for v in flat]


class TimeSeries(Encoder):
    def __init__(self, type, domaintype):
        super().__init__(type, domaintype)
        # ``domaintype`` selects the output shape. ``MultiPointSeries`` forces the
        # combined (composite-axis) representation; ``PointSeries`` uses the legacy
        # per-point coverages but auto-upgrades to MultiPointSeries when a request
        # contains more than one spatial point (decided at emit time in from_polytope).
        self.force_multi = str(domaintype).lower() == "multipointseries"
        # Provisional; finalised once the point count is known.
        self.covjson["domainType"] = "MultiPointSeries" if self.force_multi else "PointSeries"
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
        # cov = Coverage.model_validate_json(json.dumps(new_coverage))
        # self.pydantic_coverage.coverages.append(cov)

    def add_domain(self, coverage, coords, include_z=False):
        coverage["domain"]["type"] = "Domain"
        axes = {}
        axes["x"] = {"values": coords["longitude"]}
        axes["y"] = {"values": coords["latitude"]}
        if include_z:
            axes["z"] = {"values": coords["levelist"]}
        axes["t"] = {"values": coords["t"]}
        coverage["domain"]["axes"] = axes

    def add_range(self, coverage, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {}
            coverage["ranges"][param]["type"] = "NdArray"
            coverage["ranges"][param]["dataType"] = "float"
            shape, axis_names = range_shape(values[parameter], "t")
            coverage["ranges"][param]["shape"] = shape
            coverage["ranges"][param]["axisNames"] = axis_names
            coverage["ranges"][param]["values"] = values[
                parameter
            ]  # [values[parameter][val][0] for val in values[parameter].keys()]

    def add_mars_metadata(self, coverage, metadata):
        coverage["mars:metadata"] = metadata

    def add_multipoint_coverage(self, mars_metadata, t_values, points, values, include_z=False, levels=None):
        """Add a single MultiPointSeries coverage aggregating many points (and levels).

        Composite order (deterministic): **point-major, then level**. Entry
        ``k = p * n_levels + l`` is ``[lon_p, lat_p, level_l]`` (just ``[lon, lat]``
        without levels). The range values are row-major ``[t, composite]`` (``t``
        outer, then point, then level innermost) so they stay aligned with it.

        Args:
            mars_metadata: dict of MARS metadata for the coverage.
            t_values: list of ISO-8601 valid-time strings (the shared ``t`` axis).
            points: list of ``[lon, lat]`` tuples (already x/y ordered per spec).
            values: mapping ``param_id -> flat list`` of values in row-major
                ``[t, composite]`` order, length
                ``len(t_values) * len(points) * n_levels`` (``n_levels`` is 1
                without ``include_z``). ``None`` entries are emitted as ``null``.
            include_z: whether the composite tuples carry a vertical (``z``)
                component (``coordinates`` ``["x", "y", "z"]``).
            levels: the level values (``include_z`` only) crossed with every point;
                all levels live in this one coverage.
        """
        new_coverage = {
            "mars:metadata": {},
            "type": "Coverage",
            "domain": {},
            "ranges": {},
        }
        self.add_mars_metadata(new_coverage, mars_metadata)
        composite = self.add_multipoint_domain(new_coverage, t_values, points, include_z, levels)
        self.add_multipoint_range(new_coverage, t_values, composite, values)
        self.covjson["coverages"].append(new_coverage)

    def add_multipoint_domain(self, coverage, t_values, points, include_z=False, levels=None):
        """Build the domain; returns the composite value list (see ``add_multipoint_coverage``)."""
        coverage["domain"]["type"] = "Domain"
        axes = {}
        axes["t"] = {"values": t_values}
        # OGC CoverageJSON 21-069r2 §9.10.5: a MultiPointSeries domain has only the
        # ``composite`` and ``t`` axes. The vertical coordinate is therefore a third
        # component of each composite tuple, never a separate ``z`` axis.
        if include_z:
            coordinates = ["x", "y", "z"]
            values = [[*pt, level] for pt in points for level in levels]
        else:
            coordinates = ["x", "y"]
            values = points
        axes["composite"] = {
            "dataType": "tuple",
            "coordinates": coordinates,
            "values": values,
        }
        coverage["domain"]["axes"] = axes
        return values

    def add_multipoint_range(self, coverage, t_values, composite, values):
        for parameter in values.keys():
            param = self.convert_param_id_to_param(parameter)
            coverage["ranges"][param] = {
                "type": "NdArray",
                "dataType": "float",
                "shape": [len(t_values), len(composite)],
                "axisNames": ["t", "composite"],
                "values": values[parameter],
            }

    def _set_references(self, include_z):
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

    @staticmethod
    def _emit_composite_points(coords, date, points):
        """Return the spec-ordered ``[x, y] = [lon, lat]`` composite point list.

        Internal storage is ``coords[date]["composite"][i] = [lat, lon]``; the
        CoverageJSON ``composite`` axis wants ``[lon, lat]`` (x first), so swap.
        """
        return [[coords[date]["composite"][i][1], coords[date]["composite"][i][0]] for i in range(points)]

    def _collapse_reanalysis(self, fields, coords, mars_metadata, range_dict):
        """Collapse all hdates for each point into a single coverage.

        Used only for the reanalysis code path (``class=ce``, ``stream=efcl``,
        ``date_key="hdate"``). Instead of one coverage per hdate, each spatial
        point (per level/number) yields a single coverage whose ``t``-axis is the
        chronologically-sorted list of ``hdate + step`` valid-times, with the
        parameter values concatenated in the same order.

        When more than one point is present (or MultiPointSeries is forced) all
        points are further collapsed into a single MultiPointSeries coverage per
        (level, number) sharing a ``composite`` axis; otherwise the legacy
        per-point PointSeries coverages are emitted.

        The scalar ``"Forecast date"`` metadata is intentionally omitted (it no
        longer represents a single timestep); the PointSeries decoder handles
        this via its ``has_forecast_date is False`` fast-path, while the
        MultiPointSeries decoder keys off the ``composite`` axis directly.
        """
        points = len(coords[fields["dates"][0]]["composite"])
        first_date = fields["dates"][0]

        # Optional independent time-of-day offset (separate-datetime reforecast).
        time_offset = fields.get("time_offset")

        # Ordered list of (stamp, date, step) across every hdate/step combination.
        stamp_order = []
        for date in fields["dates"]:
            for step in fields["step"]:
                stamp = valid_datetime(date, step, time_offset)
                stamp_order.append((stamp, date, step))
        # Stable sort by valid-time so ties preserve insertion order.
        stamp_order.sort(key=lambda x: x[0])

        t_values = [iso_utc(stamp) for stamp, _, _ in stamp_order]

        use_multi = self.force_multi or points > 1
        self.covjson["domainType"] = "MultiPointSeries" if use_multi else "PointSeries"

        if use_multi:
            include_z = fields["has_level_axis"]
            pts = self._emit_composite_points(coords, first_date, points)
            # All levels share one coverage per number (composite = point x level).
            levels = fields["levels"] if include_z else fields["levels"][:1]
            for num in fields["number"]:
                val_dict = {}
                for para in fields["param"]:
                    # Row-major [t, composite]: t (valid-time) outer, then point, level innermost.
                    flat = []
                    first_error = None
                    for _stamp, date, step in stamp_order:
                        for i in range(points):
                            for level in levels:
                                key = (date, level, num, para, step)
                                try:
                                    flat.append(range_dict[key][i])
                                except (KeyError, IndexError):
                                    flat.append(_MISSING)
                                    if first_error is None:
                                        first_error = IndexError(
                                            f"Key {key} not found in range_dict. "
                                            f"Please ensure all axes are compressed in config"
                                        )
                    val_dict[para] = _resolve_missing(flat, include_z, first_error)
                mm = mars_metadata.copy()
                mm["number"] = num
                if include_z:
                    mm.pop("levelist", None)  # levels live in the composite z component
                else:
                    mm["levelist"] = levels[0]
                mm.pop("step", None)
                mm.pop("Forecast date", None)
                self.add_multipoint_coverage(mm, t_values, pts, val_dict, include_z=include_z, levels=levels)
            return self.covjson

        for i in range(points):
            lat = coords[first_date]["composite"][i][0]
            lon = coords[first_date]["composite"][i][1]
            for level in fields["levels"]:
                for num in fields["number"]:
                    val_dict = {}
                    for para in fields["param"]:
                        val_dict[para] = []
                        for _stamp, date, step in stamp_order:
                            key = (date, level, num, para, step)
                            try:
                                val_dict[para].append(range_dict[key][i])
                            except (KeyError, IndexError):
                                raise IndexError(
                                    f"Key {key} not found in range_dict. "
                                    f"Please ensure all axes are compressed in config"
                                )
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["levelist"] = level
                    mm.pop("step", None)
                    mm.pop("Forecast date", None)
                    coord_entry = {
                        "latitude": [lat],
                        "longitude": [lon],
                        "levelist": [level],
                        "t": t_values,
                    }
                    self.add_coverage(mm, coord_entry, val_dict, fields.get("has_level_axis", False))

        return self.covjson

    def from_xarray(self, datasets):
        """
        Converts an xarray dataset or a list of xarray datasets into an OGC CoverageJSON
        coverageCollection of type PointSeries.

        Args:
            datasets (Union[xarray.Dataset, List[xarray.Dataset]]): An xarray dataset or a list of xarray datasets.

        Returns:
            dict: The CoverageJSON representation of the coverage collection.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        self.covjson["type"] = "CoverageCollection"
        self.covjson["domainType"] = "PointSeries"
        self.covjson["coverages"] = []

        include_z = "levelist" in datasets[0].coords

        # Add reference system
        self._set_references(include_z)

        for data_var in datasets[0].data_vars:
            data_var = self.convert_param_to_param_id(data_var)
            self.add_parameter(data_var)

        for dataset in datasets:
            # Process each "number" in the dataset
            for num in dataset["number"].values:
                dv_dict = {}
                for dv in dataset.data_vars:
                    dv_dict[dv] = dataset[dv].sel(number=num).values[0][0][0][0].tolist()

                mars_metadata = {}
                for metadata in dataset.attrs:
                    mars_metadata[metadata] = dataset.attrs[metadata]
                mars_metadata["number"] = int(num)

                self.add_coverage(
                    mars_metadata,
                    {
                        "latitude": [float(x) for x in dataset["latitude"].values],
                        "longitude": [float(x) for x in dataset["longitude"].values],
                        "levelist": [float(x) for x in dataset["levelist"].values] if include_z else None,
                        "t": [str(x) for x in dataset["t"].values],
                    },
                    dv_dict,
                    include_z=include_z,
                )

        return self.covjson

    def from_polytope(self, result, date_key: str = "date", reforecast: bool = False) -> dict:
        """Encode a polytope ``TensorIndexTree`` result into a PointSeries CoverageJSON collection.

        Args:
            result: The polytope ``TensorIndexTree`` containing the data to be converted.
            date_key: Tree axis name to treat as the time dimension
                (``"date"`` for forecasts, ``"hdate"`` for hindcast/reforecast).
            reforecast: When ``True`` the separate-datetime reforecast walker is used,
                which keeps an independent ``time`` axis as a scalar time-of-day offset
                instead of folding it into the ``hdate`` time dimension.
        Returns:
            dict: The CoverageJSON representation of the coverage collection.
        """
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
        fields["has_level_axis"] = False

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)  # noqa: E501
        if reforecast:
            self.walk_tree_reforecast(result, fields, coords, mars_metadata, range_dict)
        else:
            self.walk_tree(result, fields, coords, mars_metadata, range_dict, date_key=date_key)
        end = time.time()
        delta = end - start
        logging.debug("Tree walking ends at: %s", end)  # noqa: E501
        logging.debug("Tree walking takes: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coords creation: %s", start)  # noqa: E501

        include_z = fields["has_level_axis"]
        self._set_references(include_z)

        coordinates = {}

        levels = fields["levels"]
        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        # Reanalysis code path: collapse all hdates for a point into a single
        # coverage. Strictly limited to class=ce, stream=efcl over the hdate axis
        # so all other requests keep their existing per-date output.
        if date_key == "hdate" and mars_metadata.get("class") == "ce" and mars_metadata.get("stream") == "efcl":
            return self._collapse_reanalysis(fields, coords, mars_metadata, range_dict)

        points = len(coords[fields["dates"][0]]["composite"])

        # Emit-time output-shape decision: force MultiPointSeries when requested,
        # otherwise auto-upgrade PointSeries -> MultiPointSeries for multi-point requests.
        use_multi = self.force_multi or points > 1
        self.covjson["domainType"] = "MultiPointSeries" if use_multi else "PointSeries"

        for date in fields["dates"]:
            coordinates[date] = []
            for i, point in enumerate(range(points)):
                coordinates[date].append(
                    {
                        "latitude": [coords[date]["composite"][i][0]],
                        "longitude": [coords[date]["composite"][i][1]],
                        "levelist": [levels[0]],
                    }
                )
                coordinates[date][i]["t"] = []
                for level in fields["levels"]:
                    for num in fields["number"]:
                        for para in fields["param"]:
                            for step in fields["step"]:
                                coordinates[date][i]["t"].append(
                                    valid_time(date, step, anoffset=anoffset_timedelta(mars_metadata))
                                )
                            break
                        break
                    break

        logging.debug("Coordinates created: %s", coordinates)  # noqa: E501

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        logging.debug("The points found were: %s", points)  # noqa: E501
        logging.debug("The fields retrieved were: %s", fields)  # noqa: E501
        logging.debug("The range_dict created was: %s", range_dict)  # noqa: E501

        if use_multi:
            # MultiPointSeries: one coverage per (date, number) aggregating every
            # spatial point and every level onto a shared t-axis + composite axis.
            levels = fields["levels"] if include_z else fields["levels"][:1]
            for date in fields["dates"]:
                # t-axis is identical for all points on a given date.
                t_values = coordinates[date][0]["t"]
                # composite values are [x, y] = [lon, lat] per spec (swap from the
                # internal [lat, lon] storage).
                pts = [[coords[date]["composite"][i][1], coords[date]["composite"][i][0]] for i in range(points)]
                for num in fields["number"]:
                    val_dict = {}
                    for para in fields["param"]:
                        # Row-major [t, composite]: t outer, then point, level innermost.
                        flat = []
                        first_error = None
                        for step in fields["step"]:
                            for i in range(points):
                                for level in levels:
                                    key = (date, level, num, para, step)
                                    try:
                                        flat.append(range_dict[key][i])
                                    except (KeyError, IndexError):
                                        flat.append(_MISSING)
                                        if first_error is None:
                                            first_error = IndexError(
                                                f"Key {key} not found in range_dict. "
                                                f"Please ensure all axes are compressed in config"
                                            )
                        val_dict[para] = _resolve_missing(flat, include_z, first_error)
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["Forecast date"] = date
                    if include_z:
                        mm.pop("levelist", None)  # levels live in the composite z component
                    else:
                        mm["levelist"] = levels[0]
                    mm.pop("step", None)
                    self.add_multipoint_coverage(mm, t_values, pts, val_dict, include_z=include_z, levels=levels)
        else:
            for i, point in enumerate(range(points)):
                for date in fields["dates"]:
                    for level in fields["levels"]:
                        for num in fields["number"]:
                            val_dict = {}
                            for para in fields["param"]:
                                val_dict[para] = []
                                for step in fields["step"]:
                                    key = (date, level, num, para, step)
                                    try:
                                        val_dict[para].append(range_dict[key][i])
                                    except IndexError:
                                        logging.debug(
                                            f"Index {i} out of range for key {key} in range_dict. "
                                            f"Available keys: {list(range_dict.keys())}"
                                        )
                                        raise IndexError(
                                            f"Key {key} not found in range_dict. "
                                            f"Please ensure all axes are compressed in config"
                                        )
                            mm = mars_metadata.copy()
                            mm["number"] = num
                            mm["Forecast date"] = date
                            mm["levelist"] = level
                            coordinates[date][i]["levelist"] = [level]
                            del mm["step"]
                            self.add_coverage(mm, coordinates[date][i], val_dict, include_z)

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson

    def from_polytope_reforecast(self, result) -> dict:
        """Encode separate-datetime reforecast/reanalysis data (``class=ce``).

        Handles a result tree in which ``date``, ``hdate`` and ``time`` are
        **independent axes** (rather than a single merged datetime). ``hdate`` is
        the reforecast time dimension and ``time`` an independent time-of-day
        axis; both may be multi-valued and compressed into a single tree node.

        Every ``(hdate, time, step)`` combination for a spatial point yields a
        valid-time ``hdate + time + step``; all such valid-times for a point are
        collapsed into a single PointSeries coverage whose ``t``-axis holds them
        chronologically sorted, with the parameter values in matching order.

        Backward compatible with the legacy tree where ``time`` was pre-merged
        into ``hdate`` (no separate ``time`` node) and ``hdate`` was branched:
        the timestamps then reduce to ``hdate + step``.
        """
        import itertools

        import numpy as np

        from .encoder import is_merged_node

        include_z = self._tree_has_axis(result, "levelist")
        self._set_references(include_z)

        # Axes that must not leak into the per-coverage mars:metadata block.
        exclude_meta = {
            "latitude",
            "longitude",
            "hdate",
            "time",
            "step",
            "param",
            "number",
            "levelist",
        }

        # coverage_key -> accumulated coverage state.
        coverages = {}
        coverage_order = []
        param_order = []
        anoffsets = {}
        # Set True as soon as a forecast (efas) coverage is created, so the final
        # emission can be reordered date -> time -> point (see below).
        forecast_result = False
        # Requested spatial points / levels in first-seen tree order, recorded
        # independently of the values so a point or level that is null everywhere
        # still appears (as nulls) in a MultiPointSeries composite.
        requested_points = []
        requested_levels = []

        def stringify(value):
            if isinstance(value, (np.datetime64, np.timedelta64)):
                # np.datetime64 already stringifies as ISO-8601 (e.g. 2024-03-01
                # or 2024-03-01T00:00:00); np.timedelta64 has no ISO date form.
                return str(value)
            if isinstance(value, (pd.Timestamp, datetime)):
                # pandas/py datetimes render as "YYYY-MM-DD HH:MM:SS" via str();
                # emit ISO-8601 ("YYYY-MM-DDTHH:MM:SS") so metadata dates are
                # spec-compliant.
                return value.isoformat()
            if isinstance(value, timedelta):
                return str(value)
            return value

        def emit(full_path, flat_result):
            nonlocal forecast_result
            axis_names = [name for name, _ in full_path]
            axis_values = [values for _, values in full_path]
            for idx, combo in enumerate(itertools.product(*axis_values)):
                value = flat_result[idx]
                d = dict(zip(axis_names, combo))
                lat = float(d["latitude"])
                lon = float(d["longitude"])
                level = d.get("levelist", 0)
                try:
                    level = int(level)
                except (TypeError, ValueError):
                    pass
                if (lat, lon) not in requested_points:
                    requested_points.append((lat, lon))
                if level not in requested_levels:
                    requested_levels.append(level)
                if value is None:
                    continue
                number = d.get("number", 0)
                try:
                    number = int(number)
                except (TypeError, ValueError):
                    pass
                para = d.get("param")
                step = d.get("step", 0)
                hdate = d.get("hdate", d.get("date"))
                time_off = d.get("time")
                if "anoffset" in d:
                    anoffsets["anoffset"] = self._single_anoffset(anoffsets, d["anoffset"])
                # ce/ef/gf: step is counted from the analysis reference, anoffset before the run.
                stamp = valid_datetime(hdate, step, time_off, anoffset_timedelta(d))

                # Stream determines the coverage grouping for class=ce:
                #   * efcl (reforecast/reanalysis): collapse all hdates into a
                #     single coverage whose t-axis holds the hdates.
                #   * efas (forecast): date+time is the forecast *run* reference;
                #     each (date, time) run is its own coverage and the steps
                #     become that coverage's t-axis.
                #   * anything else (e.g. enfh): legacy one-coverage-per-hdate
                #     with a scalar "Forecast date".
                stream = d.get("stream")
                is_ce = d.get("class") == "ce"
                collapse = is_ce and stream == "efcl"
                forecast = is_ce and stream == "efas"
                if collapse:
                    key = (lat, lon, level, number)
                elif forecast:
                    # Reference datetime of the forecast run = date + time.
                    reference = reference_datetime(hdate, time_off)
                    key = (lat, lon, level, number, reference)
                else:
                    key = (lat, lon, level, number, stringify(hdate))
                if key not in coverages:
                    meta = {}
                    for name in axis_names:
                        if name in exclude_meta:
                            continue
                        meta[name] = stringify(d[name])
                    if "anoffset" in meta:
                        meta["anoffset"] = anoffsets["anoffset"]
                    meta["number"] = number
                    meta["levelist"] = level
                    # efas: Forecast date is the run (date + time); efcl: none;
                    # otherwise one coverage per hdate.
                    set_forecast_date(meta, reference if forecast else hdate)
                    coverages[key] = {
                        "lat": lat,
                        "lon": lon,
                        "level": level,
                        "number": number,
                        "meta": meta,
                        "params": {},
                    }
                    if forecast:
                        forecast_result = True
                        # Ordering key so coverages come out date -> time -> point
                        # (reference = date + time). Insertion/tree order is
                        # otherwise time-major for multi-date forecast requests.
                        coverages[key]["sort_key"] = (reference, lat, lon, level, number)
                    coverage_order.append(key)

                if para not in param_order:
                    param_order.append(para)
                params = coverages[key]["params"]
                params.setdefault(para, []).append((stamp, float(value)))

        def recurse(node, path):
            children = node.children
            if len(children) == 0:
                # Leaf longitude node: ``path`` already carries latitude/longitude.
                emit(path, node.result)
                return
            for child in children:
                if is_merged_node(child):
                    # Compacted lat/lon leaf: carries its own (lat, lon) + result.
                    lat, lon = child.values[0], child.values[1]
                    merged_path = path + [
                        ("latitude", (lat,)),
                        ("longitude", (lon,)),
                    ]
                    emit(merged_path, child.result)
                    continue
                recurse(child, path + [(child.axis.name, tuple(child.values))])

        recurse(result, [])

        if not coverages:
            raise ValueError("No data was returned.")

        for para in param_order:
            self.add_parameter(para)

        # Forecast (efas) coverages: emit in date -> time -> point order.
        if forecast_result:
            coverage_order.sort(key=lambda k: coverages[k]["sort_key"])

        def series_of(cov):
            """Return ``(t_values, val_dict)`` for one accumulated coverage."""
            params = cov["params"]
            paras = list(params.keys())
            reference = params[paras[0]]
            # Stable sort by valid-time; ties preserve insertion order.
            order = sorted(range(len(reference)), key=lambda i: reference[i][0])
            t_values = [iso_utc(reference[i][0]) for i in order]
            val_dict = {para: [params[para][i][1] for i in order] for para in paras}
            return t_values, val_dict

        # Emit-time output-shape decision: a single spatial point stays a
        # PointSeries; several points (or an explicit ``multipointseries``
        # domaintype) are collapsed into MultiPointSeries coverages.
        point_order = []
        for key in coverage_order:
            point = (coverages[key]["lat"], coverages[key]["lon"])
            if point not in point_order:
                point_order.append(point)
        if include_z:
            # With levels, composite entries are the full requested points x levels
            # cross product, so a point that is null everywhere keeps its (null) entries.
            # Without levels the long-standing behaviour (points without data are
            # dropped) is unchanged.
            requested = list(requested_points)
            if forecast_result:
                # efas coverages are emitted ascending by (lat, lon[, level]).
                requested.sort()
            point_order = requested
        use_multi = self.force_multi or len(point_order) > 1
        self.covjson["domainType"] = "MultiPointSeries" if use_multi else "PointSeries"

        if use_multi:
            # One coverage per (number[, run/hdate]) group; every spatial point and
            # every level of the group shares the t-axis and becomes composite
            # entries (point-major, then level; both in first-seen order).
            level_order = []
            for key in coverage_order:
                if coverages[key]["level"] not in level_order:
                    level_order.append(coverages[key]["level"])
            if include_z:
                # Every requested level, including levels that are null everywhere.
                level_order = list(requested_levels)
                if forecast_result:
                    try:
                        level_order.sort()
                    except TypeError:
                        pass
            levels = level_order if include_z else level_order[:1]
            groups = {}
            group_order = []
            for key in coverage_order:
                gkey = key[3:]
                if gkey not in groups:
                    groups[gkey] = {}
                    group_order.append(gkey)
                groups[gkey][(coverages[key]["lat"], coverages[key]["lon"], coverages[key]["level"])] = key

            for gkey in group_order:
                members = groups[gkey]
                cells = [(lat, lon, level) for lat, lon in point_order for level in levels]
                present = [c for c in cells if c in members]
                if not include_z and len(present) != len(cells):
                    raise ValueError(
                        "MultiPointSeries requires every point to have data for each coverage; "
                        f"group {gkey} is missing {len(cells) - len(present)} point(s)."
                    )
                first_cov = coverages[members[present[0]]]
                t_values, first_vals = series_of(first_cov)
                per_cell = []
                for cell in cells:
                    if cell not in members:
                        # Missing (point, level) cell: null series on the shared t-axis.
                        per_cell.append({para: [None] * len(t_values) for para in first_vals})
                        continue
                    t_p, vals_p = series_of(coverages[members[cell]])
                    if t_p != t_values or set(vals_p) != set(first_vals):
                        raise ValueError(
                            "MultiPointSeries requires every point to share the same valid-times and parameters."
                        )
                    per_cell.append(vals_p)
                val_dict = {}
                for para in first_vals:
                    # Row-major [t, composite]: t outer, composite (point-major, level) inner.
                    val_dict[para] = [vals[para][ti] for ti in range(len(t_values)) for vals in per_cell]
                pts = [[lon, lat] for lat, lon in point_order]
                meta = dict(first_cov["meta"])
                if include_z:
                    meta.pop("levelist", None)  # levels live in the composite z component
                self.add_multipoint_coverage(meta, t_values, pts, val_dict, include_z=include_z, levels=levels)
            return self.covjson

        for key in coverage_order:
            cov = coverages[key]
            t_values, val_dict = series_of(cov)
            coord_entry = {
                "latitude": [cov["lat"]],
                "longitude": [cov["lon"]],
                "levelist": [cov["level"]],
                "t": t_values,
            }
            self.add_coverage(cov["meta"], coord_entry, val_dict, include_z)

        return self.covjson

    def from_polytope_month(self, result):
        """Convert a Polytope result for monthly-mean streams (e.g. clmn) into CovJSON.

        These streams index time with ``year`` and ``month`` axes rather than
        ``date``/``time``/``step``.  Each (year, month) pair becomes a single
        timestep represented as the ISO-8601 string ``"YYYY-MM"``.
        """
        coords = {}
        mars_metadata = {}
        range_dict = {}
        fields = {}
        fields["lat"] = 0
        fields["param"] = 0
        fields["number"] = [0]
        fields["years"] = []
        fields["months"] = []
        fields["dates"] = []  # populated as "YYYY-MM" keys once both year and month are seen
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

        # Use the first date key to discover how many spatial points were found.
        if not fields["dates"]:
            raise ValueError("No year/month data was found in the result tree.")
        first_date = fields["dates"][0]
        points = len(coords[first_date]["composite"])

        # Build coordinate structures: one entry per spatial point per date.
        coordinates = {}
        for date in fields["dates"]:
            coordinates[date] = []
            for i in range(points):
                coord_entry = {
                    "latitude": [coords[date]["composite"][i][0]],
                    "longitude": [coords[date]["composite"][i][1]],
                    "levelist": [fields["levels"][0]],
                    # "YYYY-MM" is the timestep; append a day so it is a valid
                    # ISO-8601 datetime string (first day of the month).
                    "t": [f"{date}-01T00:00:00Z"],
                }
                coordinates[date].append(coord_entry)

        end = time.time()
        logging.debug("Coords creation ends: %s", end)
        logging.debug("Coords creation takes: %s", end - start)

        start = time.time()
        logging.debug("Coverage creation: %s", start)

        logging.debug("The points found were: %s", points)
        logging.debug("The fields retrieved were: %s", fields)
        logging.debug("The range_dict created was: %s", range_dict)

        use_multi = self.force_multi or points > 1
        self.covjson["domainType"] = "MultiPointSeries" if use_multi else "PointSeries"

        if use_multi:
            # MultiPointSeries: one coverage per number aggregating every spatial
            # point and level onto a shared t-axis (one entry per "YYYY-MM") + composite.
            t_values = [f"{date}-01T00:00:00Z" for date in fields["dates"]]
            pts = self._emit_composite_points(coords, first_date, points)
            levels = fields["levels"] if include_z else fields["levels"][:1]
            for num in fields["number"]:
                val_dict = {}
                for para in fields["param"]:
                    # Row-major [t, composite]: date (t) outer, then point, level innermost.
                    flat = []
                    first_error = None
                    for date in fields["dates"]:
                        for i in range(points):
                            for level in levels:
                                key = (date, level, num, para)
                                try:
                                    flat.extend(range_dict[key][i])
                                except (KeyError, IndexError) as exc:
                                    logging.debug(
                                        "Key %s not found or index %s out of range in range_dict: %s",
                                        key,
                                        i,
                                        exc,
                                    )
                                    flat.append(_MISSING)
                                    if first_error is None:
                                        first_error = exc
                    val_dict[para] = _resolve_missing(flat, include_z, first_error)
                mm = mars_metadata.copy()
                mm["number"] = num
                if include_z:
                    mm.pop("levelist", None)  # levels live in the composite z component
                else:
                    mm["levelist"] = levels[0]
                self.add_multipoint_coverage(mm, t_values, pts, val_dict, include_z=include_z, levels=levels)
            end = time.time()
            logging.debug("Coverage creation ends: %s", end)
            logging.debug("Coverage creation takes: %s", end - start)
            return self.covjson

        for i in range(points):
            for j, level in enumerate(fields["levels"]):
                for num in fields["number"]:
                    val_dict = {}
                    for para in fields["param"]:
                        val_dict[para] = []
                        for date in fields["dates"]:
                            key = (date, level, num, para)
                            try:
                                val_dict[para].extend(range_dict[key][i])
                            except (KeyError, IndexError) as exc:
                                logging.debug(
                                    "Key %s not found or index %s out of range in range_dict: %s",
                                    key,
                                    i,
                                    exc,
                                )
                                raise
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    mm["levelist"] = level
                    # Use all date keys as the time series for this coverage.
                    coord_entry = coordinates[first_date][i].copy()
                    coord_entry["levelist"] = [level]
                    coord_entry["t"] = [f"{date}-01T00:00:00Z" for date in fields["dates"]]
                    self.add_coverage(mm, coord_entry, val_dict, include_z)

        end = time.time()
        logging.debug("Coverage creation ends: %s", end)
        logging.debug("Coverage creation takes: %s", end - start)

        return self.covjson

    @staticmethod
    def _set_step_forecast_date(mm, dates):
        """Set the scalar ``"Forecast date"`` for a ``from_polytope_step`` coverage.

        The coverage ``t`` axis spans every date in ``dates``. A scalar forecast
        date is only meaningful when there is exactly one; with several dates it is
        omitted (identically for PointSeries and MultiPointSeries coverages).
        """
        if len(dates) == 1:
            mm["Forecast date"] = dates[0]
        else:
            mm.pop("Forecast date", None)

    def from_polytope_step(self, result):
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
        fields["times"] = []
        fields["has_level_axis"] = False

        start = time.time()
        logging.debug("Tree walking starts at: %s", start)  # noqa: E501
        self.walk_tree_step(result, fields, coords, mars_metadata, range_dict)
        end = time.time()
        delta = end - start
        logging.debug("Tree walking ends at: %s", end)  # noqa: E501
        logging.debug("Tree walking takes: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coords creation: %s", start)  # noqa: E501

        include_z = fields["has_level_axis"]
        self._set_references(include_z)

        coordinates = {}

        if fields["param"] == 0:
            raise ValueError("No data was returned.")
        for para in fields["param"]:
            self.add_parameter(para)

        logging.debug("The parameters added were: %s", self.parameters)  # noqa: E501

        points = len(coords[fields["dates"][0]]["composite"])

        for step in fields["step"]:
            coordinates[fields["dates"][0]] = []
            for i, point in enumerate(range(points)):
                for j, level in enumerate(fields["levels"]):
                    coordinates[fields["dates"][0]].append(
                        {
                            "latitude": [coords[fields["dates"][0]]["composite"][i][0]],
                            "longitude": [coords[fields["dates"][0]]["composite"][i][1]],
                            "levelist": [level],
                        }
                    )
                    coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)]["t"] = []
                    for num in fields["number"]:
                        for para in fields["param"]:
                            for date in fields["dates"]:
                                for times in fields["times"]:
                                    coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)]["t"].append(
                                        self._step_path_valid_time(date, times, fields)
                                    )
                            break
                        break

        end = time.time()
        delta = end - start
        logging.debug("Coords creation: %s", end)  # noqa: E501
        logging.debug("Coords creation: %s", delta)  # noqa: E501

        start = time.time()
        logging.debug("Coverage creation: %s", start)  # noqa: E501

        use_multi = self.force_multi or points > 1
        self.covjson["domainType"] = "MultiPointSeries" if use_multi else "PointSeries"

        if use_multi:
            # MultiPointSeries: one coverage per number aggregating every spatial
            # point and level onto a shared t-axis (date outer, time inner) + composite.
            first_date = fields["dates"][0]
            pts = self._emit_composite_points(coords, first_date, points)
            levels = fields["levels"] if include_z else fields["levels"][:1]
            # t-axis is identical for every point and level.
            t_values = coordinates[first_date][0]["t"]
            for num in fields["number"]:
                val_dict = {}
                for para in fields["param"]:
                    # Row-major [t, composite]: (date, time) outer, then point, level innermost.
                    flat = []
                    first_error = None
                    for date in fields["dates"]:
                        per_date_len = None
                        for level in levels:
                            try:
                                per_date_len = len(range_dict[(date, level, num, para)][0])
                                break
                            except (KeyError, IndexError):
                                continue
                        if per_date_len is None:
                            raise KeyError((date, levels[0], num, para))
                        for t_idx in range(per_date_len):
                            for i in range(points):
                                for level in levels:
                                    key = (date, level, num, para)
                                    try:
                                        flat.append(range_dict[key][i][t_idx])
                                    except (KeyError, IndexError) as exc:
                                        flat.append(_MISSING)
                                        if first_error is None:
                                            first_error = exc
                    val_dict[para] = _resolve_missing(flat, include_z, first_error)
                mm = mars_metadata.copy()
                mm["number"] = num
                self._set_step_forecast_date(mm, fields["dates"])
                if include_z:
                    mm.pop("levelist", None)  # levels live in the composite z component
                else:
                    mm["levelist"] = levels[0]
                self.add_multipoint_coverage(mm, t_values, pts, val_dict, include_z=include_z, levels=levels)
            end = time.time()
            delta = end - start
            logging.debug("Coverage creation: %s", end)  # noqa: E501
            logging.debug("Coverage creation: %s", delta)  # noqa: E501
            return self.covjson

        for i, point in enumerate(range(points)):
            for j, level in enumerate(fields["levels"]):
                for num in fields["number"]:
                    val_dict = {}
                    for para in fields["param"]:
                        val_dict[para] = []
                        for date in fields["dates"]:
                            key = (date, level, num, para)
                            # for k, v in range_dict.items():
                            #    if k == key:
                            # val_dict[para].append(v[0])
                            val_dict[para].extend(range_dict[key][i])
                    mm = mars_metadata.copy()
                    mm["number"] = num
                    self._set_step_forecast_date(mm, fields["dates"])
                    self.add_coverage(
                        mm,
                        coordinates[fields["dates"][0]][(i * len(fields["levels"]) + j)],
                        val_dict,
                        include_z,
                    )

        end = time.time()
        delta = end - start
        logging.debug("Coverage creation: %s", end)  # noqa: E501
        logging.debug("Coverage creation: %s", delta)  # noqa: E501

        return self.covjson
