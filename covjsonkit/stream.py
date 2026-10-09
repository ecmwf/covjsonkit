"""Streaming CoverageJSON encoder over a block stream.

``CovjsonStreamEncoder`` consumes the block IR produced by polytope-mars (``polytope_mars.blocks``) and
returns the CoverageJSON document in pieces: ``begin(header)`` -> collection opening,
``encode_iter(block)`` -> the fragments of one block, ``end()`` -> collection closing.  Concatenating the
pieces gives exactly the bytes the legacy path produced with
``json.dumps(encoder.from_polytope*(tree)).encode()``, except that missing values (NaN) are written as
``null``.

``encode_iter`` yields one fragment per slice of the block's arrays, each at most
``max_fragment_bytes`` (constructor argument, default :data:`DEFAULT_MAX_FRAGMENT_BYTES` = 8 MiB), so
neither the encoder nor its consumer ever holds the text of a whole block: a whole-world O2560
coordinates block is ~800 MB of JSON but is handed over 8 MiB at a time.  The fragments of a block must
be consumed completely and in order before the next block is encoded (the encoder's state advances as
they are produced).  ``encode(block)`` is the buffered equivalent, ``b"".join(encode_iter(block))``, kept
for callers that want one ``bytes`` per block.  The structural fragments (a coverage's metadata and ``t``
axis, the collection's ``referencing`` and ``parameters``) are written whole: they are a few hundred
bytes, and only point-feature collections, which are small by construction, are buffered to ``end()``.

Blocks are read structurally (attributes only), so this module imports neither polytope-feature nor
polytope-mars:

* header: ``feature_type``, ``domain_type``, ``time_axis``, ``parameters`` (each with ``id``,
  ``shortname``, ``name``, ``unit``, ``description``), ``extra``;
* group (``block.group``): ``index``, ``t``, ``params``, ``levels``, ``n_points``, ``mars_metadata``;
* coordinates block: ``lat``, ``lon`` (float arrays); values block: ``param``, ``level``, ``values``
  (float array, NaN = missing); group end: only ``group``.

A group's points may arrive in one block or in several consecutive ones: blocks are placed by arrival
order, so a producer that splits a field writes the same bytes as one that does not.

MultiPoint coverages are written as the blocks arrive (memory independent of the coverage size, apart
from the coordinates of the current coverage when it has several levels).  PointSeries,
VerticalProfile and Trajectory collections are small and their legacy layout is point-major across field
groups, so their groups are buffered and written by ``end()``.
"""

from __future__ import annotations

import json
import math
from typing import Iterator

import numpy as np
import orjson

__all__ = [
    "CovjsonStreamEncoder",
    "DEFAULT_MAX_FRAGMENT_BYTES",
    "legacy_referencing",
    "pointseries_coverages",
]

#: Default upper bound on the size of one fragment of :meth:`CovjsonStreamEncoder.encode_iter`.
DEFAULT_MAX_FRAGMENT_BYTES = 8 * 1024 * 1024

#: Longest text one float64 can take (``-1.2345678901234567e-308``), used to size array slices.
_FLOAT_CHARS = 24


_CRS = {"type": "GeographicCRS", "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
_LATLON = ["latitude", "longitude", "levelist"]
_XYZ = ["x", "y", "z"]


def _dumps(obj) -> bytes:
    """``json.dumps`` bytes (default separators, ASCII escapes) for small structural fragments."""
    return json.dumps(obj).encode("utf-8")


def _needs_repr(arr: np.ndarray) -> np.ndarray:
    """Mask of the values orjson prints differently from ``json.dumps`` (0 < |x| < 1e-4)."""
    with np.errstate(invalid="ignore"):
        return (np.abs(arr) < 1e-4) & (arr != 0)


def _fragment_limit(value) -> int:
    """``value`` as a positive fragment size in bytes."""
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"max_fragment_bytes must be an integer number of bytes, got {value!r}") from None
    if limit <= 0:
        raise ValueError(f"max_fragment_bytes must be positive, got {limit}")
    return limit


def _fragment_slices(n: int, item_bytes: int, max_fragment_bytes: int) -> Iterator[tuple]:
    """Per fragment, the ``(start, stop)`` point range to serialise into it.

    ``item_bytes`` is an upper bound on the text of one point (its floats at their longest plus
    separators), so a fragment stays within ``max_fragment_bytes``.  A fragment always holds at least one
    point: with a limit smaller than one point's text, the fragment is one point long and exceeds it.
    """
    per_fragment = max(1, max_fragment_bytes // max(1, item_bytes))
    for start in range(0, n, per_fragment):
        yield start, min(start + per_fragment, n)


def format_floats(values) -> list:
    """Each value as ``json.dumps`` writes a Python float, with NaN/inf as ``null``.

    orjson and CPython both print the shortest round-trip representation; they differ only in the
    exponent form of numbers below 1e-4 (orjson ``1e-5``/``0.00002`` vs Python ``1e-05``/``2e-05``),
    which are re-printed with ``repr``.
    """
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    parts = orjson.dumps(arr, option=orjson.OPT_SERIALIZE_NUMPY)[1:-1].split(b",")
    for i in np.flatnonzero(_needs_repr(arr)):
        parts[i] = repr(arr[i].item()).encode("ascii")
    return parts


def float_list_bytes(values) -> bytes:
    """``a, b, c`` (no brackets), as ``json.dumps`` separates list items."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if not _needs_repr(arr).any():
        # fast path without one Python object per value
        return orjson.dumps(arr, option=orjson.OPT_SERIALIZE_NUMPY)[1:-1].replace(b",", b", ")
    return b", ".join(format_floats(arr))


def composite_tuples(lat, lon, level) -> bytes:
    """``[lat, lon, level], ...`` (no outer brackets) as ``json.dumps`` writes the composite values."""
    lat = np.ascontiguousarray(lat, dtype=np.float64)
    lon = np.ascontiguousarray(lon, dtype=np.float64)
    if lat.size == 0:
        return b""
    suffix = b", " + _dumps(level) + b"]"
    if not (_needs_repr(lat).any() or _needs_repr(lon).any()):
        pairs = orjson.dumps(np.column_stack((lat, lon)), option=orjson.OPT_SERIALIZE_NUMPY)[1:-1]
        return pairs.replace(b",", b", ").replace(b"]", suffix)
    return b", ".join([b"[" + a + b", " + o + suffix for a, o in zip(format_floats(lat), format_floats(lon))])


def _py_values(values) -> list:
    """Floats for ``json.dumps``; NaN -> None."""
    return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v for v in values]


def legacy_referencing(header) -> list:
    """The ``coordinates`` of the reference system each legacy encoder wrote (preserved quirk)."""
    domain, feature, role = header.domain_type, header.feature_type, header.time_axis
    if domain == "Trajectory":
        return list(_LATLON) if role == "hdate" else ["t", "x", "y", "z"]
    if domain == "VerticalProfile":
        return list(_LATLON)
    if domain == "PointSeries":
        if role == "step":
            return list(_XYZ)
        if role == "month" and feature == "position":
            return list(_XYZ)
        return list(_LATLON)
    # MultiPoint
    if role == "hdate":
        return list(_LATLON)
    if role == "date":
        return list(_LATLON) if feature in ("boundingbox", "circle") else list(_XYZ)
    if role == "month":
        return list(_LATLON) if feature == "circle" else list(_XYZ)
    return list(_XYZ)


def _parameter(p) -> dict:
    return {
        "type": "Parameter",
        "description": {"en": p.description},
        "unit": {"symbol": p.unit},
        "observedProperty": {"id": p.shortname, "label": {"en": p.name}},
    }


def _range(name, values: list) -> dict:
    return {"type": "NdArray", "dataType": "float", "shape": [len(values)], "axisNames": [name], "values": values}


def _series(groups) -> list:
    """Consecutive buffered groups with equal metadata and levels form one coverage series."""
    series = []
    for g in groups:
        if series and series[-1][0]["meta"] == g["meta"] and series[-1][0]["levels"] == g["levels"]:
            series[-1].append(g)
        else:
            series.append([g])
    return series


def pointseries_coverages(groups, header, shortname) -> list:
    """Legacy PointSeries layout (TimeSeries / Position encoders): one coverage per (point, series, level).

    A series is a run of field groups (time instants) sharing their coverage metadata; its ``t`` axis is
    the groups' ``t`` values in order.  Coverages are point-major (``for point: for series``) unless
    ``header.extra["pointseries_order"] == "series_major"``.  Position coverages from ``from_polytope``
    write the first level on their levelist axis whatever their level (preserved quirk).
    """
    position_quirk = header.feature_type == "position" and header.time_axis == "date"
    order = (header.extra or {}).get("pointseries_order", "point_major")
    all_series = _series(groups)

    def coverage(series, i, level):
        first = series[0]
        levels = first["levels"]
        if position_quirk and levels:
            levelist = [levels[0]]
        else:
            levelist = [0 if level is None else level]
        t = [v for g in series for v in g["t"]]
        ranges = {}
        for p in header.parameters:
            if not any(p.id in g["values_params"] for g in series):
                continue
            vals = []
            for g in series:
                arr = g["values"].get((p.id, level))
                vals.append(arr[i].item() if arr is not None else None)
            ranges[shortname[p.id]] = _range(shortname[p.id], _py_values(vals))
        return {
            "mars:metadata": first["meta"],
            "type": "Coverage",
            "domain": {
                "type": "Domain",
                "axes": {
                    "latitude": {"values": [first["lat"][i].item()]},
                    "longitude": {"values": [first["lon"][i].item()]},
                    "levelist": {"values": levelist},
                    "t": {"values": t},
                },
            },
            "ranges": ranges,
        }

    out = []
    if order == "series_major":
        for series in all_series:
            for i in range(len(series[0]["lat"])):
                for level in series[0]["levels"] or (None,):
                    out.append(coverage(series, i, level))
        return out
    n_points = len(all_series[0][0]["lat"]) if all_series else 0
    for i in range(n_points):
        for series in all_series:
            for level in series[0]["levels"] or (None,):
                out.append(coverage(series, i, level))
    return out


def verticalprofile_coverages(groups, header, shortname) -> list:
    """Legacy VerticalProfile layout: one coverage per (point, group), all levels on the levelist axis."""
    out = []
    n_points = len(groups[0]["lat"]) if groups else 0
    for i in range(n_points):
        for g in groups:
            levels = list(g["levels"]) if g["levels"] else [0]
            keys = list(g["levels"]) if g["levels"] else [None]
            ranges = {}
            for p in header.parameters:
                if p.id not in g["values_params"]:
                    continue
                vals = [g["values"][(p.id, k)][i].item() for k in keys]
                ranges[shortname[p.id]] = {
                    "type": "NdArray",
                    "dataType": "float",
                    "shape": [len(vals)],
                    "axisNames": ["levelist"],
                    "values": _py_values(vals),
                }
            out.append(
                {
                    "mars:metadata": g["meta"],
                    "type": "Coverage",
                    "domain": {
                        "type": "Domain",
                        "axes": {
                            "latitude": {"values": [g["lat"][i].item()]},
                            "longitude": {"values": [g["lon"][i].item()]},
                            "levelist": {"values": levels},
                            "t": {"values": list(g["t"])},
                        },
                    },
                    "ranges": ranges,
                }
            )
    return out


def trajectory_coverages(groups, header, shortname, coordinates) -> list:
    """Legacy Trajectory layout: one coverage per series; composite ``[t, lat, lon, level]``."""
    out = []
    for series in _series(groups):
        composite = []
        values = {p.id: [] for p in header.parameters}
        for g in series:
            t = g["t"][0] if g["t"] else 0
            keys = list(g["levels"]) if g["levels"] else [None]
            for k in keys:
                level = 0 if k is None else k
                composite.extend([t, la.item(), lo.item(), level] for la, lo in zip(g["lat"], g["lon"]))
                for p in header.parameters:
                    arr = g["values"].get((p.id, k))
                    values[p.id].extend(arr.tolist() if arr is not None else [None] * len(g["lat"]))
        ranges = {
            shortname[p.id]: _range(shortname[p.id], _py_values(values[p.id]))
            for p in header.parameters
            if any(p.id in g["values_params"] for g in series)
        }
        out.append(
            {
                "mars:metadata": series[0]["meta"],
                "type": "Coverage",
                "domain": {
                    "type": "Domain",
                    "axes": {"composite": {"dataType": "tuple", "coordinates": coordinates, "values": composite}},
                },
                "ranges": ranges,
            }
        )
    return out


class _OpenCoverage:
    """State of the MultiPoint coverage being written."""

    __slots__ = ("group", "n_tuples", "coords", "composite_closed", "param", "n_values")

    def __init__(self, group):
        self.group = group
        self.n_tuples = 0
        #: (lat, lon) arrays of every band, kept only when the coverage has more than one level
        self.coords: list = []
        self.composite_closed = False
        self.param = None
        self.n_values = 0


class CovjsonStreamEncoder:
    """CoverageJSON encoder for the polytope-mars block stream (one instance per request)."""

    content_type = "application/prs.coverage+json"
    file_extension = "covjson"

    def __init__(self, config=None, max_fragment_bytes: int = DEFAULT_MAX_FRAGMENT_BYTES):
        self.config = dict(config or {})
        #: upper bound on one fragment of :meth:`encode_iter` (``config["max_fragment_bytes"]`` wins)
        self.max_fragment_bytes = _fragment_limit(self.config.get("max_fragment_bytes") or max_fragment_bytes)
        self._header = None
        self.n_coverages = 0
        self._streaming = True
        self._buffered: list = []
        self._current: _OpenCoverage | None = None

    @property
    def header(self):
        if self._header is None:
            raise RuntimeError("CovjsonStreamEncoder.begin() must be called first")
        return self._header

    def _open(self) -> _OpenCoverage:
        if self._current is None:
            raise RuntimeError("values block before the coordinates of its group")
        return self._current

    # -- protocol ----------------------------------------------------------------------------------------

    def begin(self, header) -> bytes:
        self._header = header
        self.n_coverages = 0
        self._shortname = {p.id: p.shortname for p in header.parameters}
        self._coordinates = legacy_referencing(header)
        self._streaming = header.domain_type == "MultiPoint" or (
            header.domain_type == "Trajectory" and header.time_axis == "hdate"
        )
        self._with_t = header.domain_type == "MultiPoint"
        self._buffered = []
        self._current = None
        domain = "MultiPoint" if header.domain_type in ("shapefile",) else header.domain_type
        return b'{"type": "CoverageCollection", "domainType": ' + _dumps(domain) + b', "coverages": ['

    def encode(self, block) -> bytes:
        """The whole block as one ``bytes`` (``b"".join(self.encode_iter(block))``)."""
        return b"".join(self.encode_iter(block))

    def encode_iter(self, block) -> Iterator[bytes]:
        """The block as fragments of at most ``max_fragment_bytes``, each from a slice of its arrays.

        The fragments of a block must be consumed completely, in order, before the next block is
        encoded: the encoder's state (separators, open ranges) advances as they are produced.
        """
        if hasattr(block, "lat"):
            return self._coords(block)
        if hasattr(block, "values") and hasattr(block, "param"):
            return self._values(block)
        return self._group_end(block)

    def end(self) -> bytes:
        out = []
        if not self._streaming:
            covs = self._buffered_coverages()
            for i, cov in enumerate(covs):
                out.append((b", " if self.n_coverages or i else b"") + _dumps(cov))
            self.n_coverages += len(covs)
            self._buffered = []
        parameters = {p.shortname: _parameter(p) for p in self.header.parameters}
        referencing = [{"coordinates": self._coordinates, "system": dict(_CRS)}]
        out.append(b'], "referencing": ' + _dumps(referencing) + b', "parameters": ' + _dumps(parameters) + b"}")
        return b"".join(out)

    # -- MultiPoint: stream -------------------------------------------------------------------------------

    def _coords(self, block) -> Iterator[bytes]:
        g = block.group
        if not self._streaming:
            cur = self._buffer_group(g)
            cur["lat"].append(np.asarray(block.lat, dtype=np.float64))
            cur["lon"].append(np.asarray(block.lon, dtype=np.float64))
            return
        if self._current is None or self._current.group is not g:
            yield self._open_coverage(g)
        cur = self._open()
        levels = list(g.levels) if g.levels else [0]
        yield from self._tuples(cur, block.lat, block.lon, levels[0])
        if len(levels) > 1:
            cur.coords.append((np.array(block.lat, dtype=np.float64), np.array(block.lon, dtype=np.float64)))

    def _tuples(self, cur: _OpenCoverage, lat, lon, level) -> Iterator[bytes]:
        """The composite tuples of one block of points at one level, sliced into fragments.

        The ``(n, 3)`` tuples are built per slice, so the text of a whole block never exists at once.
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        item = 2 * _FLOAT_CHARS + 8 + len(_dumps(level))  # "[lat, lon, level], "
        for a, b in _fragment_slices(lat.size, item, self.max_fragment_bytes):
            body = composite_tuples(lat[a:b], lon[a:b], level)
            if not body:
                continue
            sep = b", " if cur.n_tuples else b""
            cur.n_tuples += b - a
            yield sep + body

    def _open_coverage(self, g) -> bytes:
        self._current = _OpenCoverage(g)
        head = (b", " if self.n_coverages else b"") + b'{"mars:metadata": ' + _dumps(g.mars_metadata)
        head += b', "type": "Coverage", "domain": {"type": "Domain", "axes": {'
        if self._with_t:
            head += b'"t": {"values": ' + _dumps(list(g.t)) + b"}, "
        head += b'"composite": {"dataType": "tuple", "coordinates": ' + _dumps(self._coordinates) + b', "values": ['
        self.n_coverages += 1
        return head

    def _close_composite(self) -> Iterator[bytes]:
        cur = self._open()
        if cur.composite_closed:
            return
        cur.composite_closed = True
        levels = list(cur.group.levels)
        for level in levels[1:]:
            for lat, lon in cur.coords:
                yield from self._tuples(cur, lat, lon, level)
        cur.coords = []
        yield b']}}}, "ranges": {'

    def _values(self, block) -> Iterator[bytes]:
        g = block.group
        if not self._streaming:
            cur = self._buffer_group(g)
            cur["values"].setdefault((block.param, block.level), []).append(np.asarray(block.values, dtype=np.float64))
            return
        cur = self._open()
        yield from self._close_composite()
        if cur.param != block.param:
            name = self._shortname.get(block.param, block.param)
            n = g.n_points * max(1, len(g.levels))
            head = b"]}, " if cur.param is not None else b""
            head += _dumps(name) + b': {"type": "NdArray", "dataType": "float", "shape": [' + str(n).encode()
            head += b'], "axisNames": [' + _dumps(str(name)) + b'], "values": ['
            cur.param = block.param
            cur.n_values = 0
            yield head
        values = np.asarray(block.values)
        for a, b in _fragment_slices(values.size, _FLOAT_CHARS + 2, self.max_fragment_bytes):
            body = float_list_bytes(values[a:b])
            if not body:
                continue
            sep = b", " if cur.n_values else b""
            cur.n_values += b - a
            yield sep + body

    def _group_end(self, block) -> Iterator[bytes]:
        if not self._streaming:
            cur = self._buffered[-1] if self._buffered else None
            if cur is not None and cur["group"] is block.group:
                cur["lat"] = np.concatenate(cur["lat"]) if cur["lat"] else np.empty(0)
                cur["lon"] = np.concatenate(cur["lon"]) if cur["lon"] else np.empty(0)
                cur["values"] = {k: np.concatenate(v) for k, v in cur["values"].items()}
                cur["done"] = True
            return
        cur = self._current
        if cur is None or cur.group is not block.group:
            return
        yield from self._close_composite()
        yield b"]}}}" if cur.param is not None else b"}}"
        self._current = None

    # -- buffered domains ---------------------------------------------------------------------------------

    def _buffer_group(self, g) -> dict:
        if not self._buffered or self._buffered[-1]["group"] is not g:
            self._buffered.append(
                {
                    "group": g,
                    "meta": g.mars_metadata,
                    "levels": tuple(g.levels),
                    "t": tuple(g.t),
                    "values_params": set(g.params),
                    "lat": [],
                    "lon": [],
                    "values": {},
                    "done": False,
                }
            )
        return self._buffered[-1]

    def _buffered_coverages(self) -> list:
        groups = [g for g in self._buffered if g["done"]]
        if not groups:
            return []
        domain = self.header.domain_type
        if domain == "PointSeries":
            return pointseries_coverages(groups, self.header, self._shortname)
        if domain == "VerticalProfile":
            return verticalprofile_coverages(groups, self.header, self._shortname)
        if domain == "Trajectory":
            return trajectory_coverages(groups, self.header, self._shortname, self._coordinates)
        raise ValueError(f"Unsupported domain type {domain!r}")
