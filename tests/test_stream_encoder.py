"""CovjsonStreamEncoder on synthetic blocks (no polytope needed).

The blocks are local stand-ins with the attributes polytope-mars' block IR exposes; the encoder only
reads attributes.  Expected bytes are ``json.dumps`` of the document the legacy encoders built.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from covjsonkit.stream import (
    CovjsonStreamEncoder,
    composite_tuples,
    float_list_bytes,
    format_floats,
    legacy_referencing,
)


@dataclass(frozen=True)
class Param:
    id: str
    shortname: str
    name: str
    unit: str
    description: str = ""


@dataclass(frozen=True)
class Header:
    feature_type: str
    domain_type: str
    time_axis: str
    parameters: tuple
    mars_metadata: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Group:
    index: int
    path: dict
    t: tuple
    params: tuple
    levels: tuple
    n_points: int
    n_bands: int
    mars_metadata: dict


@dataclass(frozen=True)
class Coords:
    group: Any
    band: int
    offset: int
    lat: np.ndarray
    lon: np.ndarray


@dataclass(frozen=True)
class Values:
    group: Any
    param: str
    level: Any
    band: int
    offset: int
    values: np.ndarray


@dataclass(frozen=True)
class End:
    group: Any


T2 = Param("167", "2t", "2 metre temperature", "K", "Température à 2 m")
U10 = Param("165", "10u", "10 metre U wind component", "m s**-1")
CRS = {"type": "GeographicCRS", "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}


def _parameters(params):
    return {
        p.shortname: {
            "type": "Parameter",
            "description": {"en": p.description},
            "unit": {"symbol": p.unit},
            "observedProperty": {"id": p.shortname, "label": {"en": p.name}},
        }
        for p in params
    }


def _nan_to_none(values):
    return [None if v != v else v for v in np.asarray(values, dtype=np.float64).tolist()]


def multipoint_blocks(fields, lat, lon, levels=(), meta=None, t=("2024-01-01T00:00:00Z",), n_bands=1, index=0):
    """Blocks of one MultiPoint group; ``fields`` = {(param, level): values}."""
    params = tuple(dict.fromkeys(p for p, _ in fields))
    bounds = np.array_split(np.arange(len(lat)), n_bands)
    g = Group(index, {}, t, params, tuple(levels), len(lat), n_bands, meta or {"class": "od", "number": 0})
    starts = [idx[0].item() if len(idx) else 0 for idx in bounds]
    blocks: list[Any] = [Coords(g, b, starts[b], lat[idx], lon[idx]) for b, idx in enumerate(bounds)]
    for p in params:
        for lev in levels or (None,):
            for b, idx in enumerate(bounds):
                blocks.append(Values(g, p, lev, b, starts[b], fields[(p, lev)][idx]))
    blocks.append(End(g))
    return blocks


def encode(header, blocks) -> bytes:
    enc = CovjsonStreamEncoder({"param_db": "ecmwf"})
    out = [enc.begin(header)]
    out += [enc.encode(b) for b in blocks]
    out.append(enc.end())
    return b"".join(out)


def legacy_multipoint(header, groups):
    """What BoundingBox.from_polytope + json.dumps produced for the same data."""
    coords = legacy_referencing(header)
    covs = []
    for fields, lat, lon, levels, meta, t in groups:
        composite = [[float(a), float(o), lev] for lev in (levels or (0,)) for a, o in zip(lat, lon)]
        ranges = {}
        params = tuple(dict.fromkeys(p for p, _ in fields))
        short = {p.id: p.shortname for p in header.parameters}
        for p in params:
            vals = [v for lev in (levels or (None,)) for v in _nan_to_none(fields[(p, lev)])]
            ranges[short[p]] = {
                "type": "NdArray",
                "dataType": "float",
                "shape": [len(vals)],
                "axisNames": [short[p]],
                "values": vals,
            }
        covs.append(
            {
                "mars:metadata": meta,
                "type": "Coverage",
                "domain": {
                    "type": "Domain",
                    "axes": {
                        "t": {"values": list(t)},
                        "composite": {"dataType": "tuple", "coordinates": coords, "values": composite},
                    },
                },
                "ranges": ranges,
            }
        )
    doc = {
        "type": "CoverageCollection",
        "domainType": "MultiPoint",
        "coverages": covs,
        "referencing": [{"coordinates": coords, "system": CRS}],
        "parameters": _parameters(header.parameters),
    }
    return json.dumps(doc).encode()


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.round(rng.uniform(-90, 90, n), 12)
    lon = np.round(rng.uniform(0, 360, n), 12)
    return lat, lon


BBOX = Header("boundingbox", "MultiPoint", "date", (U10, T2))


def test_multipoint_matches_legacy_json_dumps():
    lat, lon = _data(7)
    fields = {("165", None): np.linspace(1, 7, 7) * 1.1e10 + 0.123, ("167", None): np.arange(7) * 0.5}
    meta = {"class": "od", "Forecast date": "2024-01-01T00:00:00Z", "number": 0, "step": 0}
    out = encode(BBOX, multipoint_blocks(fields, lat, lon, meta=meta))
    expected = legacy_multipoint(BBOX, [(fields, lat, lon, (), meta, ("2024-01-01T00:00:00Z",))])
    assert out == expected
    json.loads(out)


@pytest.mark.parametrize("n_bands", [1, 2, 3, 13])
def test_band_size_invariance(n_bands):
    lat, lon = _data(13, 1)
    levels = ("500", "850")
    fields = {(p, lev): np.random.default_rng(2).normal(size=13) * 1e3 for p in ("165", "167") for lev in levels}
    ref = encode(BBOX, multipoint_blocks(fields, lat, lon, levels=levels, n_bands=1))
    assert encode(BBOX, multipoint_blocks(fields, lat, lon, levels=levels, n_bands=n_bands)) == ref
    group = (fields, lat, lon, levels, {"class": "od", "number": 0}, ("2024-01-01T00:00:00Z",))
    expected = legacy_multipoint(BBOX, [group])
    assert ref == expected


def test_nan_is_null_and_missing_param_is_omitted():
    lat, lon = _data(4, 3)
    vals = np.array([1.0, np.nan, 3.0, np.nan])
    out = encode(BBOX, multipoint_blocks({("167", None): vals}, lat, lon))
    doc = json.loads(out)
    assert b"NaN" not in out
    ranges = doc["coverages"][0]["ranges"]
    assert list(ranges) == ["2t"]  # 10u not in group.params -> no range
    assert ranges["2t"]["values"] == [1.0, None, 3.0, None]
    assert list(doc["parameters"]) == ["10u", "2t"]  # header parameters are always listed


def test_several_groups_and_empty_collection():
    lat, lon = _data(3, 4)
    blocks = multipoint_blocks({("167", None): np.ones(3)}, lat, lon, index=0)
    blocks += multipoint_blocks({("167", None): np.zeros(3)}, lat, lon, index=1, t=("2024-01-02T00:00:00Z",))
    doc = json.loads(encode(BBOX, blocks))
    assert [c["domain"]["axes"]["t"]["values"] for c in doc["coverages"]] == [
        ["2024-01-01T00:00:00Z"],
        ["2024-01-02T00:00:00Z"],
    ]
    empty = json.loads(encode(BBOX, []))
    assert empty["coverages"] == [] and list(empty) == ["type", "domainType", "coverages", "referencing", "parameters"]


def test_float_formatting_matches_json_dumps():
    rng = np.random.default_rng(5)
    vals = np.concatenate(
        [
            rng.normal(size=5000) * 10.0 ** rng.integers(-12, 20, 5000),
            [0.0, -0.0, 1.0, 1e16, 1e-4, 9.99e-5, 2e-5, 1e-7, -3e-9, 123456789.123, 5e-324, 1.7976931348623157e308],
        ]
    )
    assert b", ".join(format_floats(vals)) == json.dumps(vals.tolist()).encode()[1:-1]
    assert format_floats(np.array([np.nan, np.inf])) == [b"null", b"null"]


@pytest.mark.parametrize("small", [False, True], ids=["fast-path", "repr-fallback"])
@pytest.mark.parametrize("level", [0, "500", 850])
def test_composite_and_value_bytes_match_json_dumps(small, level):
    lat, lon = _data(2000, 6)
    lon[3] = -0.0
    if small:
        lon[7] = 3e-5  # forces the per-value path
    pairs = [[a, o, level] for a, o in zip(lat.tolist(), lon.tolist())]
    assert composite_tuples(lat, lon, level) == json.dumps(pairs).encode()[1:-1]
    values = np.concatenate([lat * 1e9, [np.nan, 2e-7 if small else 2.0]])
    expected = json.dumps([None if v != v else v for v in values.tolist()]).encode()[1:-1]
    assert float_list_bytes(values) == expected
    assert composite_tuples(np.empty(0), np.empty(0), level) == b""


@pytest.mark.parametrize(
    "feature,role,coords",
    [
        ("boundingbox", "date", ["latitude", "longitude", "levelist"]),
        ("polygon", "date", ["x", "y", "z"]),
        ("polygon", "step", ["x", "y", "z"]),
        ("polygon", "hdate", ["latitude", "longitude", "levelist"]),
        ("boundingbox", "month", ["x", "y", "z"]),
        ("circle", "month", ["latitude", "longitude", "levelist"]),
    ],
)
def test_multipoint_referencing_quirks(feature, role, coords):
    referencing = legacy_referencing(Header(feature, "MultiPoint", role, ()))
    assert referencing == coords


def _point_group(index, t, meta, values, levels=(), lat=(51.5, -33.9), lon=(0.1, 18.4)):
    params = tuple(dict.fromkeys(p for p, _ in values))
    g = Group(index, {}, t, params, tuple(levels), len(lat), 1, meta)
    blocks: list[Any] = [Coords(g, 0, 0, np.array(lat), np.array(lon))]
    for (p, lev), v in values.items():
        blocks.append(Values(g, p, lev, 0, 0, np.asarray(v, dtype=float)))
    return blocks + [End(g)]


def test_pointseries_layout_is_point_major_over_consecutive_groups():
    header = Header("timeseries", "PointSeries", "date", (T2,))
    meta = {"class": "od", "number": 0, "levelist": 0}
    blocks = []
    for i, t in enumerate(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"]):
        blocks += _point_group(i, (t,), meta, {("167", None): [i, 10 + i]})
    doc = json.loads(encode(header, blocks))
    assert len(doc["coverages"]) == 2
    first, second = doc["coverages"]
    assert first["domain"]["axes"]["t"]["values"] == [
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T02:00:00Z",
    ]
    assert first["ranges"]["2t"]["values"] == [0.0, 1.0, 2.0]
    assert second["ranges"]["2t"]["values"] == [10.0, 11.0, 12.0]
    assert second["domain"]["axes"]["latitude"]["values"] == [-33.9]
    assert doc["referencing"][0]["coordinates"] == ["latitude", "longitude", "levelist"]


def test_verticalprofile_layout():
    header = Header("verticalprofile", "VerticalProfile", "date", (T2,))
    blocks = _point_group(
        0, ("2024-01-01T00:00:00Z",), {"number": 0}, {("167", 500): [1, 2], ("167", 850): [3, 4]}, levels=(500, 850)
    )
    doc = json.loads(encode(header, blocks))
    assert [c["ranges"]["2t"]["values"] for c in doc["coverages"]] == [[1.0, 3.0], [2.0, 4.0]]
    assert doc["coverages"][0]["domain"]["axes"]["levelist"]["values"] == [500, 850]
    assert doc["coverages"][0]["ranges"]["2t"]["axisNames"] == ["levelist"]


def test_trajectory_layout():
    header = Header("trajectory", "Trajectory", "date", (T2,))
    blocks = _point_group(0, (0,), {"number": 0}, {("167", None): [1, 2]})
    doc = json.loads(encode(header, blocks))
    (cov,) = doc["coverages"]
    assert cov["domain"]["axes"]["composite"]["values"] == [[0, 51.5, 0.1, 0], [0, -33.9, 18.4, 0]]
    assert cov["domain"]["axes"]["composite"]["coordinates"] == ["t", "x", "y", "z"]


def test_no_polytope_imports():
    code = (
        "import sys, covjsonkit.stream; "
        "roots = ('polytope_feature', 'polytope_mars', 'covjson_pydantic'); "
        "bad = [m for m in sys.modules if m.split('.')[0] in roots]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
