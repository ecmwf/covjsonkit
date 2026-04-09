import numpy as np

from polytope_feature.datacube.datacube_axis import IntDatacubeAxis
from polytope_feature.datacube.tensor_index_tree import TensorIndexTree

from covjsonkit.api import Covjsonkit


def node(name, values):
    ax = IntDatacubeAxis()
    ax.name = name
    return TensorIndexTree(axis=ax, values=tuple(values))


def chain(*nodes):
    for a, b in zip(nodes, nodes[1:]):
        a.add_child(b)
    return nodes[0]


def tip(tree):
    while tree.children:
        tree = tree.children[0]
    return tree


def make_leaf(lon, result):
    leaf = node("longitude", (lon,))
    leaf.result = [np.float64(r) for r in result]
    return leaf


def make_point(lat, lon, result):
    lat_n = node("latitude", (lat,))
    lat_n.add_child(make_leaf(lon, result))
    return lat_n


# Axis ordering for EFAS reanalysis (between hdate and latitude in the tree)
EFAS_SUFFIX = [
    ("domain", ("g",)),
    ("expver", ("4321",)),
    ("levtype", ("sfc",)),
    ("model", ("lisflood",)),
    ("origin", ("ecmf",)),
    ("param", ("240023",)),
    ("step", (6,)),
    ("stream", ("efcl",)),
    ("type", ("sfo",)),
]


def efas_branch(hdate, lat, lon, result):
    """hdate → [EFAS_SUFFIX axes] → lat → lon(leaf). Single-point branch."""
    return chain(
        node("hdate", (hdate,)),
        *[node(n, v) for n, v in EFAS_SUFFIX],
        make_point(lat, lon, result),
    )


EXPECTED_REANALYSIS_METADATA = {
    "class": "ce",
    "date": "2024-03-01",
    "domain": "g",
    "expver": "4321",
    "levtype": "sfc",
    "model": "lisflood",
    "origin": "ecmf",
    "stream": "efcl",
    "type": "sfo",
    "number": 0,
    "levelist": 0,
}


class TestTimeseriesFromPolytope:

    def test_standard_forecast_single_point(self):
        # od/oper/fc/sfc, 1 point, param 167 (2t), steps 0 and 6
        # axis ordering from a real polytope pprint
        leaf = make_leaf(11.0, [264.931, 263.831])

        tree = chain(
            TensorIndexTree(),
            node("class", ("od",)),
            node("date", (np.datetime64("2025-01-01T00:00:00"),)),
            node("domain", ("g",)),
            node("expver", ("0001",)),
            node("levtype", ("sfc",)),
            node("param", ("167",)),
            node("step", (0, 6)),
            node("stream", ("oper",)),
            node("type", ("fc",)),
            node("latitude", (48.0,)),
            leaf,
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [48.0]},
            "longitude": {"values": [11.0]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z"]},
        }

        assert cov["ranges"] == {
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["2t"],
                "values": [264.931, 263.831],
            }
        }

        assert cov["mars:metadata"] == {
            "class": "od",
            "Forecast date": "2025-01-01T00:00:00Z",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "number": 0,
            "levelist": 0,
        }

    def test_hdate_reanalysis_single_point(self):
        # EFAS reanalysis: 1 hdate (with time pre-merged by polytope-mars), 1 point
        # polytope-mars merges time into hdate: pd.Timestamp("20250714T0600") → np.datetime64
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            efas_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        # t = hdate(2025-07-14T06:00) + step(6h) = 2025-07-14T12:00:00Z
        assert cov["domain"]["axes"] == {
            "latitude": {"values": [51.5]},
            "longitude": {"values": [6.5]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-07-14T12:00:00Z"]},
        }

        assert cov["ranges"] == {
            "dis06": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [1],
                "axisNames": ["dis06"],
                "values": [42.17],
            }
        }

        # TODO: does "Forecast date" make sense here? this is reanalysis, not a forecast.
        # hdate is a hindcast reference date. might drop it or rename. discuss w/ Adam
        # assert "Forecast date" not in cov["mars:metadata"]
        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_multiple_times(self):
        # 1 hdate × 2 times (pre-merged into 2 hdate values), 1 point → 1 coverage, 2 t-values
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(efas_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(efas_branch(np.datetime64("2025-07-14T12:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [51.5]},
            "longitude": {"values": [6.5]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"]},
        }

        assert cov["ranges"] == {
            "dis06": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["dis06"],
                "values": [42.17, 55.30],
            }
        }

        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_multiple_hdates(self):
        # 2 hdates × 1 time (pre-merged), 1 point → 1 coverage, 2 t-values
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(efas_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(efas_branch(np.datetime64("2025-07-15T06:00:00"), 51.5, 6.5, [55.30]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [51.5]},
            "longitude": {"values": [6.5]},
            "levelist": {"values": [0]},
            "t": {"values": ["2025-07-14T12:00:00Z", "2025-07-15T12:00:00Z"]},
        }

        assert cov["ranges"] == {
            "dis06": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2],
                "axisNames": ["dis06"],
                "values": [42.17, 55.30],
            }
        }

        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_multiple_times_and_hdates(self):
        # 2 hdates × 2 times (pre-merged into 4 hdate values), 1 point → 1 coverage, 4 t-values
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(efas_branch(np.datetime64("2025-07-14T06:00:00"), 51.5, 6.5, [42.17]))
        date.add_child(efas_branch(np.datetime64("2025-07-14T12:00:00"), 51.5, 6.5, [55.30]))
        date.add_child(efas_branch(np.datetime64("2025-07-15T06:00:00"), 51.5, 6.5, [61.44]))
        date.add_child(efas_branch(np.datetime64("2025-07-15T12:00:00"), 51.5, 6.5, [73.82]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

        assert cov["domain"]["axes"] == {
            "latitude": {"values": [51.5]},
            "longitude": {"values": [6.5]},
            "levelist": {"values": [0]},
            "t": {
                "values": [
                    "2025-07-14T12:00:00Z",
                    "2025-07-15T12:00:00Z",
                    "2025-07-14T18:00:00Z",
                    "2025-07-15T18:00:00Z",
                ]
            },
        }

        assert cov["ranges"] == {
            "dis06": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [4],
                "axisNames": ["dis06"],
                "values": [42.17, 55.30, 61.44, 73.82],
            }
        }

        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_two_points(self):
        # 1 hdate (pre-merged with time), 2 points → 2 coverages (one per point)
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("hdate", (np.datetime64("2025-07-14T06:00:00"),)),
            *[node(n, v) for n, v in EFAS_SUFFIX],
        )
        sfo = tip(tree)
        sfo.add_child(make_point(51.5, 6.5, [42.17]))
        sfo.add_child(make_point(52.0, 7.0, [38.91]))

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov0["domain"]["axes"]["longitude"]["values"] == [6.5]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov1["domain"]["axes"]["longitude"]["values"] == [7.0]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [38.91]

        assert cov0["mars:metadata"] == EXPECTED_REANALYSIS_METADATA
        assert cov1["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_two_points_two_times(self):
        # 2 hdate values (= 1 hdate × 2 times pre-merged), 2 points → 2 coverages, each with 2 t-values
        # TODO: in the future we might want MultiPointSeries support to emit a single
        # coverage with a composite spatial axis instead of one coverage per point
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)

        for hdate_val, vals in [
            (np.datetime64("2025-07-14T06:00:00"), [42.17, 38.91]),
            (np.datetime64("2025-07-14T12:00:00"), [55.30, 49.62]),
        ]:
            branch = chain(
                node("hdate", (hdate_val,)),
                *[node(n, v) for n, v in EFAS_SUFFIX],
            )
            sfo = tip(branch)
            sfo.add_child(make_point(51.5, 6.5, [vals[0]]))
            sfo.add_child(make_point(52.0, 7.0, [vals[1]]))
            date.add_child(branch)

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 2

        cov0 = covjson["coverages"][0]
        assert cov0["domain"]["axes"]["latitude"]["values"] == [51.5]
        assert cov0["domain"]["axes"]["longitude"]["values"] == [6.5]
        assert cov0["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"]
        assert cov0["ranges"]["dis06"]["values"] == [42.17, 55.30]

        cov1 = covjson["coverages"][1]
        assert cov1["domain"]["axes"]["latitude"]["values"] == [52.0]
        assert cov1["domain"]["axes"]["longitude"]["values"] == [7.0]
        assert cov1["domain"]["axes"]["t"]["values"] == ["2025-07-14T12:00:00Z", "2025-07-14T18:00:00Z"]
        assert cov1["ranges"]["dis06"]["values"] == [38.91, 49.62]

        assert cov0["mars:metadata"] == EXPECTED_REANALYSIS_METADATA
        assert cov1["mars:metadata"] == EXPECTED_REANALYSIS_METADATA
