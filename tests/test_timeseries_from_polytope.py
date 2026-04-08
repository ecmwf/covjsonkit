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
        # EFAS reanalysis: 1 hdate, 1 time, 1 point → 1 coverage, 1 t-value
        # TODO: clarify if time gets absorbed into date by polytope-mars before it
        # reaches covjsonkit, or if it always arrives as a separate axis in the tree
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("time", ("0600",)),
            efas_branch(np.datetime64("2025-07-14"), 51.5, 6.5, [42.17]),
        )

        covjson = Covjsonkit().encode("CoverageCollection", "PointSeries").from_polytope(tree)

        assert len(covjson["coverages"]) == 1
        cov = covjson["coverages"][0]

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
        assert "Forecast date" not in cov["mars:metadata"]
        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_multiple_times(self):
        # 1 hdate, 2 times, 1 point → 1 coverage, 2 t-values
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)
        date.add_child(chain(node("time", ("0600",)), efas_branch(np.datetime64("2025-07-14"), 51.5, 6.5, [42.17])))
        date.add_child(chain(node("time", ("1200",)), efas_branch(np.datetime64("2025-07-14"), 51.5, 6.5, [55.30])))

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

        assert "Forecast date" not in cov["mars:metadata"]
        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_multiple_hdates(self):
        # 2 hdates, 1 time, 1 point → 1 coverage, 2 t-values
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("time", ("0600",)),
        )
        time = tip(tree)
        time.add_child(efas_branch(np.datetime64("2025-07-14"), 51.5, 6.5, [42.17]))
        time.add_child(efas_branch(np.datetime64("2025-07-15"), 51.5, 6.5, [55.30]))

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

        assert "Forecast date" not in cov["mars:metadata"]
        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_multiple_times_and_hdates(self):
        # 2 times × 2 hdates, 1 point → 1 coverage, 4 t-values
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)

        t0600 = node("time", ("0600",))
        t0600.add_child(efas_branch(np.datetime64("2025-07-14"), 51.5, 6.5, [42.17]))
        t0600.add_child(efas_branch(np.datetime64("2025-07-15"), 51.5, 6.5, [55.30]))

        t1200 = node("time", ("1200",))
        t1200.add_child(efas_branch(np.datetime64("2025-07-14"), 51.5, 6.5, [61.44]))
        t1200.add_child(efas_branch(np.datetime64("2025-07-15"), 51.5, 6.5, [73.82]))

        date.add_child(t0600)
        date.add_child(t1200)

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

        assert "Forecast date" not in cov["mars:metadata"]
        assert cov["mars:metadata"] == EXPECTED_REANALYSIS_METADATA

    def test_hdate_two_points(self):
        # 1 time, 1 hdate, 2 points → 2 coverages (one per point)
        tree = chain(
            TensorIndexTree(),
            node("class", ("ce",)),
            node("date", (np.datetime64("2024-03-01"),)),
            node("time", ("0600",)),
            node("hdate", (np.datetime64("2025-07-14"),)),
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

        assert "Forecast date" not in cov0["mars:metadata"]
        assert "Forecast date" not in cov1["mars:metadata"]

    def test_hdate_two_points_two_times(self):
        # 2 times, 1 hdate, 2 points → 2 coverages, each with 2 t-values
        # TODO: in the future we might want MultiPointSeries support to emit a single
        # coverage with a composite spatial axis instead of one coverage per point
        tree = chain(TensorIndexTree(), node("class", ("ce",)), node("date", (np.datetime64("2024-03-01"),)))
        date = tip(tree)

        for time_val in ("0600", "1200"):
            t = node("time", (time_val,))
            branch = chain(
                node("hdate", (np.datetime64("2025-07-14"),)),
                *[node(n, v) for n, v in EFAS_SUFFIX],
            )
            sfo = tip(branch)
            sfo.add_child(make_point(51.5, 6.5, [42.17 if time_val == "0600" else 55.30]))
            sfo.add_child(make_point(52.0, 7.0, [38.91 if time_val == "0600" else 49.62]))
            t.add_child(branch)
            date.add_child(t)

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

        assert "Forecast date" not in cov0["mars:metadata"]
        assert "Forecast date" not in cov1["mars:metadata"]
