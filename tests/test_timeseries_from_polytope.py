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


class TestTimeseriesFromPolytope:

    def test_standard_forecast_single_point(self):
        # od/oper/fc/sfc, 1 point, param 167 (2t), steps 0 and 6
        # axis ordering from a real polytope pprint
        leaf = node("longitude", (11.0,))
        leaf.result = [np.float64(264.931), np.float64(263.831)]

        tree = chain(
            TensorIndexTree(),  # root
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
