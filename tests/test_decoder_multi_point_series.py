import copy
import json
from pathlib import Path

import pytest

from covjsonkit.api import Covjsonkit


@pytest.fixture
def covjson():
    path = Path(__file__).parent / "data/test_multipointseries_coverage.json"
    return json.loads(path.read_text())


def test_api_decodes_multipointseries(covjson):
    decoder = Covjsonkit().decode(covjson)

    assert decoder.__class__.__name__ == "MultiPointSeries"
    assert decoder.type == "CoverageCollection"
    assert decoder.parameters == ["t"]


def test_preserves_domains_ranges_values_and_coordinates(covjson):
    decoder = Covjsonkit().decode(covjson)

    assert decoder.domains == [coverage["domain"] for coverage in covjson["coverages"]]
    assert decoder.ranges == [coverage["ranges"] for coverage in covjson["coverages"]]
    assert decoder.get_values() == {
        "t": [
            [100.0, 101.0, 102.0, 103.0],
            [110.0, 111.0, 112.0, 113.0],
            [200.0, 201.0, 202.0, 203.0],
            [210.0, 211.0, 212.0, 213.0],
        ]
    }
    assert decoder.get_coordinates()["t"][0] == [
        [50.0, 10.0, None, "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", 0],
        [60.0, 20.0, None, "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", 0],
        [50.0, 10.0, None, "2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z", 0],
        [60.0, 20.0, None, "2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z", 0],
    ]


def test_three_dimensional_tuples_add_levelist_coordinate(covjson):
    covjson = copy.deepcopy(covjson)
    for coverage in covjson["coverages"]:
        composite = coverage["domain"]["axes"]["composite"]
        composite["coordinates"].append("z")
        composite["values"] = [point + [850.0] for point in composite["values"]]
    covjson["referencing"].append({"coordinates": ["z"], "system": {"type": "VerticalCRS"}})

    dataset = Covjsonkit().decode(covjson).to_xarray()

    assert dataset.levelist.dims == ("points",)
    assert dataset.levelist.values.tolist() == [850.0, 850.0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda coverage: coverage["domain"]["axes"]["composite"]["values"].__setitem__(0, [10.0]),
            "tuple width",
        ),
        (lambda coverage: coverage["ranges"]["t"].__setitem__("shape", [2, 1]), "shape"),
        (lambda coverage: coverage["ranges"]["t"].__setitem__("values", [100.0]), "value count"),
    ],
)
def test_rejects_invalid_composite_ranges(covjson, mutate, message):
    mutate(covjson["coverages"][0])

    with pytest.raises(ValueError, match=message):
        Covjsonkit().decode(covjson)


def test_geotiff_is_unsupported(covjson):
    with pytest.raises(TypeError, match="GeoTIFF"):
        Covjsonkit().decode(covjson).to_geotiff()
