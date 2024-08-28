import pytest
from covjson_pydantic.coverage import CoverageCollection
from covjson_pydantic.domain import DomainType

from covjsonkit.api import Covjsonkit


def get_timestamps(start_dt, end_dt, delta):
    dates = []
    while start_dt <= end_dt:
        # add current date to list by converting  it to iso format
        dates.append(start_dt.isoformat().replace("T", " "))
        # increment start date by timedelta
        start_dt += delta
    return dates


class TestEncoder:
    def setup_method(self, method):
        self.covjson = {
            "type": "CoverageCollection",
            "domainType": "MultiPoint",
            "coverages": [
                {
                    "mars:metadata": {
                        "class": "od",
                        "stream": "oper",
                        "levtype": "pl",
                        "date": "20170101",
                        "step": "0",
                        "number": "0",
                    },
                    "type": "Coverage",
                    "domain": {
                        "type": "Domain",
                        "axes": {
                            "t": {"values": ["2017-01-01T00:00:00"]},
                            "composite": {
                                "dataType": "tuple",
                                "coordinates": ["x", "y", "z"],
                                "values": [[1, 20, 1], [2, 21, 3], [3, 17, 7]],
                            },
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [3],
                            "axisNames": ["t"],
                            "values": [
                                264.93115234375,
                                263.83115234375,
                                265.12313132266,
                            ],
                        },
                        "p": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [3],
                            "axisNames": ["t"],
                            "values": [
                                9.93115234375,
                                7.83115234375,
                                14.12313132266,
                            ],
                        },
                    },
                },
                {
                    "mars:metadata": {
                        "class": "od",
                        "stream": "oper",
                        "levtype": "pl",
                        "date": "20170101",
                        "step": "1",
                        "number": "0",
                    },
                    "type": "Coverage",
                    "domain": {
                        "type": "Domain",
                        "axes": {
                            "t": {"values": ["2017-01-01T01:00:00"]},
                            "composite": {
                                "dataType": "tuple",
                                "coordinates": ["x", "y", "z"],
                                "values": [[1, 20, 1], [2, 21, 3], [3, 17, 7]],
                            },
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [3],
                            "axisNames": ["t"],
                            "values": [
                                266.93115234375,
                                293.83115234375,
                                165.12313132266,
                            ],
                        },
                        "p": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [3],
                            "axisNames": ["t"],
                            "values": [
                                1.93115234375,
                                22.83115234375,
                                12.12313132266,
                            ],
                        },
                    },
                },
            ],
            "referencing": [
                {
                    "coordinates": ["x", "y", "z"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                    },
                }
            ],
            "parameters": {
                "2t": {
                    "type": "Parameter",
                    "description": "Temperature",
                    "unit": {"symbol": "K"},
                    "observedProperty": {"id": "2t", "label": {"en": "Temperature"}},
                },
                "p": {
                    "type": "Parameter",
                    "description": "Pressure",
                    "unit": {"symbol": "pa"},
                    "observedProperty": {"id": "p", "label": {"en": "Pressure"}},
                },
            },
        }

    @pytest.mark.skip(reason="Trajecotry not implemented in covjson-pydantic yet")
    def test_CoverageCollection(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "Path")
        assert encoder_obj.covjson["type"] == "CoverageCollection"

    @pytest.mark.skip(reason="Trajecotry not implemented in covjson-pydantic yet")
    def test_standard_Coverage(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "Path")
        covjson = CoverageCollection(
            type="CoverageCollection", coverages=[], domainType=DomainType.trajectory, parameters={}, referencing=[]
        )

        assert encoder_obj.get_json() == covjson.model_dump_json(exclude_none=True, indent=4)

    @pytest.mark.skip(reason="Trajecotry not implemented in covjson-pydantic yet")
    def test_add_parameter(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "Path")
        encoder_obj.add_parameter(167)
        encoder_obj.add_parameter(166)

        json_string = encoder_obj.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        assert CoverageCollection.model_validate_json(json_string)

    @pytest.mark.skip(reason="Trajecotry not implemented in covjson-pydantic yet")
    def test_add_reference(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "Path")
        encoder_obj.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        encoder_obj.add_reference({"coordinates": ["t"], "system": {"type": "TemporalRS", "calendar": "Gregorian"}})

        json_string = encoder_obj.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        assert CoverageCollection.model_validate_json(json_string)

    @pytest.mark.skip(reason="Trajecotry not implemented in covjson-pydantic yet")
    def test_add_coverage(self):
        encoder = Covjsonkit().encode("CoverageCollection", "Path")
        encoder.add_parameter(167)
        encoder.add_reference({"coordinates": ["t"], "system": {"type": "TemporalRS", "calendar": "Gregorian"}})
        encoder.add_reference(
            {
                "coordinates": ["x", "y"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        # metadatas = []
        coords = []
        # values = []

        metadata = {
            "class": "od",
            "stream": "oper",
            "levtype": "pl",
            "date": "20170101",
            "step": "0",
            "number": 0,
        }
        coords = {}
        coords["composite"] = [
            ["2017-01-01T00:00:00Z", 20, 1],
            ["2017-01-01T00:00:00Z", 21, 3],
            ["2017-01-01T00:00:00Z", 17, 7],
        ]
        value = {"2t": [111, 222, 333]}
        encoder.add_coverage(metadata, coords, value)

        json_string = encoder.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        assert CoverageCollection.model_validate_json(json_string)
        print(json_string)

    """

    def test_from_xarray(self):
        ds = xr.open_dataset("new_timeseries.nc")
        encoder = Covjsonkit().encode("CoverageCollection", "PointSeries")
        encoder.from_xarray(ds)

        """
