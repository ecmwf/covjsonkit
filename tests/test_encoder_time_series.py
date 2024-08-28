import random
from datetime import datetime, timedelta

from covjson_pydantic.coverage import CoverageCollection

from covjsonkit.api import Covjsonkit


def get_timestamps(start_dt, end_dt, delta):
    dates = []
    while start_dt <= end_dt:
        # add current date to list by converting  it to iso format
        dates.append(start_dt.isoformat() + "Z")  # .replace("T", " "))
        # increment start date by timedelta
        start_dt += delta
    return dates


class TestEncoder:
    def setup_method(self, method):
        self.covjson = {
            "type": "CoverageCollection",
            "domainType": "PointSeries",
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
                            "x": {"values": [3]},
                            "y": {"values": [7]},
                            "z": {"values": [1]},
                            "t": {
                                "values": [
                                    "2017-01-01 00:00:00",
                                    "2017-01-01 06:00:00",
                                    "2017-01-01 12:00:00",
                                ]
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
                        "date": "20170102",
                        "step": "0",
                        "number": "0",
                    },
                    "type": "Coverage",
                    "domain": {
                        "type": "Domain",
                        "axes": {
                            "x": {"values": [3]},
                            "y": {"values": [7]},
                            "z": {"values": [1]},
                            "t": {
                                "values": [
                                    "2017-01-02 00:00:00",
                                    "2017-01-02 06:00:00",
                                    "2017-01-02 12:00:00",
                                ]
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
                                263.83115234375,
                                265.12313132266,
                                264.93115234375,
                            ],
                        },
                        "p": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [3],
                            "axisNames": ["t"],
                            "values": [
                                13.83115234375,
                                14.12313132266,
                                7.93115234375,
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
                "t": {
                    "type": "Parameter",
                    "description": "Temperature",
                    "unit": {"symbol": "K"},
                    "observedProperty": {"id": "t", "label": {"en": "Temperature"}},
                },
                "p": {
                    "type": "Parameter",
                    "description": "Pressure",
                    "unit": {"symbol": "pa"},
                    "observedProperty": {"id": "p", "label": {"en": "Pressure"}},
                },
            },
        }

    def test_CoverageCollection(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "PointSeries")
        assert encoder_obj.covjson["type"] == "CoverageCollection"

    def test_standard_Coverage(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "PointSeries")
        # covjson = CoverageCollection(
        #    type="CoverageCollection", coverages=[], domainType=DomainType.point_series, parameters={}, referencing=[]
        # )

        covjson = {"type": "CoverageCollection", "domainType": "PointSeries", "coverages": []}
        assert encoder_obj.covjson == covjson

    def test_add_parameter(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "PointSeries")
        encoder_obj.add_parameter(167)
        encoder_obj.add_parameter(166)

        json_string = encoder_obj.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        assert CoverageCollection.model_validate_json(json_string)

    def test_add_reference(self):
        encoder_obj = Covjsonkit().encode("CoverageCollection", "PointSeries")
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

    def test_add_coverage(self):
        encoder = Covjsonkit().encode("CoverageCollection", "PointSeries")
        encoder.add_parameter(167)
        encoder.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        encoder.add_reference({"coordinates": ["t"], "system": {"type": "TemporalRS", "calendar": "Gregorian"}})

        # metadatas = []
        coords = []
        values = []
        for number in range(0, 10):
            metadata = {
                "class": "od",
                "stream": "oper",
                "levtype": "pl",
                "date": "20170101",
                "step": "0",
                "number": str(number),
            }
            timestamps = get_timestamps(
                datetime(2017, 1, 1, 0, 00),
                datetime(2017, 1, 5, 0, 00),
                timedelta(hours=6),
            )
            coord = {
                "x": [3],
                "y": [7],
                "z": [1],
                "t": timestamps,
            }
            coords.append(coord)
            value = {"2t": {0: [random.uniform(230, 270) for _ in range(0, len(timestamps))]}}
            values.append(value)
            encoder.add_coverage(metadata, coord, value)

        json_string = encoder.pydantic_coverage.model_dump_json(exclude_none=True, indent=4)
        assert CoverageCollection.model_validate_json(json_string)

        print(json_string)

    # @pytest.mark.data
    # def test_from_xarray(self):
    #    ds = xr.open_dataset("new_timeseries.nc")
    #    encoder = Covjsonkit().encode("CoverageCollection", "PointSeries")
    #    encoder.from_xarray(ds)
