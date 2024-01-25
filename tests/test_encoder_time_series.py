import random
from datetime import datetime, timedelta

import pytest
import xarray as xr

from eccovjson.api import Eccovjson


def get_timestamps(start_dt, end_dt, delta):
    dates = []
    while start_dt <= end_dt:
        # add current date to list by converting  it to iso format
        dates.append(start_dt.isoformat().replace("T", " "))
        # increment start date by timedelta
        start_dt += delta
    return dates


class TestEecoder:
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
        encoder_obj = Eccovjson().encode("CoverageCollection", "PointSeries")
        assert encoder_obj.type == "CoverageCollection"

    def test_standard_Coverage(self):
        encoder_obj = Eccovjson().encode("CoverageCollection", "PointSeries")
        covjson = {
            "type": "CoverageCollection",
            "domainType": "PointSeries",
            "coverages": [],
            "referencing": [],
            "parameters": {},
        }

        assert encoder_obj.covjson == covjson

    def test_add_parameter(self):
        encoder_obj = Eccovjson().encode("CoverageCollection", "PointSeries")
        encoder_obj.add_parameter("t")
        encoder_obj.add_parameter("tp")
        print(encoder_obj.covjson)
        covjson = {
            "type": "CoverageCollection",
            "domainType": "PointSeries",
            "coverages": [],
            "referencing": [],
            "parameters": {
                "t": {
                    "type": "Parameter",
                    "description": "Temperature",
                    "unit": {"symbol": "K"},
                    "observedProperty": {
                        "id": "t",
                        "label": {"en": "Temperature"},
                    },
                },
                "tp": {
                    "type": "Parameter",
                    "description": "Total Precipitation",
                    "unit": {"symbol": "m"},
                    "observedProperty": {
                        "id": "tp",
                        "label": {"en": "Total Precipitation"},
                    },
                },
            },
        }

        assert encoder_obj.covjson == covjson

    def test_add_reference(self):
        encoder_obj = Eccovjson().encode("CoverageCollection", "PointSeries")
        encoder_obj.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )
        covjson = {
            "type": "CoverageCollection",
            "domainType": "PointSeries",
            "coverages": [],
            "referencing": [
                {
                    "coordinates": ["x", "y", "z"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                    },
                }
            ],
            "parameters": {},
        }

        assert encoder_obj.covjson == covjson

    def test_add_coverage(self):
        encoder = Eccovjson().encode("CoverageCollection", "PointSeries")
        encoder.add_parameter("t")
        encoder.add_reference(
            {
                "coordinates": ["x", "y", "z"],
                "system": {
                    "type": "GeographicCRS",
                    "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
            }
        )

        # metadatas = []
        coords = []
        values = []
        for number in range(0, 50):
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
                datetime(2017, 1, 14, 0, 00),
                timedelta(hours=6),
            )
            coord = {
                "x": [3],
                "y": [7],
                "z": [1],
                "t": timestamps,
            }
            coords.append(coord)
            value = {"t": [random.uniform(230, 270) for _ in range(0, len(timestamps))]}
            values.append(value)
            encoder.add_coverage(metadata, coord, value)
            # print(encoder.covjson)

    @pytest.mark.data
    def test_from_xarray(self):
        ds = xr.open_dataset("new_timeseries.nc")
        encoder = Eccovjson().encode("CoverageCollection", "PointSeries")
        encoder.from_xarray(ds)


"""
    def test_add_coverage_marsmetadata(self):
        encoder_obj = TimeSeries.TimeSeries("CoverageCollection", "PointSeries")
        encoder_obj.add_coverage(
            {
                "class": "od",
                "stream": "oper",
                "levtype": "pl",
                "date": "20170101",
                "step": "0",
                "number": "0",
            },
            {},
            {},
        )
        covjson = {
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
                            "x": {"values": []},
                            "y": {"values": []},
                            "z": {"values": []},
                            "t": {"values": []},
                        },
                    },
                    "ranges": {},
                }
            ],
            "referencing": [],
            "parameters": {},
        }
        print(encoder_obj.covjson)
        assert encoder_obj.covjson == covjson
"""
