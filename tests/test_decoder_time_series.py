import pytest
import json

from eccovjson.decoder import decoder
from eccovjson.decoder import VerticalProfile
from eccovjson.decoder import TimeSeries
from eccovjson.api import Eccovjson
from earthkit import data
import xarray as xr


class TestDecoder:
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

    def test_timeseries_type(self):
        # decoder = TimeSeries.TimeSeries(self.covjson)
        # assert decoder.type == "CoverageCollection"
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        assert decoder.type == "CoverageCollection"

    def test_timeseries_parameters(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        assert decoder.parameters == ["t", "p"]

    def test_timeseries_referencing(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        assert decoder.get_referencing() == ["x", "y", "z"]

    def test_timeseries_mars_metadata(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        metadata1 = {
            "class": "od",
            "stream": "oper",
            "levtype": "pl",
            "date": "20170101",
            "step": "0",
            "number": "0",
        }
        metadata2 = {
            "class": "od",
            "stream": "oper",
            "levtype": "pl",
            "date": "20170102",
            "step": "0",
            "number": "0",
        }
        assert decoder.mars_metadata == [metadata1, metadata2]

    def test_timeseries_domains(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        domain1 = {
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
        }
        assert decoder.domains[0] == domain1
        domain2 = {
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
        }
        assert decoder.domains[1] == domain2

    def test_timeseries_ranges(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        range1 = {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["t"],
                "values": [264.93115234375, 263.83115234375, 265.12313132266],
            },
            "p": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["t"],
                "values": [9.93115234375, 7.83115234375, 14.12313132266],
            },
        }
        assert decoder.ranges[0] == range1
        range2 = {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["t"],
                "values": [263.83115234375, 265.12313132266, 264.93115234375],
            },
            "p": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["t"],
                "values": [13.83115234375, 14.12313132266, 7.93115234375],
            },
        }
        assert decoder.ranges[1] == range2

    def test_timeseries_values(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        values = {
            "t": [
                [264.93115234375, 263.83115234375, 265.12313132266],
                [263.83115234375, 265.12313132266, 264.93115234375],
            ],
            "p": [
                [9.93115234375, 7.83115234375, 14.12313132266],
                [13.83115234375, 14.12313132266, 7.93115234375],
            ],
        }
        assert decoder.get_values() == values

    def test_timeseries_coordinates(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        coordinates = {
            "t": [
                [
                    [3, 7, 1, "20170101", "2017-01-01 00:00:00", "0"],
                    [3, 7, 1, "20170101", "2017-01-01 06:00:00", "0"],
                    [3, 7, 1, "20170101", "2017-01-01 12:00:00", "0"],
                ],
                [
                    [3, 7, 1, "20170102", "2017-01-02 00:00:00", "0"],
                    [3, 7, 1, "20170102", "2017-01-02 06:00:00", "0"],
                    [3, 7, 1, "20170102", "2017-01-02 12:00:00", "0"],
                ],
            ],
            "p": [
                [
                    [3, 7, 1, "20170101", "2017-01-01 00:00:00", "0"],
                    [3, 7, 1, "20170101", "2017-01-01 06:00:00", "0"],
                    [3, 7, 1, "20170101", "2017-01-01 12:00:00", "0"],
                ],
                [
                    [3, 7, 1, "20170102", "2017-01-02 00:00:00", "0"],
                    [3, 7, 1, "20170102", "2017-01-02 06:00:00", "0"],
                    [3, 7, 1, "20170102", "2017-01-02 12:00:00", "0"],
                ],
            ],
        }
        print(decoder.get_coordinates())
        assert decoder.get_coordinates() == coordinates

    def test_timeseries_to_xarray(self):
        decoder = Eccovjson().decode(self.covjson, self.covjson["domainType"])
        ds = decoder.to_xarray()
        # print(ds)
        # print(ds["Temperature"])
        # xrds.to_netcdf("timeseries.nc")
        # ds = xr.open_dataset("timeseries.nc")
        ekds = data.from_object(ds)
        # print(ekds)
        # print(type(ekds))
        # print(ekds.ls())


"""
[<xarray.DataArray 't' (x: 1, y: 1, z: 1, t: 6)>
array([[[[264.93115234, 263.83115234, 265.12313132, 263.83115234,
          265.12313132, 264.93115234]]]])
Coordinates:
  * x        (x) int64 3
  * y        (y) int64 7
  * z        (z) int64 1
  * t        (t) datetime64[ns] 2017-01-01 ... 2017-01-02T12:00:00
Attributes:
    type:       Parameter
    units:      K
    long_name:  Temperature, <xarray.DataArray 'p' (x: 1, y: 1, z: 1, t: 6)>
array([[[[ 9.93115234,  7.83115234, 14.12313132, 13.83115234,
          14.12313132,  7.93115234]]]])
Coordinates:
  * x        (x) int64 3
  * y        (y) int64 7
  * z        (z) int64 1
  * t        (t) datetime64[ns] 2017-01-01 ... 2017-01-02T12:00:00
Attributes:
    type:       Parameter
    units:      pa
    long_name:  Pressure]
    """
