# from earthkit import data

from covjsonkit.api import Covjsonkit


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
        decoder = Covjsonkit().decode(self.covjson)
        assert decoder.type == "CoverageCollection"

    def test_timeseries_parameters(self):
        decoder = Covjsonkit().decode(self.covjson)
        assert decoder.parameters == ["t", "p"]

    def test_timeseries_referencing(self):
        decoder = Covjsonkit().decode(self.covjson)
        assert decoder.get_referencing() == ["x", "y", "z"]

    def test_timeseries_mars_metadata(self):
        decoder = Covjsonkit().decode(self.covjson)
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
        decoder = Covjsonkit().decode(self.covjson)
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
        decoder = Covjsonkit().decode(self.covjson)
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
        decoder = Covjsonkit().decode(self.covjson)
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
        decoder = Covjsonkit().decode(self.covjson)
        coordinates = {
            "t": [
                [
                    [3, 7, 1, "2017-01-01 00:00:00", "2017-01-01 00:00:00", "0"],
                    [3, 7, 1, "2017-01-01 00:00:00", "2017-01-01 06:00:00", "0"],
                    [3, 7, 1, "2017-01-01 00:00:00", "2017-01-01 12:00:00", "0"],
                ],
                [
                    [3, 7, 1, "2017-01-02 00:00:00", "2017-01-02 00:00:00", "0"],
                    [3, 7, 1, "2017-01-02 00:00:00", "2017-01-02 06:00:00", "0"],
                    [3, 7, 1, "2017-01-02 00:00:00", "2017-01-02 12:00:00", "0"],
                ],
            ],
            "p": [
                [
                    [3, 7, 1, "2017-01-01 00:00:00", "2017-01-01 00:00:00", "0"],
                    [3, 7, 1, "2017-01-01 00:00:00", "2017-01-01 06:00:00", "0"],
                    [3, 7, 1, "2017-01-01 00:00:00", "2017-01-01 12:00:00", "0"],
                ],
                [
                    [3, 7, 1, "2017-01-02 00:00:00", "2017-01-02 00:00:00", "0"],
                    [3, 7, 1, "2017-01-02 00:00:00", "2017-01-02 06:00:00", "0"],
                    [3, 7, 1, "2017-01-02 00:00:00", "2017-01-02 12:00:00", "0"],
                ],
            ],
        }
        print(decoder.get_coordinates())
        assert decoder.get_coordinates() == coordinates

    def test_timeseries_to_xarray(self):
        # decoder = Covjsonkit().decode(self.covjson)
        # ds = decoder.to_xarray()
        # print(ds)
        # print(ds["Temperature"])
        # xrds.to_netcdf("timeseries.nc")
        # ds = xr.open_dataset("timeseries.nc")
        # ekds = data.from_object(ds)
        # print(ekds)
        # print(type(ekds))
        # print(ekds.ls())
        pass
