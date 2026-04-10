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
                            "latitude": {"values": [3]},
                            "longitude": {"values": [7]},
                            "levelist": {"values": [1]},
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
                            "latitude": {"values": [3]},
                            "longitude": {"values": [7]},
                            "levelist": {"values": [1]},
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
                "latitude": {"values": [3]},
                "longitude": {"values": [7]},
                "levelist": {"values": [1]},
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
                "latitude": {"values": [3]},
                "longitude": {"values": [7]},
                "levelist": {"values": [1]},
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

    def test_to_xarray_monthly_means(self):
        """to_xarray() must work for monthly-means CovJSON that lacks 'Forecast date'.

        from_polytope_month() packs all time steps into a single coverage's t-axis
        and does not write 'Forecast date' into mars:metadata.  The decoder must
        fall back to reading time values directly from the domain's t-axis.
        """
        import numpy as np

        covjson = {
            "type": "CoverageCollection",
            "domainType": "PointSeries",
            "coverages": [
                {
                    "mars:metadata": {
                        "class": "d1",
                        "stream": "clmn",
                        "levtype": "sfc",
                        "number": 0,
                        "levelist": 0,
                    },
                    "type": "Coverage",
                    "domain": {
                        "type": "Domain",
                        "axes": {
                            "latitude": {"values": [9.896853442816]},
                            "longitude": {"values": [9.84375]},
                            "levelist": {"values": [0]},
                            "t": {
                                "values": [
                                    "2020-02-01T00:00:00Z",
                                    "2020-03-01T00:00:00Z",
                                ]
                            },
                        },
                    },
                    "ranges": {
                        "mean2t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [2],
                            "axisNames": ["mean2t"],
                            "values": [300.34325408935547, 301.6697769165039],
                        }
                    },
                }
            ],
            "referencing": [
                {
                    "coordinates": ["latitude", "longitude", "levelist"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                    },
                }
            ],
            "parameters": {
                "mean2t": {
                    "type": "Parameter",
                    "description": {"en": None},
                    "unit": {"symbol": "K"},
                    "observedProperty": {
                        "id": "mean2t",
                        "label": {"en": "Mean 2 metre temperature"},
                    },
                }
            },
        }

        decoder = Covjsonkit().decode(covjson)
        ds = decoder.to_xarray()

        # Single coverage → single Dataset (not a list)
        assert hasattr(ds, "data_vars"), "Expected an xr.Dataset, not a list"

        # Time dimension has one entry per month
        assert "t" in ds.dims
        assert ds.dims["t"] == 2

        # Both month timestamps are present
        import pandas as pd

        expected_times = pd.to_datetime(["2020-02-01T00:00:00", "2020-03-01T00:00:00"])
        np.testing.assert_array_equal(ds["t"].values, expected_times)

        # Values are correct
        np.testing.assert_allclose(
            ds["mean2t"].values,
            [300.34325408935547, 301.6697769165039],
        )

        # MARS metadata is attached as dataset attributes
        assert ds.attrs["stream"] == "clmn"
        assert ds.attrs["class"] == "d1"
