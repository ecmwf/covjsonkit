from eccovjson.api import Eccovjson


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
        encoder_obj = Eccovjson().encode("CoverageCollection", "wkt")
        assert encoder_obj.type == "CoverageCollection"

    def test_standard_Coverage(self):
        encoder_obj = Eccovjson().encode("CoverageCollection", "wkt")
        covjson = {
            "type": "CoverageCollection",
            "domainType": "MultiPoint",
            "coverages": [],
            "referencing": [],
            "parameters": {},
        }

        assert encoder_obj.covjson == covjson

    def test_add_parameter(self):
        encoder_obj = Eccovjson().encode("CoverageCollection", "wkt")
        encoder_obj.add_parameter(167)
        encoder_obj.add_parameter(166)
        covjson = {
            "type": "CoverageCollection",
            "domainType": "MultiPoint",
            "coverages": [],
            "referencing": [],
            "parameters": {
                "10v": {
                    "type": "Parameter",
                    "description": (
                        "This parameter is the northward component of the 10m wind."
                        " It is the horizontal speed of air moving towards the north"
                        ", at a height of ten metres"
                        " above the surface of the Earth, in metres per second."
                        "<br/><br/>Care should be taken when"
                        " comparing this parameter with observations, because"
                        " wind observations vary on small space"
                        " and time scales and are affected by the local terrain,"
                        " vegetation and buildings that are"
                        " represented only on average in the ECMWF Integrated Forecasting"
                        " System.<br/><br/>This parameter"
                        " can be combined with the U component of 10m wind to give "
                        "the speed and direction of the horizontal"
                        " 10m wind."
                    ),
                    "unit": {"symbol": "m s**-1"},
                    "observedProperty": {"id": "10v", "label": {"en": "10 metre V wind component"}},
                },
                "2t": {
                    "type": "Parameter",
                    "description": (
                        "This parameter is the temperature of air at 2m above the surface of land,"
                        " sea or in-land waters.<br/><br/>2m temperature is calculated by "
                        "interpolating between the lowest"
                        " model level and the Earth's surface, taking account of the "
                        "atmospheric conditions."
                        "<a href='https://www.ecmwf.int/sites/default/files/elibrar"
                        "y/2016/17117-part-iv-physical-processes.pdf#subsection.3.10.3'>"
                        " See further information </a>.<br/><br/>This parameter has "
                        "units of kelvin (K). Temperature measured in kelvin"
                        " can be converted to degrees Celsius (°C) by subtracting 273.15."
                        "<br/><br/>Please note that the encodings listed"
                        " here for s2s & uerra (which includes encodings for carra/cerra) "
                        "include entries for Mean 2 metre temperature."
                        " The specific encoding for Mean 2 metre temperature can be found in 228004."
                    ),
                    "unit": {"symbol": "K"},
                    "observedProperty": {
                        "id": "2t",
                        "label": {"en": "2 metre temperature"},
                    },
                },
            },
        }

        assert encoder_obj.covjson == covjson

    def test_add_reference(self):
        encoder_obj = Eccovjson().encode("CoverageCollection", "wkt")
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
            "domainType": "MultiPoint",
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
        encoder = Eccovjson().encode("CoverageCollection", "wkt")
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
        coords["t"] = ["2017-01-01T00:00:00"]
        coords["composite"] = [[1, 20, 1], [2, 21, 3], [3, 17, 7]]
        value = {"2t": [111, 222, 333]}
        encoder.add_coverage(metadata, coords, value)
        # print(encoder.covjson)

        print(encoder.covjson)

    """

    def test_from_xarray(self):
        ds = xr.open_dataset("new_timeseries.nc")
        encoder = Eccovjson().encode("CoverageCollection", "PointSeries")
        encoder.from_xarray(ds)

        """
