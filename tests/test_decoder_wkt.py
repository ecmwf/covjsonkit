# from earthkit import data

from covjsonkit.decoder import Wkt


class TestDecoder:
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

    def test_bounding_box_type(self):
        decoder = Wkt.Wkt(self.covjson)
        assert decoder.type == "CoverageCollection"

    def test_bounding_box_parameters(self):
        decoder = Wkt.Wkt(self.covjson)
        assert decoder.parameters == ["t", "p"]

    def test_bounding_box_referencing(self):
        decoder = Wkt.Wkt(self.covjson)
        assert decoder.get_referencing() == ["x", "y", "z"]

    def test_bounding_box_get_parameter_metadata(self):
        decoder = Wkt.Wkt(self.covjson)
        assert decoder.get_parameter_metadata("t") == {
            "type": "Parameter",
            "description": "Temperature",
            "unit": {"symbol": "K"},
            "observedProperty": {"id": "t", "label": {"en": "Temperature"}},
        }
        assert decoder.get_parameter_metadata("p") == {
            "type": "Parameter",
            "description": "Pressure",
            "unit": {"symbol": "pa"},
            "observedProperty": {"id": "p", "label": {"en": "Pressure"}},
        }

    def test_bounding_box_mars_metadata(self):
        decoder = Wkt.Wkt(self.covjson)
        assert decoder.mars_metadata[0] == {
            "class": "od",
            "stream": "oper",
            "levtype": "pl",
            "date": "20170101",
            "step": "0",
            "number": "0",
        }
        assert decoder.mars_metadata[1] == {
            "class": "od",
            "stream": "oper",
            "levtype": "pl",
            "date": "20170101",
            "step": "1",
            "number": "0",
        }

    def test_bounding_box_domains(self):
        decoder = Wkt.Wkt(self.covjson)
        domain1 = {
            "type": "Domain",
            "axes": {
                "t": {"values": ["2017-01-01T00:00:00"]},
                "composite": {
                    "dataType": "tuple",
                    "coordinates": ["x", "y", "z"],
                    "values": [[1, 20, 1], [2, 21, 3], [3, 17, 7]],
                },
            },
        }
        assert decoder.domains[0] == domain1
        domain2 = {
            "type": "Domain",
            "axes": {
                "t": {"values": ["2017-01-01T01:00:00"]},
                "composite": {
                    "dataType": "tuple",
                    "coordinates": ["x", "y", "z"],
                    "values": [[1, 20, 1], [2, 21, 3], [3, 17, 7]],
                },
            },
        }
        assert decoder.domains[1] == domain2

    def test_bounding_box_ranges(self):
        decoder = Wkt.Wkt(self.covjson)
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
                "values": [266.93115234375, 293.83115234375, 165.12313132266],
            },
            "p": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [3],
                "axisNames": ["t"],
                "values": [1.93115234375, 22.83115234375, 12.12313132266],
            },
        }
        assert decoder.ranges[1] == range2

    def test_bounding_box_get_coordinates(self):
        decoder = Wkt.Wkt(self.covjson)
        coordinates = {
            "t": {"values": ["2017-01-01T00:00:00"]},
            "composite": {
                "dataType": "tuple",
                "coordinates": ["x", "y", "z"],
                "values": [[1, 20, 1], [2, 21, 3], [3, 17, 7]],
            },
        }
        assert decoder.get_coordinates() == coordinates

    def test_bounding_box_get_values(self):
        decoder = Wkt.Wkt(self.covjson)
        values = {
            "t": [
                [264.93115234375, 263.83115234375, 265.12313132266],
                [266.93115234375, 293.83115234375, 165.12313132266],
            ],
            "p": [
                [9.93115234375, 7.83115234375, 14.12313132266],
                [1.93115234375, 22.83115234375, 12.12313132266],
            ],
        }
        assert decoder.get_values() == values

    # def test_bounding_box_to_xarray(self):
    #    decoder = BoundingBox.BoundingBox(self.covjson)
    #    dataset = decoder.to_xarray()
    #    print(dataset)
