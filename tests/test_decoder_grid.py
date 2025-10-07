# from earthkit import data

from covjsonkit.decoder import Grid


class TestDecoder:
    def setup_method(self, method):
        self.covjson = {
            "type": "CoverageCollection",
            "domainType": "Grid",
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
                            "t": {"values": [0, 1]},
                            "latitude": {"values": [0.035149384216, 0.105448152647, 0.175746921078]},
                            "longitude": {"values": [0.0, 0.070093457944, 0.140186915888]},
                            "levelist": {"values": [0]},
                        },
                    },
                    "ranges": {
                        "10v": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [2, 1, 3, 3],
                            "axisNames": ["t", "levelist", "latitude", "longitude"],
                            "values": [
                                6.583221435546875,
                                6.821502685546875,
                                7.079315185546875,
                                6.731658935546875,
                                6.913299560546875,
                                7.069549560546875,
                                6.897674560546875,
                                7.046112060546875,
                                7.153533935546875,
                                6.6175384521484375,
                                6.5999603271484375,
                                6.7191009521484375,
                                6.4573822021484375,
                                6.6487884521484375,
                                6.9749603271484375,
                                6.5316009521484375,
                                6.8128509521484375,
                                7.1409759521484375,
                            ],
                        },
                        "2t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [2, 1, 3, 3],
                            "axisNames": ["t", "levelist", "latitude", "longitude"],
                            "values": [
                                297.5169372558594,
                                297.6946716308594,
                                297.8177185058594,
                                297.5989685058594,
                                297.7767028808594,
                                297.8411560058594,
                                297.7395935058594,
                                297.9095153808594,
                                297.8763122558594,
                                297.45335388183594,
                                297.39671325683594,
                                297.48655700683594,
                                297.42210388183594,
                                297.43186950683594,
                                297.54124450683594,
                                297.46897888183594,
                                297.53147888183594,
                                297.64476013183594,
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
                            "t": {"values": [0, 1]},
                            "latitude": {"values": [0.035149384216, 0.105448152647, 0.175746921078]},
                            "longitude": {"values": [0.0, 0.070093457944, 0.140186915888]},
                            "levelist": {"values": [0]},
                        },
                    },
                    "ranges": {
                        "10v": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [2, 1, 3, 3],
                            "axisNames": ["t", "levelist", "latitude", "longitude"],
                            "values": [
                                6.583221435546875,
                                6.821502685546875,
                                7.079315185546875,
                                6.731658935546875,
                                6.913299560546875,
                                7.069549560546875,
                                6.897674560546875,
                                7.046112060546875,
                                7.153533935546875,
                                6.6175384521484375,
                                6.5999603271484375,
                                6.7191009521484375,
                                6.4573822021484375,
                                6.6487884521484375,
                                6.9749603271484375,
                                6.5316009521484375,
                                6.8128509521484375,
                                7.1409759521484375,
                            ],
                        },
                        "2t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [2, 1, 3, 3],
                            "axisNames": ["t", "levelist", "latitude", "longitude"],
                            "values": [
                                297.5169372558594,
                                297.6946716308594,
                                297.8177185058594,
                                297.5989685058594,
                                297.7767028808594,
                                297.8411560058594,
                                297.7395935058594,
                                297.9095153808594,
                                297.8763122558594,
                                297.45335388183594,
                                297.39671325683594,
                                297.48655700683594,
                                297.42210388183594,
                                297.43186950683594,
                                297.54124450683594,
                                297.46897888183594,
                                297.53147888183594,
                                297.64476013183594,
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
                "10v": {
                    "type": "Parameter",
                    "description": "10 metre V wind component",
                    "unit": {"symbol": "m s**-1"},
                    "observedProperty": {"id": "10v", "label": {"en": "10 metre V wind component"}},
                },
                "2t": {
                    "type": "Parameter",
                    "description": "Temperature",
                    "unit": {"symbol": "K"},
                    "observedProperty": {"id": "2t", "label": {"en": "2 metre temperature"}},
                },
            },
        }

    def test_grid_type(self):
        decoder = Grid.Grid(self.covjson)
        assert decoder.type == "CoverageCollection"

    def test_Grid_parameters(self):
        decoder = Grid.Grid(self.covjson)
        assert decoder.parameters == ["10v", "2t"]

    def test_grid_referencing(self):
        decoder = Grid.Grid(self.covjson)
        assert decoder.get_referencing() == ["x", "y", "z"]

    def test_grid_get_parameter_metadata(self):
        decoder = Grid.Grid(self.covjson)
        assert decoder.get_parameter_metadata("2t") == {
            "type": "Parameter",
            "description": "Temperature",
            "unit": {"symbol": "K"},
            "observedProperty": {"id": "2t", "label": {"en": "2 metre temperature"}},
        }
        assert decoder.get_parameter_metadata("10v") == {
            "type": "Parameter",
            "description": "10 metre V wind component",
            "unit": {"symbol": "m s**-1"},
            "observedProperty": {"id": "10v", "label": {"en": "10 metre V wind component"}},
        }

    def test_grid_mars_metadata(self):
        decoder = Grid.Grid(self.covjson)
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

    def test_grid_domains(self):
        decoder = Grid.Grid(self.covjson)
        domain1 = {
            "type": "Domain",
            "axes": {
                "t": {"values": [0, 1]},
                "latitude": {"values": [0.035149384216, 0.105448152647, 0.175746921078]},
                "longitude": {"values": [0.0, 0.070093457944, 0.140186915888]},
                "levelist": {"values": [0]},
            },
        }
        assert decoder.domains[0] == domain1
        domain2 = {
            "type": "Domain",
            "axes": {
                "t": {"values": [0, 1]},
                "latitude": {"values": [0.035149384216, 0.105448152647, 0.175746921078]},
                "longitude": {"values": [0.0, 0.070093457944, 0.140186915888]},
                "levelist": {"values": [0]},
            },
        }
        assert decoder.domains[1] == domain2

    def test_grid_ranges(self):
        decoder = Grid.Grid(self.covjson)
        range1 = {
            "10v": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2, 1, 3, 3],
                "axisNames": ["t", "levelist", "latitude", "longitude"],
                "values": [
                    6.583221435546875,
                    6.821502685546875,
                    7.079315185546875,
                    6.731658935546875,
                    6.913299560546875,
                    7.069549560546875,
                    6.897674560546875,
                    7.046112060546875,
                    7.153533935546875,
                    6.6175384521484375,
                    6.5999603271484375,
                    6.7191009521484375,
                    6.4573822021484375,
                    6.6487884521484375,
                    6.9749603271484375,
                    6.5316009521484375,
                    6.8128509521484375,
                    7.1409759521484375,
                ],
            },
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2, 1, 3, 3],
                "axisNames": ["t", "levelist", "latitude", "longitude"],
                "values": [
                    297.5169372558594,
                    297.6946716308594,
                    297.8177185058594,
                    297.5989685058594,
                    297.7767028808594,
                    297.8411560058594,
                    297.7395935058594,
                    297.9095153808594,
                    297.8763122558594,
                    297.45335388183594,
                    297.39671325683594,
                    297.48655700683594,
                    297.42210388183594,
                    297.43186950683594,
                    297.54124450683594,
                    297.46897888183594,
                    297.53147888183594,
                    297.64476013183594,
                ],
            },
        }
        assert decoder.ranges[0] == range1
        range2 = {
            "10v": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2, 1, 3, 3],
                "axisNames": ["t", "levelist", "latitude", "longitude"],
                "values": [
                    6.583221435546875,
                    6.821502685546875,
                    7.079315185546875,
                    6.731658935546875,
                    6.913299560546875,
                    7.069549560546875,
                    6.897674560546875,
                    7.046112060546875,
                    7.153533935546875,
                    6.6175384521484375,
                    6.5999603271484375,
                    6.7191009521484375,
                    6.4573822021484375,
                    6.6487884521484375,
                    6.9749603271484375,
                    6.5316009521484375,
                    6.8128509521484375,
                    7.1409759521484375,
                ],
            },
            "2t": {
                "type": "NdArray",
                "dataType": "float",
                "shape": [2, 1, 3, 3],
                "axisNames": ["t", "levelist", "latitude", "longitude"],
                "values": [
                    297.5169372558594,
                    297.6946716308594,
                    297.8177185058594,
                    297.5989685058594,
                    297.7767028808594,
                    297.8411560058594,
                    297.7395935058594,
                    297.9095153808594,
                    297.8763122558594,
                    297.45335388183594,
                    297.39671325683594,
                    297.48655700683594,
                    297.42210388183594,
                    297.43186950683594,
                    297.54124450683594,
                    297.46897888183594,
                    297.53147888183594,
                    297.64476013183594,
                ],
            },
        }
        assert decoder.ranges[1] == range2

    def test_grid_get_coordinates(self):
        decoder = Grid.Grid(self.covjson)
        coordinates = {
            "t": {"values": [0, 1]},
            "latitude": {"values": [0.035149384216, 0.105448152647, 0.175746921078]},
            "longitude": {"values": [0.0, 0.070093457944, 0.140186915888]},
            "levelist": {"values": [0]},
        }
        assert decoder.get_coordinates() == coordinates

    def test_grid_get_values(self):
        decoder = Grid.Grid(self.covjson)
        values = {
            "2t": [
                [
                    297.5169372558594,
                    297.6946716308594,
                    297.8177185058594,
                    297.5989685058594,
                    297.7767028808594,
                    297.8411560058594,
                    297.7395935058594,
                    297.9095153808594,
                    297.8763122558594,
                    297.45335388183594,
                    297.39671325683594,
                    297.48655700683594,
                    297.42210388183594,
                    297.43186950683594,
                    297.54124450683594,
                    297.46897888183594,
                    297.53147888183594,
                    297.64476013183594,
                ],
                [
                    297.5169372558594,
                    297.6946716308594,
                    297.8177185058594,
                    297.5989685058594,
                    297.7767028808594,
                    297.8411560058594,
                    297.7395935058594,
                    297.9095153808594,
                    297.8763122558594,
                    297.45335388183594,
                    297.39671325683594,
                    297.48655700683594,
                    297.42210388183594,
                    297.43186950683594,
                    297.54124450683594,
                    297.46897888183594,
                    297.53147888183594,
                    297.64476013183594,
                ],
            ],
            "10v": [
                [
                    6.583221435546875,
                    6.821502685546875,
                    7.079315185546875,
                    6.731658935546875,
                    6.913299560546875,
                    7.069549560546875,
                    6.897674560546875,
                    7.046112060546875,
                    7.153533935546875,
                    6.6175384521484375,
                    6.5999603271484375,
                    6.7191009521484375,
                    6.4573822021484375,
                    6.6487884521484375,
                    6.9749603271484375,
                    6.5316009521484375,
                    6.8128509521484375,
                    7.1409759521484375,
                ],
                [
                    6.583221435546875,
                    6.821502685546875,
                    7.079315185546875,
                    6.731658935546875,
                    6.913299560546875,
                    7.069549560546875,
                    6.897674560546875,
                    7.046112060546875,
                    7.153533935546875,
                    6.6175384521484375,
                    6.5999603271484375,
                    6.7191009521484375,
                    6.4573822021484375,
                    6.6487884521484375,
                    6.9749603271484375,
                    6.5316009521484375,
                    6.8128509521484375,
                    7.1409759521484375,
                ],
            ],
        }
        assert decoder.get_values() == values
