import pytest
import json

from covjson.decoder import decoder
from covjson.decoder import VerticalProfile
from covjson.decoder import TimeSeries


class TestDecoder:
    def setup_method(self, method):
        self.covjson = {
            "type": "CoverageCollection",
            "domainType": "VerticalProfile",
            "coverages": [
                {
                    "mars:metadata": {
                        "class": "ea",
                        "date": "2017-01-01 12:00:00",
                        "levtype": "pl",
                        "step": "0",
                        "stream": "enda",
                        "number": "0",
                    },
                    "domain": {
                        "type": "Domain",
                        "domainType": "VerticalProfile",
                        "axes": {
                            "x": {"values": ["0.0"]},
                            "y": {"values": ["0.0"]},
                            "z": {"values": ["500", "850"]},
                            "t": {"values": ["2017-01-01 12:00:00"]},
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "axisNames": ["z"],
                            "shape": [2],
                            "values": [57517.77734375, 14814.95703125],
                        },
                        "p": {
                            "type": "NdArray",
                            "dataType": "float",
                            "axisNames": ["z"],
                            "shape": [2],
                            "values": [16452.35546875, 44122.98046875],
                        },
                    },
                },
                {
                    "mars:metadata": {
                        "class": "ea",
                        "date": "2017-01-01 12:00:00",
                        "levtype": "pl",
                        "step": "0",
                        "stream": "enda",
                        "number": "1",
                    },
                    "domain": {
                        "type": "Domain",
                        "domainType": "VerticalProfile",
                        "axes": {
                            "x": {"values": ["0.0"]},
                            "y": {"values": ["0.0"]},
                            "z": {"values": ["500", "850"]},
                            "t": {"values": ["2017-01-01 12:00:00"]},
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "axisNames": ["z"],
                            "shape": [2],
                            "values": [57452.35546875, 14822.98046875],
                        },
                        "p": {
                            "type": "NdArray",
                            "dataType": "float",
                            "axisNames": ["z"],
                            "shape": [2],
                            "values": [56452.35546875, 14122.98046875],
                        },
                    },
                },
            ],
            "referencing": [
                {
                    "coordinates": ["x", "y"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                    },
                },
                {
                    "coordinates": ["z"],
                    "system": {
                        "type": "VerticalCRS",
                        "cs": {
                            "csAxes": [{"name": {"en": "level"}, "direction": "down"}]
                        },
                    },
                },
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

    def test_verticalprofile_type(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        assert decoder.type == "CoverageCollection"

    def test_verticalprofile_parameters(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        assert decoder.parameters == ["t", "p"]

    def test_verticalprofile_referencing(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        assert decoder.get_referencing() == ["x", "y", "z"]

    def test_verticalprofile_mars_metadata(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        metadata1 = {
            "class": "ea",
            "date": "2017-01-01 12:00:00",
            "levtype": "pl",
            "step": "0",
            "stream": "enda",
            "number": "0",
        }
        assert decoder.mars_metadata[0] == metadata1
        metadata2 = {
            "class": "ea",
            "date": "2017-01-01 12:00:00",
            "levtype": "pl",
            "step": "0",
            "stream": "enda",
            "number": "1",
        }
        assert decoder.mars_metadata[1] == metadata2

    def test_verticalprofile_domains(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        domain1 = {
            "type": "Domain",
            "domainType": "VerticalProfile",
            "axes": {
                "x": {"values": ["0.0"]},
                "y": {"values": ["0.0"]},
                "z": {"values": ["500", "850"]},
                "t": {"values": ["2017-01-01 12:00:00"]},
            },
        }
        assert decoder.domains[0] == domain1
        domain2 = {
            "type": "Domain",
            "domainType": "VerticalProfile",
            "axes": {
                "x": {"values": ["0.0"]},
                "y": {"values": ["0.0"]},
                "z": {"values": ["500", "850"]},
                "t": {"values": ["2017-01-01 12:00:00"]},
            },
        }
        assert decoder.domains[1] == domain2

    def test_verticalprofile_ranges(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        range1 = {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["z"],
                "shape": [2],
                "values": [57517.77734375, 14814.95703125],
            },
            "p": {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["z"],
                "shape": [2],
                "values": [16452.35546875, 44122.98046875],
            },
        }
        assert decoder.ranges[0] == range1
        range2 = {
            "t": {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["z"],
                "shape": [2],
                "values": [57452.35546875, 14822.98046875],
            },
            "p": {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["z"],
                "shape": [2],
                "values": [56452.35546875, 14122.98046875],
            },
        }
        assert decoder.ranges[1] == range2

    def test_verticalprofile_coordinates(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        coordinates = [
            ["0.0", "0.0", "500", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "500", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "850", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "850", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "500", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "500", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "850", "2017-01-01 12:00:00"],
            ["0.0", "0.0", "850", "2017-01-01 12:00:00"],
        ]
        assert decoder.get_coordinates() == coordinates

    def test_verticalprofile_values(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        values = {
            "t": [[57517.77734375, 14814.95703125], [57452.35546875, 14822.98046875]],
            "p": [[16452.35546875, 44122.98046875], [56452.35546875, 14122.98046875]],
        }
        assert decoder.get_values() == values