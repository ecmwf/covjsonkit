import pytest
import json

from covjson.decoder import Decoder
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
                    },
                    "domain": {
                        "type": "Domain",
                        "domainType": "VerticalProfile",
                        "axes": {
                            "x": {"values": ["0.0"]},
                            "y": {"values": ["0.0"]},
                            "z": {"values": ["500", "850"]},
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "axisNames": ["z"],
                            "shape": [2],
                            "values": [57517.77734375, 14814.95703125],
                        }
                    },
                },
                {
                    "mars:metadata": {
                        "class": "ea",
                        "date": "2017-01-02 12:00:00",
                        "levtype": "pl",
                        "step": "0",
                        "stream": "enda",
                    },
                    "domain": {
                        "type": "Domain",
                        "domainType": "VerticalProfile",
                        "axes": {
                            "x": {"values": ["0.0"]},
                            "y": {"values": ["0.0"]},
                            "z": {"values": ["500", "850"]},
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "axisNames": ["z"],
                            "shape": [2],
                            "values": [57452.35546875, 14822.98046875],
                        }
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

    def test_coveragecollection_type(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        assert decoder.type == "CoverageCollection"

    def test_coveragecollection_parameters(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        assert decoder.parameters == ["t", "p"]

    def test_coveragecollection_referencing(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        assert decoder.get_referencing() == ["x", "y", "z"]

    def test_coveragecollection_mars_metadata(self):
        decoder = VerticalProfile.VerticalProfile(self.covjson)
        metadata1 = {
            "class": "ea",
            "date": "2017-01-01 12:00:00",
            "levtype": "pl",
            "step": "0",
            "stream": "enda",
        }
        assert decoder.mars_metadata[0] == metadata1
        metadata2 = {
            "class": "ea",
            "date": "2017-01-02 12:00:00",
            "levtype": "pl",
            "step": "0",
            "stream": "enda",
        }
        assert decoder.mars_metadata[1] == metadata2
