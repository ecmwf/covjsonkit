"""Tests for Circle decoder's to_xarray method."""

from covjsonkit.decoder import Circle


class TestCircleDecoder:
    def setup_method(self):
        self.covjson = {
            "type": "CoverageCollection",
            "domainType": "circle",
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
                                "values": [[1, 20, 1], [2, 21, 3]],
                            },
                        },
                    },
                    "ranges": {
                        "t": {
                            "type": "NdArray",
                            "dataType": "float",
                            "shape": [2],
                            "axisNames": ["t"],
                            "values": [264.9, 263.8],
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
            },
        }

    def test_circle_to_xarray_param_t_no_collision(self):
        """to_xarray works with param 't' - no collision since dims use 'datetimes'."""
        decoder = Circle.Circle(self.covjson)
        ds = decoder.to_xarray()

        # Param 't' should be in data_vars (no collision with 'datetimes' dim)
        assert "t" in ds.data_vars, f"Expected 't' in data_vars, got {list(ds.data_vars)}"
        # Time dimension is 'datetimes', not 't'
        assert "datetimes" in ds.dims
