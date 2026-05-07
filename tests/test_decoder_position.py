"""Tests for Position decoder's to_xarray method."""

import json
from pathlib import Path

from covjsonkit.api import Covjsonkit


class TestPositionDecoder:
    """Tests for Position decoder to_xarray with param 't' collision."""

    def test_position_to_xarray_param_t(self):
        """to_xarray must not fail when parameter is named 't' (collides with time dim)."""
        path = Path(__file__).parent / "data/test_timeseries_param_t.json"
        with open(path, "r") as f:
            covjson = json.load(f)

        # Change domainType to position to use Position decoder
        covjson["domainType"] = "position"

        ds = Covjsonkit().decode(covjson).to_xarray()

        # Parameter 't' should be renamed to 'T' to avoid collision with time dimension
        assert "T" in ds.data_vars, f"Expected 'T' in data_vars, got {list(ds.data_vars)}"
        assert "t" in ds.dims or "t" in ds.coords, "Time dimension/coord 't' should still exist"
