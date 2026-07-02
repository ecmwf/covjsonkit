"""GRIB encoding backend using pymars2grib (metkit's Python binding).

This is the preferred backend when available.  pymars2grib handles all
MARS-to-GRIB key resolution internally, so this wrapper is intentionally
thin.

pymars2grib is not yet available on PyPI — it must be built from metkit
source with pybind11 support.  When it is not installed the factory in
``__init__.py`` falls back to the eccodes backend.
"""

from .base import GribBackend


class Mars2GribBackend(GribBackend):
    """Encode GRIB2 messages via pymars2grib."""

    def __init__(self):
        try:
            from pymars2grib import Mars2Grib
        except ImportError:
            raise ImportError(
                "pymars2grib is not installed. "
                "Build metkit from source with pybind11 support to use this backend."
            )
        self._encoder = Mars2Grib()

    def encode_message(self, values: list, mars: dict, misc: dict) -> bytes:
        """Encode a single GRIB2 message.

        Delegates entirely to pymars2grib which resolves GRIB header
        layout from the MARS dictionary and injects the values.

        Args:
            values: Field data values.
            mars: MARS keys describing the field.
            misc: Auxiliary metadata (grid geometry, packing hints, …).

        Returns:
            Encoded GRIB2 message as bytes.
        """
        return self._encoder.encode(values, mars, misc)
