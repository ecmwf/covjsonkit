from abc import ABC, abstractmethod


class GribBackend(ABC):
    """Abstract backend interface for encoding GRIB messages from MARS metadata + values.

    Implementations translate (values, mars_dict, misc_dict) into a single
    encoded GRIB message returned as ``bytes``.  Two concrete backends are
    provided:

    * :class:`~.eccodes_backend.EccodesBackend` – uses the eccodes Python API
      (available via ``pip install eccodes``).
    * :class:`~.mars2grib_backend.Mars2GribBackend` – wraps ECMWF's
      ``pymars2grib`` pybind11 module (requires metkit built from source).
    """

    @abstractmethod
    def encode_message(self, values: list, mars: dict, misc: dict) -> bytes:
        """Encode a single GRIB message.

        Args:
            values: Field data values (one per grid point).
            mars: MARS keys describing the field (class, stream, type, date,
                  time, step, param, levtype, levelist, …).
            misc: Non-MARS metadata such as grid geometry (gridType, N, area,
                  …) and packing options.

        Returns:
            The fully encoded GRIB message as raw bytes.
        """
        pass
