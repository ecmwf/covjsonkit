"""GRIB encoding backend using the eccodes Python API.

This is the fallback backend when pymars2grib is not available.
It builds a GRIB2 message from scratch using eccodes sample-based
creation, setting keys derived from MARS metadata and grid geometry.
"""

import logging

import eccodes

from .base import GribBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MARS levtype → eccodes typeOfLevel mapping
# ---------------------------------------------------------------------------
_LEVTYPE_MAP = {
    "sfc": "surface",
    "pl": "isobaricInhPa",
    "ml": "hybrid",
    "pv": "potentialVorticity",
    "pt": "theta",
    "dp": "depthBelowSea",
}

# ---------------------------------------------------------------------------
# MARS stream → GRIB2 code-table values
# ---------------------------------------------------------------------------
_STREAM_MAP = {
    "oper": "oper",
    "enfo": "enfo",
    "efov": "efov",
    "scda": "scda",
    "scwv": "scwv",
    "wave": "wave",
    "waef": "waef",
    "moda": "moda",
}

# ---------------------------------------------------------------------------
# MARS type → GRIB2 code-table values
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "an": "an",
    "fc": "fc",
    "pf": "pf",
    "cf": "cf",
    "em": "em",
    "es": "es",
}


class EccodesBackend(GribBackend):
    """Encode GRIB2 messages using the eccodes Python API."""

    def __init__(self):
        # Verify eccodes is functional
        try:
            eccodes.codes_get_api_version()
        except Exception as exc:
            raise ImportError(f"eccodes is installed but not functional: {exc}") from exc

    def encode_message(self, values: list, mars: dict, misc: dict) -> bytes:
        """Encode a single GRIB2 message.

        Args:
            values: Field data values.
            mars: MARS keys (class, stream, type, date, time, step, param,
                  levtype, levelist, number, …).
            misc: Grid geometry and encoding options (gridType, N, area, …).

        Returns:
            Encoded GRIB2 message as bytes.
        """
        sample_id = eccodes.codes_grib_new_from_samples("GRIB2")
        try:
            # Enable the ECMWF local definition section so that MARS
            # namespace keys (marsClass, marsStream, marsType, …) are
            # available for reading and writing.
            eccodes.codes_set(sample_id, "centre", "ecmf")
            eccodes.codes_set_long(sample_id, "setLocalDefinition", 1)

            self._set_identification(sample_id, mars)
            self._set_temporal(sample_id, mars)
            self._set_product(sample_id, mars)
            self._set_grid(sample_id, mars, misc, len(values))
            self._set_ensemble(sample_id, mars)
            self._set_values(sample_id, values)

            return eccodes.codes_get_message(sample_id)
        finally:
            eccodes.codes_release(sample_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_identification(gid: int, mars: dict) -> None:
        """Set GRIB identification keys from MARS dict."""
        if "class" in mars:
            eccodes.codes_set(gid, "marsClass", mars["class"])
        if "stream" in mars:
            stream = _STREAM_MAP.get(mars["stream"], mars["stream"])
            eccodes.codes_set(gid, "marsStream", stream)
        if "type" in mars:
            mars_type = _TYPE_MAP.get(mars["type"], mars["type"])
            eccodes.codes_set(gid, "marsType", mars_type)
        if "expver" in mars:
            eccodes.codes_set(gid, "experimentVersionNumber", mars["expver"])

    @staticmethod
    def _set_temporal(gid: int, mars: dict) -> None:
        """Set date, time, and step keys."""
        if "date" in mars:
            date_str = str(mars["date"]).replace("-", "")
            eccodes.codes_set_long(gid, "dataDate", int(date_str))
        if "time" in mars:
            time_str = str(mars["time"]).replace(":", "")
            # Pad to 4 digits (e.g. "0" → "0000", "12" → "1200")
            time_str = time_str.ljust(4, "0")
            eccodes.codes_set_long(gid, "dataTime", int(time_str))
        if "step" in mars:
            eccodes.codes_set(gid, "stepRange", str(mars["step"]))

    @staticmethod
    def _set_product(gid: int, mars: dict) -> None:
        """Set parameter and level keys.

        Important: ``typeOfLevel`` and ``level`` must be set BEFORE
        ``paramId`` because eccodes re-resolves paramId when the type
        of fixed surface changes (e.g. 10v ↔ v depending on surface
        vs. upper-air).
        """
        levtype = mars.get("levtype", "sfc")
        type_of_level = _LEVTYPE_MAP.get(levtype, levtype)
        eccodes.codes_set(gid, "typeOfLevel", type_of_level)

        if "levelist" in mars:
            eccodes.codes_set_long(gid, "level", int(mars["levelist"]))
        elif levtype == "sfc":
            eccodes.codes_set_long(gid, "level", 0)

        if "param" in mars:
            param = mars["param"]
            try:
                eccodes.codes_set_long(gid, "paramId", int(param))
            except (ValueError, TypeError):
                # If param is a shortname string, set it that way
                eccodes.codes_set(gid, "shortName", str(param))

    @staticmethod
    def _set_ensemble(gid: int, mars: dict) -> None:
        """Set ensemble-specific keys if present.

        ``productDefinitionTemplateNumber`` must be set to 1 first so
        that the ``perturbationNumber`` key becomes available in GRIB2.
        """
        if "number" in mars:
            number = int(mars["number"])
            if number > 0:
                # Switch to ensemble product definition template FIRST
                eccodes.codes_set_long(gid, "productDefinitionTemplateNumber", 1)
                eccodes.codes_set_long(gid, "perturbationNumber", number)

    def _set_grid(self, gid: int, mars: dict, misc: dict, num_values: int) -> None:
        """Set grid definition from misc dict.

        Supports reduced_gg (reduced Gaussian) and regular_ll (regular lat/lon).
        Falls back to a simple points-based approach if grid type is unknown.
        """
        grid_type = misc.get("gridType", "reduced_gg")

        if grid_type == "reduced_gg":
            self._set_reduced_gaussian_grid(gid, misc, num_values)
        elif grid_type == "regular_ll":
            self._set_regular_ll_grid(gid, misc, num_values)
        else:
            # Fallback: set as unstructured grid with explicit coordinates
            logger.warning("Unknown gridType '%s', falling back to unstructured grid", grid_type)
            self._set_unstructured_grid(gid, misc, num_values)

    @staticmethod
    def _set_reduced_gaussian_grid(gid: int, misc: dict, num_values: int) -> None:
        """Configure a reduced Gaussian grid (sub-area)."""
        eccodes.codes_set(gid, "gridType", "reduced_gg")

        n = misc.get("N", 1280)
        eccodes.codes_set_long(gid, "N", n)

        # Area bounds (N/W/S/E) in degrees
        if "area" in misc:
            area = misc["area"]
            eccodes.codes_set_double(gid, "latitudeOfFirstGridPointInDegrees", float(area[0]))
            eccodes.codes_set_double(gid, "longitudeOfFirstGridPointInDegrees", float(area[1]))
            eccodes.codes_set_double(gid, "latitudeOfLastGridPointInDegrees", float(area[2]))
            eccodes.codes_set_double(gid, "longitudeOfLastGridPointInDegrees", float(area[3]))

        # Nj (number of latitude rows in the sub-area)
        if "Nj" in misc:
            eccodes.codes_set_long(gid, "Nj", int(misc["Nj"]))

        # pl array (number of points per latitude row) — required for sub-area
        if "pl" in misc:
            eccodes.codes_set_array(gid, "pl", misc["pl"])

        eccodes.codes_set_long(gid, "numberOfDataPoints", num_values)

    @staticmethod
    def _set_regular_ll_grid(gid: int, misc: dict, num_values: int) -> None:
        """Configure a regular lat/lon grid (sub-area)."""
        eccodes.codes_set(gid, "gridType", "regular_ll")

        if "area" in misc:
            area = misc["area"]
            eccodes.codes_set_double(gid, "latitudeOfFirstGridPointInDegrees", float(area[0]))
            eccodes.codes_set_double(gid, "longitudeOfFirstGridPointInDegrees", float(area[1]))
            eccodes.codes_set_double(gid, "latitudeOfLastGridPointInDegrees", float(area[2]))
            eccodes.codes_set_double(gid, "longitudeOfLastGridPointInDegrees", float(area[3]))

        if "Dx" in misc:
            eccodes.codes_set_double(gid, "iDirectionIncrementInDegrees", float(misc["Dx"]))
        if "Dy" in misc:
            eccodes.codes_set_double(gid, "jDirectionIncrementInDegrees", float(misc["Dy"]))
        if "Ni" in misc:
            eccodes.codes_set_long(gid, "Ni", int(misc["Ni"]))
        if "Nj" in misc:
            eccodes.codes_set_long(gid, "Nj", int(misc["Nj"]))

        eccodes.codes_set_long(gid, "numberOfDataPoints", num_values)

    @staticmethod
    def _set_unstructured_grid(gid: int, misc: dict, num_values: int) -> None:
        """Fallback: configure as an unstructured grid."""
        eccodes.codes_set(gid, "gridType", "unstructured_grid")
        eccodes.codes_set_long(gid, "numberOfDataPoints", num_values)

    @staticmethod
    def _set_values(gid: int, values: list) -> None:
        """Pack field values into the GRIB data section."""
        # Use 16 bits per value (MARS default for oper data)
        eccodes.codes_set_long(gid, "bitsPerValue", 16)
        eccodes.codes_set_values(gid, values)
