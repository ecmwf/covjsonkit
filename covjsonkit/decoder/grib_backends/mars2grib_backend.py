"""GRIB encoding backend using pymars2grib (metkit's Python binding).

Hybrid approach: pymars2grib generates a full GRIB message with correct
header layout (sections 1, 4, 5) from MARS keys.  Then eccodes overwrites
section 3 (grid geometry) with sub-area information and updates the data
values.

NOTE: We do NOT use ``skipSection3``.  In the current metkit+eccodes
build, ``skipSection3=True`` triggers the ``gridSpec`` string setter path,
and eccodes reports ``GridSpec::pack_string not available``.  Instead we
let mars2grib emit a normal template and overwrite section 3 via eccodes
afterwards.

pymars2grib is not yet available on PyPI — it must be built from metkit
source with pybind11 support.  When not installed, the factory in
``__init__.py`` falls back to the eccodes-only backend.

Build instructions: see ``mars2grib-bundle/build.sh``.
"""

import logging

from .base import GribBackend

logger = logging.getLogger(__name__)

# MARS keys that pymars2grib expects as integers (long).
# The C++ layer uses strict type checking via eckit::LocalConfiguration.
_INT_MARS_KEYS = frozenset({"param", "step", "number", "levelist", "date", "time"})

# Ensemble stream/type combinations that require numberOfForecastsInEnsemble
# in the misc dict — mars2grib has no default deduction for this key.
_ENSEMBLE_TYPES = frozenset({"pf", "cf", "em", "es", "fcmean", "fcstdev"})
_ENSEMBLE_STREAMS = frozenset({"enfo", "elda", "waef", "eefo", "efov", "efho"})


def _coerce_mars_types(mars: dict) -> dict:
    """Coerce MARS dict values to the types pymars2grib expects."""
    result = {}
    for key, value in mars.items():
        if key in _INT_MARS_KEYS:
            try:
                result[key] = int(value)
            except (ValueError, TypeError):
                result[key] = value
        else:
            result[key] = value
    return result


def _build_mars2grib_mars_dict(mars: dict, misc: dict) -> dict:
    """Build the mars dict that pymars2grib expects.

    Adds required keys (origin, grid, packing) that mars2grib needs for
    header layout resolution but our _build_mars_dict doesn't provide.
    Also strips ``number`` for non-ensemble types — polytope-mars sets
    ``number: 0`` on oper coverages, which incorrectly routes ``fc`` into
    mars2grib's ensemble path and aborts on ``typeOfEnsembleForecast``.
    """
    result = _coerce_mars_types(mars)

    # Drop `number` for non-ensemble types.  polytope-mars includes
    # `number: 0` on oper coverages; mars2grib treats its presence as a
    # signal to encode as an ensemble member.
    stream = str(result.get("stream", "")).lower()
    typ = str(result.get("type", "")).lower()
    is_ensemble = stream in _ENSEMBLE_STREAMS or typ in _ENSEMBLE_TYPES
    if not is_ensemble and "number" in result:
        result.pop("number")

    # origin — required by mars2grib
    result.setdefault("origin", "ecmf")

    # packing — CCSDS matches operational MARS data
    result.setdefault("packing", "ccsds")

    # grid — required by mars2grib; derive from misc gridType/N.
    # This drives the template's section 3, which we overwrite anyway,
    # but mars2grib needs *some* valid grid to route the encoder.
    if "grid" not in result:
        grid_type = misc.get("gridType", "reduced_gg")
        n = misc.get("N", 1280)
        if grid_type == "reduced_gg":
            result["grid"] = f"O{n}"
        elif grid_type == "regular_ll":
            dx = misc.get("iDirectionIncrementInDegrees", 0.1)
            dy = misc.get("jDirectionIncrementInDegrees", 0.1)
            result["grid"] = f"{dy}/{dx}"
        else:
            result["grid"] = f"O{n}"
            logger.warning("Unknown gridType '%s', defaulting to O%d", grid_type, n)

    return result


def _build_mars2grib_misc_dict(mars: dict, misc: dict) -> dict:
    """Build the misc dict that pymars2grib expects.

    mars2grib silently ignores geometry keys (gridType, Nj, pl, area, …)
    — those are only used by our eccodes overwrite step.  But some
    concepts (notably ensembles) need explicit deduction hints here.
    """
    result = {}

    # Ensemble deduction: mars2grib has no default for numberOfForecastsInEnsemble
    stream = str(mars.get("stream", "")).lower()
    typ = str(mars.get("type", "")).lower()
    if stream in _ENSEMBLE_STREAMS or typ in _ENSEMBLE_TYPES:
        # Prefer explicit misc override; otherwise fall back to ECMWF ENS size
        result["numberOfForecastsInEnsemble"] = int(misc.get("numberOfForecastsInEnsemble", 51))

    return result


def _apply_section3_with_eccodes(template_bytes: bytes, misc: dict, values: list) -> bytes:
    """Overwrite GRIB section 3 (grid definition) and data values via eccodes."""
    import eccodes

    gid = eccodes.codes_new_from_message(template_bytes)

    try:
        grid_type = misc.get("gridType", "reduced_gg")
        eccodes.codes_set(gid, "gridType", grid_type)

        if grid_type == "reduced_gg":
            n = misc.get("N", 1280)
            nj = misc.get("Nj", 1)
            pl = misc.get("pl", [len(values)])

            eccodes.codes_set_long(gid, "numberOfParallelsBetweenAPoleAndTheEquator", int(n))
            eccodes.codes_set_long(gid, "Nj", int(nj))
            eccodes.codes_set_long_array(gid, "pl", [int(p) for p in pl])

        elif grid_type == "regular_ll":
            for key in ("Ni", "Nj"):
                if key in misc:
                    eccodes.codes_set_long(gid, key, int(misc[key]))
            for key in ("iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees"):
                if key in misc:
                    eccodes.codes_set_double(gid, key, float(misc[key]))

        # Area bounds
        area = misc.get("area")
        if area and len(area) == 4:
            eccodes.codes_set_double(gid, "latitudeOfFirstGridPointInDegrees", float(area[0]))
            eccodes.codes_set_double(gid, "longitudeOfFirstGridPointInDegrees", float(area[1]))
            eccodes.codes_set_double(gid, "latitudeOfLastGridPointInDegrees", float(area[2]))
            eccodes.codes_set_double(gid, "longitudeOfLastGridPointInDegrees", float(area[3]))

        # Packing precision
        bits_per_value = misc.get("bitsPerValue", 16)
        eccodes.codes_set_long(gid, "bitsPerValue", int(bits_per_value))

        # Set the actual data values
        eccodes.codes_set_values(gid, [float(v) for v in values])

        return eccodes.codes_get_message(gid)

    finally:
        eccodes.codes_release(gid)


class Mars2GribBackend(GribBackend):
    """Encode GRIB2 messages via pymars2grib + eccodes hybrid.

    mars2grib resolves the GRIB header layout (sections 1, 4, 5) from
    MARS keys.  eccodes then overwrites section 3 (grid geometry) with
    sub-area info and updates the data values.
    """

    def __init__(self):
        try:
            from pymars2grib import Mars2Grib
        except ImportError:
            raise ImportError(
                "pymars2grib is not installed. "
                "Build metkit from source with pybind11 support to use this backend. "
                "See mars2grib-bundle/build.sh."
            )
        # No skipSection3 — that path is broken in current builds.
        # We let mars2grib produce a normal template and overwrite section 3.
        self._encoder = Mars2Grib()

    def encode_message(self, values: list, mars: dict, misc: dict) -> bytes:
        """Encode a single GRIB2 message using the hybrid approach.

        1. mars2grib generates a full GRIB message from MARS keys
        2. eccodes overwrites section 3 (grid geometry) with sub-area info
        3. eccodes updates the data values

        Args:
            values: Field data values (sub-area, N→S/W→E ordered).
            mars: MARS keys describing the field (class, stream, type, param, …).
            misc: Grid geometry, encoding hints, ensemble deductions, …
        """
        typed_mars = _build_mars2grib_mars_dict(mars, misc)
        typed_misc = _build_mars2grib_misc_dict(mars, misc)

        # mars2grib requires values matching its template's grid size, but
        # since we overwrite section 3 the template's grid size is thrown
        # away.  Just pass the real values — eccodes will re-set them.
        template_bytes = self._encoder.encode(values, typed_mars, typed_misc)

        return _apply_section3_with_eccodes(template_bytes, misc, values)
