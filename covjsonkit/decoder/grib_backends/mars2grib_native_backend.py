"""GRIB encoding backend using pymars2grib with **native sub-area support**
via compound gridSpec + ``skipSection3``.

This is the "clean" mars2grib path recommended by the metkit developer and
documented in ``Mars2GribEncoding.md``.  Unlike the hybrid backend
(``mars2grib_backend.py``), it does NOT touch section 3 with eccodes:

    1. Build a compound gridSpec string:
           mars["grid"] = "{grid: O1280, area: [N, W, S, E]}"
    2. Encode with ``Mars2Grib(options={"skipSection3": True})``
    3. metkit forwards the string to eccodes' ``gridSpec`` accessor
       (``GridSpec::pack_string``), eckit-geo builds the reduced-gg
       sub-area natively, section 3 is written by eccodes with the
       correct corners, ``numberOfDataPoints``, and full ``pl``.

Runtime requirement:
    ``ECCODES_ECKIT_GEO=2`` must be exported (or ``=1``).  Without it,
    ``GridSpec::pack_string`` returns ``GRIB_NOT_IMPLEMENTED`` — see
    ``eccodes/geo/eckit.h`` for the ``EckitGeoLevel`` enum.  Our
    ``mars2grib-bundle/env.sh`` sets this correctly.

Ensemble note:
    Same as the hybrid backend — mars2grib has no default deduction for
    ``numberOfForecastsInEnsemble``, so we inject it into ``misc`` when
    the stream/type is ensemble.
"""

import logging
import os

from .base import GribBackend
from .mars2grib_backend import _ENSEMBLE_STREAMS, _ENSEMBLE_TYPES, _coerce_mars_types

logger = logging.getLogger(__name__)


def _build_compound_grid_spec(misc: dict, values_len: int) -> str:
    """Build the compound gridSpec string that eckit-geo can decode.

    Format::

        {grid: <name>, area: [N, W, S, E]}

    for reduced-gg grids, or::

        {grid: [dx, dy], area: [N, W, S, E]}

    for regular_ll.  The area is taken from ``misc["area"]`` if present,
    otherwise from the coverage bounds computed by
    ``BoundingBox._build_misc_dict``.
    """
    grid_type = misc.get("gridType", "reduced_gg")

    if grid_type == "reduced_gg":
        n = misc.get("N", 1280)
        grid_str = f"O{int(n)}"
    elif grid_type == "regular_ll":
        dx = misc.get("iDirectionIncrementInDegrees", 0.1)
        dy = misc.get("jDirectionIncrementInDegrees", 0.1)
        grid_str = f"[{dx}, {dy}]"
    else:
        n = misc.get("N", 1280)
        grid_str = f"O{int(n)}"
        logger.warning("Unknown gridType '%s', defaulting to O%d", grid_type, n)

    area = misc.get("area")
    if area and len(area) == 4:
        return f"{{grid: {grid_str}, area: [{area[0]}, {area[1]}, {area[2]}, {area[3]}]}}"

    # No area — full-globe (uncommon for our use case)
    logger.warning("No area in misc dict; encoding as full globe (values_len=%d)", values_len)
    return f"{{grid: {grid_str}}}"


def _build_mars_dict(mars: dict, misc: dict) -> dict:
    """Build the mars dict for the compound-gridSpec path.

    Same as the hybrid backend, but ``grid`` is a compound spec, and we
    also strip ``number`` for non-ensemble types (polytope-mars sets
    ``number: 0`` on oper coverages, which otherwise routes ``fc`` into
    ensemble encoding and aborts).
    """
    result = _coerce_mars_types(mars)

    # Strip spurious `number: 0` on oper coverages
    stream = str(result.get("stream", "")).lower()
    typ = str(result.get("type", "")).lower()
    is_ensemble = stream in _ENSEMBLE_STREAMS or typ in _ENSEMBLE_TYPES
    if not is_ensemble and "number" in result:
        result.pop("number")

    result.setdefault("origin", "ecmf")
    result.setdefault("packing", "ccsds")

    # Compound gridSpec — verbatim to eccodes' gridSpec accessor
    result["grid"] = _build_compound_grid_spec(misc, values_len=0)
    return result


def _build_misc_dict(mars: dict, misc: dict, values_len: int) -> dict:
    """Build the misc dict — mars2grib only cares about a few keys here.

    Geometry keys (gridType, Nj, pl, area…) are ignored by mars2grib —
    they go into the compound gridSpec string on ``mars["grid"]`` instead.
    """
    result = {}

    # Ensemble deduction
    stream = str(mars.get("stream", "")).lower()
    typ = str(mars.get("type", "")).lower()
    if stream in _ENSEMBLE_STREAMS or typ in _ENSEMBLE_TYPES:
        result["numberOfForecastsInEnsemble"] = int(misc.get("numberOfForecastsInEnsemble", 51))

    # Packing precision
    if "bitsPerValue" in misc:
        result["bitsPerValue"] = int(misc["bitsPerValue"])

    return result


class Mars2GribNativeBackend(GribBackend):
    """Encode GRIB2 messages natively via ``Mars2Grib(skipSection3=True)``.

    Uses compound gridSpec on ``mars["grid"]`` so eckit-geo builds the
    sub-area natively and eccodes writes section 3 correctly.  No manual
    section-3 overwrite required.

    Trade-off vs the hybrid backend:
      * ``+`` No hand-rolled ``Nj``/``pl`` computation — eckit-geo owns it.
      * ``+`` Corner lat/lons match MARS's exact snap-out convention.
      * ``-`` Requires ``ECCODES_ECKIT_GEO`` env var; without it, aborts.
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

        # Runtime env-var check — GridSpec::pack_string is gated on this.
        if os.environ.get("ECCODES_ECKIT_GEO", "0") == "0":
            logger.warning(
                "ECCODES_ECKIT_GEO is not set — mars2grib_native will abort "
                "when encoding.  Run `source mars2grib-bundle/env.sh` first."
            )

        self._encoder = Mars2Grib(options={"skipSection3": True})

    def encode_message(self, values: list, mars: dict, misc: dict) -> bytes:
        """Encode a single GRIB2 message.

        Args:
            values: Field data values (sub-area, N→S/W→E ordered).
            mars: MARS keys.
            misc: Grid geometry (used to build the compound gridSpec).
        """
        typed_mars = _build_mars_dict(mars, misc)
        typed_misc = _build_misc_dict(mars, misc, values_len=len(values))
        return self._encoder.encode([float(v) for v in values], typed_mars, typed_misc)
