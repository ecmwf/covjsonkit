"""Swappable GRIB encoding backends.

Use :func:`get_backend` to obtain the best available backend at runtime.
The factory tries ``pymars2grib`` first (preferred) and falls back to
``eccodes`` when it is not installed.
"""

from .base import GribBackend  # noqa: F401


def get_backend(preferred: str = "auto") -> GribBackend:
    """Return a ready-to-use GRIB encoding backend.

    Args:
        preferred: One of ``"auto"``, ``"mars2grib"``, or ``"eccodes"``.
            * ``"auto"`` – try mars2grib first, fall back to eccodes.
            * ``"mars2grib"`` – require mars2grib (raises ImportError if
              unavailable).
            * ``"eccodes"`` – use eccodes directly.

    Returns:
        A :class:`GribBackend` instance.

    Raises:
        ImportError: If the explicitly requested backend is not installed.
    """
    if preferred in ("mars2grib", "auto"):
        try:
            from .mars2grib_backend import Mars2GribBackend

            return Mars2GribBackend()
        except ImportError:
            if preferred == "mars2grib":
                raise ImportError(
                    "pymars2grib is not installed. "
                    "Build metkit from source with pybind11 support, or use backend='eccodes'."
                )

    try:
        from .eccodes_backend import EccodesBackend

        return EccodesBackend()
    except ImportError:
        raise ImportError(
            "No GRIB backend available. Install eccodes: pip install eccodes, "
            "or build pymars2grib from metkit source."
        )
