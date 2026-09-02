"""Swappable GRIB encoding backends.

Use :func:`get_backend` to obtain the best available backend at runtime.
Available backends (in preference order for ``"auto"``):

  * ``mars2grib``        – hybrid: mars2grib for header, eccodes for section 3
  * ``mars2grib_native`` – pure mars2grib via compound gridSpec + skipSection3
  * ``eccodes``          – hand-rolled with eccodes only
"""

from .base import GribBackend  # noqa: F401


def get_backend(preferred: str = "auto") -> GribBackend:
    """Return a ready-to-use GRIB encoding backend.

    Args:
        preferred: One of ``"auto"``, ``"mars2grib"``, ``"mars2grib_native"``,
            or ``"eccodes"``.
            * ``"auto"`` – try mars2grib (hybrid), fall back to eccodes.
            * ``"mars2grib"`` – hybrid mars2grib + eccodes section 3.
            * ``"mars2grib_native"`` – pure mars2grib via compound gridSpec
              + ``skipSection3``. Requires ``ECCODES_ECKIT_GEO`` env var.
            * ``"eccodes"`` – use eccodes directly.

    Returns:
        A :class:`GribBackend` instance.

    Raises:
        ImportError: If the explicitly requested backend is not installed.
    """
    if preferred == "mars2grib_native":
        from .mars2grib_native_backend import Mars2GribNativeBackend

        return Mars2GribNativeBackend()

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
