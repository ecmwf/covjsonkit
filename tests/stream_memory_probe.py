"""Peak RSS of streaming one large MultiPoint coverage, as a subprocess probe.

Run it directly to measure; it prints one JSON line::

    python tests/stream_memory_probe.py [n_points]

The blocks are duck-typed stand-ins for polytope-mars' ``CoordsBlock`` / ``ValuesBlock`` / ``GroupEnd``.
The process' peak RSS is reset once the arrays are allocated (``/proc/self/clear_refs``, so the peak a
parent process left behind does not mask the measurement), hence the reported growth is what encoding
costs on top of the data itself.  ``test_stream_memory.py`` asserts the bounds.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
from types import SimpleNamespace

import numpy as np

from covjsonkit.stream import CovjsonStreamEncoder


def peak_rss_bytes() -> int:
    """Peak resident set size of this process since the last :func:`reset_peak_rss`."""
    try:
        with open("/proc/self/status", "r") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def reset_peak_rss() -> bool:
    """Forget the peak RSS so far (Linux >= 4.0); ``False`` when the kernel does not allow it."""
    try:
        with open("/proc/self/clear_refs", "w") as clear_refs:
            clear_refs.write("5")
    except OSError:
        return False
    return True


def one_coverage(n_points: int):
    """Header and blocks of one coverage of ``n_points`` points with one parameter."""
    param = SimpleNamespace(id="167", shortname="2t", name="2 metre temperature", unit="K", description="")
    header = SimpleNamespace(
        feature_type="boundingbox",
        domain_type="MultiPoint",
        time_axis="date",
        parameters=(param,),
        mars_metadata={},
        extra={},
    )
    group = SimpleNamespace(
        index=0,
        path={},
        t=("2024-01-01T00:00:00Z",),
        params=("167",),
        levels=(),
        n_points=n_points,
        n_bands=1,
        mars_metadata={"class": "od", "stream": "oper", "number": 0, "step": 0},
    )
    coords = SimpleNamespace(
        group=group,
        band=0,
        offset=0,
        lat=np.linspace(-90.0, 90.0, n_points),
        lon=np.linspace(0.0, 359.9, n_points),
    )
    values = SimpleNamespace(
        group=group,
        param="167",
        level=None,
        band=0,
        offset=0,
        values=np.linspace(200.0, 320.0, n_points),
    )
    return header, [coords, values, SimpleNamespace(group=group)]


def measure(n_points: int) -> dict:
    header, blocks = one_coverage(n_points)
    encoder = CovjsonStreamEncoder()
    gc.collect()
    peak_reset = reset_peak_rss()
    baseline = peak_rss_bytes()

    total = len(encoder.begin(header))
    n_fragments = 0
    largest = 0
    for block in blocks:
        for fragment in encoder.encode_iter(block):
            n_fragments += 1
            largest = max(largest, len(fragment))
            total += len(fragment)
    total += len(encoder.end())

    peak = peak_rss_bytes()
    return {
        "n_points": n_points,
        "document_bytes": total,
        "n_fragments": n_fragments,
        "largest_fragment_bytes": largest,
        "max_fragment_bytes": encoder.max_fragment_bytes,
        "peak_was_reset": peak_reset,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak,
        "peak_rss_growth_bytes": peak - baseline,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the peak RSS of streaming one large coverage.")
    parser.add_argument("n_points", nargs="?", type=int, default=5_000_000)
    print(json.dumps(measure(parser.parse_args().n_points)))
