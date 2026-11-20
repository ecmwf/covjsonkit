"""The streaming encoder's memory bound: peak RSS must not follow the size of a block.

The measurement runs in a subprocess (``stream_memory_probe.py``) so the peak RSS it reports is only
this encoding; run that module directly to see the numbers.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from covjsonkit.stream import DEFAULT_MAX_FRAGMENT_BYTES

PROBE = str(Path(__file__).with_name("stream_memory_probe.py"))
MB = 1024 * 1024


def _probe(n_points: int) -> dict:
    run = subprocess.run([sys.executable, PROBE, str(n_points)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    try:
        return json.loads(last)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"probe printed no measurement: {last!r}\n{run.stderr}") from exc


def test_peak_rss_growth_is_bounded_and_independent_of_the_block_size():
    small = _probe(1_000_000)
    big = _probe(5_000_000)
    if not (small["peak_was_reset"] and big["peak_was_reset"]):
        pytest.skip("this kernel does not support resetting the peak RSS via /proc/self/clear_refs")
    assert big["document_bytes"] > 300 * MB  # ~320 MB of JSON text out of 5M points
    assert big["largest_fragment_bytes"] <= DEFAULT_MAX_FRAGMENT_BYTES
    assert small["largest_fragment_bytes"] <= DEFAULT_MAX_FRAGMENT_BYTES
    assert big["peak_rss_growth_bytes"] < 100 * MB, big
    # five times the points for 5 times the text, but not for more memory
    assert big["peak_rss_growth_bytes"] < small["peak_rss_growth_bytes"] + 32 * MB, f"{small} vs {big}"
