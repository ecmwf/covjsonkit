"""Unit tests for the shared valid-time / anoffset helpers in encoder.py."""

import numpy as np

from covjsonkit.encoder.encoder import anoffset_hours, efas_anoffset_hours, valid_time


class TestValidTime:
    def test_integer_step(self):
        assert valid_time("20250101T000000", 6) == "2025-01-01T06:00:00Z"

    def test_z_suffixed_date(self):
        assert valid_time("2025-01-01T00:00:00Z", 6) == "2025-01-01T06:00:00Z"

    def test_string_step_with_hour_suffix(self):
        assert valid_time("20250101T000000", "6h") == "2025-01-01T06:00:00Z"

    def test_timedelta_step(self):
        assert valid_time("20250101T000000", np.timedelta64(6, "h")) == "2025-01-01T06:00:00Z"

    def test_single_element_list_step(self):
        assert valid_time("20250101T000000", [6]) == "2025-01-01T06:00:00Z"

    def test_anoffset_subtracted(self):
        # 00:00 + step 6h - anoffset 6h = 00:00
        assert valid_time("20250101T000000", 6, 6) == "2025-01-01T00:00:00Z"

    def test_anoffset_partial(self):
        # 00:00 + step 6h - anoffset 3h = 03:00
        assert valid_time("20250101T000000", 6, 3) == "2025-01-01T03:00:00Z"

    def test_anoffset_zero_is_noop(self):
        assert valid_time("20250101T000000", 6, 0) == "2025-01-01T06:00:00Z"


class TestAnoffsetHours:
    def test_absent_returns_zero(self):
        assert anoffset_hours({}) == 0

    def test_none_returns_zero(self):
        assert anoffset_hours({"anoffset": None}) == 0

    def test_int_value(self):
        assert anoffset_hours({"anoffset": 6}) == 6

    def test_list_value(self):
        assert anoffset_hours({"anoffset": [6]}) == 6

    def test_hour_suffixed_string(self):
        assert anoffset_hours({"anoffset": "6h"}) == 6

    def test_unparseable_returns_zero(self):
        assert anoffset_hours({"anoffset": "garbage"}) == 0


class TestEfasAnoffsetHours:
    def test_efas_with_anoffset(self):
        mm = {"class": "ce", "stream": "efas", "anoffset": 6}
        assert efas_anoffset_hours(mm) == 6

    def test_efas_without_anoffset(self):
        mm = {"class": "ce", "stream": "efas"}
        assert efas_anoffset_hours(mm) == 0

    def test_non_efas_stream_ignored(self):
        # anoffset present but stream is not efas -> not applied.
        mm = {"class": "od", "stream": "oper", "anoffset": 6}
        assert efas_anoffset_hours(mm) == 0

    def test_efcl_stream_ignored(self):
        # Reanalysis (efcl) must not receive the efas correction.
        mm = {"class": "ce", "stream": "efcl", "anoffset": 6}
        assert efas_anoffset_hours(mm) == 0

    def test_wrong_class_ignored(self):
        mm = {"class": "od", "stream": "efas", "anoffset": 6}
        assert efas_anoffset_hours(mm) == 0
