"""Unit tests for the shared valid-time / anoffset helpers in encoder.py."""

from datetime import timedelta

import numpy as np
import pytest

from covjsonkit.encoder.encoder import anoffset_timedelta, anoffset_value, valid_time


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
        assert valid_time("20250101T000000", 6, anoffset=6) == "2025-01-01T00:00:00Z"

    def test_anoffset_partial(self):
        # 00:00 + step 6h - anoffset 3h = 03:00
        assert valid_time("20250101T000000", 6, anoffset=3) == "2025-01-01T03:00:00Z"

    def test_anoffset_zero_is_noop(self):
        assert valid_time("20250101T000000", 6, anoffset=0) == "2025-01-01T06:00:00Z"

    def test_anoffset_with_time(self):
        # 12:00 run + step 6h - anoffset 30h = previous day 12:00 (fu-edzw fixture)
        assert valid_time("2026-05-01", 6, np.timedelta64(12, "h"), anoffset=30) == "2026-04-30T12:00:00Z"


class TestAnoffsetValue:
    @pytest.mark.parametrize("value", [6, "6", "6h", [6], ("6",), np.int64(6), timedelta(hours=6)])
    def test_integer_hours(self, value):
        assert anoffset_value(value) == 6

    def test_multiple_values_rejected(self):
        with pytest.raises(ValueError, match="single anoffset"):
            anoffset_value([6, 18])

    def test_unparseable_rejected(self):
        with pytest.raises(ValueError):
            anoffset_value("garbage")


class TestAnoffsetTimedelta:
    def test_efas_with_anoffset(self):
        mm = {"class": "ce", "stream": "efas", "anoffset": 6}
        assert anoffset_timedelta(mm) == timedelta(hours=6)

    def test_efas_string_anoffset(self):
        mm = {"class": "ce", "stream": "efas", "anoffset": "90"}
        assert anoffset_timedelta(mm) == timedelta(hours=90)

    def test_efas_without_anoffset(self):
        mm = {"class": "ce", "stream": "efas"}
        assert anoffset_timedelta(mm) == timedelta(0)

    def test_none_is_zero(self):
        mm = {"class": "ce", "stream": "efas", "anoffset": None}
        assert anoffset_timedelta(mm) == timedelta(0)

    def test_non_efas_stream_ignored(self):
        # anoffset present but stream is not efas -> not applied.
        mm = {"class": "od", "stream": "oper", "anoffset": 6}
        assert anoffset_timedelta(mm) == timedelta(0)

    def test_efcl_stream_ignored(self):
        # Reanalysis (efcl) must not receive the efas correction.
        mm = {"class": "ce", "stream": "efcl", "anoffset": 6}
        assert anoffset_timedelta(mm) == timedelta(0)

    def test_wrong_class_ignored(self):
        mm = {"class": "od", "stream": "efas", "anoffset": 6}
        assert anoffset_timedelta(mm) == timedelta(0)
