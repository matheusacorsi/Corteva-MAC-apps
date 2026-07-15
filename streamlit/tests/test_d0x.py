from __future__ import annotations

import numpy as np
import pytest

from conftest import REAL_ANA, REAL_D01, make_d0x_bytes, requires_real_data
from epg_tool.io.d0x import find_d0x_series, parse_d0x_file, parse_d0x_header, sort_key


def test_parse_header_fields():
    data = make_d0x_bytes(
        samples=[0.0],
        recorded_at="13-10-2025 10:16:30",
        rec_time_hours="8,10",
        sample_rate_hz="100,000",
    )
    header = parse_d0x_header(data)
    assert header.recorded_at.isoformat() == "2025-10-13T10:16:30"
    assert header.rec_time_hours == pytest.approx(8.10)
    assert header.sample_rate_hz == pytest.approx(100.0)


def test_parse_samples_roundtrip(tmp_path):
    samples = [0.1, -0.2, 0.3, -0.4]
    path = tmp_path / "x.D01"
    path.write_bytes(make_d0x_bytes(samples))
    d0x = parse_d0x_file(path)
    np.testing.assert_allclose(d0x.samples, samples, atol=1e-6)
    assert d0x.n_samples == 4


def test_body_not_multiple_of_4_raises(tmp_path):
    data = make_d0x_bytes([0.0, 0.0])
    path = tmp_path / "bad.D01"
    path.write_bytes(data + b"\x00\x00\x00")  # 3 stray bytes
    with pytest.raises(ValueError, match="multiple of 4"):
        parse_d0x_file(path)


def test_missing_terminator_raises():
    with pytest.raises(ValueError, match="terminator"):
        parse_d0x_header(b"not a real header")


def test_sort_key_extracts_numeric_suffix(tmp_path):
    assert sort_key(tmp_path / "foo.D01") == 1
    assert sort_key(tmp_path / "foo.D09") == 9
    assert sort_key(tmp_path / "foo.D12") == 12


def test_find_d0x_series_sorted(tmp_path):
    for n in [3, 1, 2]:
        (tmp_path / f"bar.D0{n}").write_bytes(make_d0x_bytes([0.0]))
    series = find_d0x_series(tmp_path / "bar.D02")
    assert [p.name for p in series] == ["bar.D01", "bar.D02", "bar.D03"]


@requires_real_data
def test_real_d01_header_matches_inspection():
    d0x = parse_d0x_file(REAL_D01)
    assert d0x.header.sample_rate_hz == pytest.approx(100.0)
    assert d0x.n_samples == 360_000
    assert d0x.duration_s == pytest.approx(3600.0)


@requires_real_data
def test_real_full_series_duration_matches_ana_sentinel():
    from epg_tool.io.ana import parse_ana_rows

    series = find_d0x_series(REAL_D01)
    total_samples = sum(parse_d0x_file(p).n_samples for p in series)
    rate = parse_d0x_file(series[0]).header.sample_rate_hz
    total_duration = total_samples / rate

    rows = parse_ana_rows(REAL_ANA)
    sentinel_time = max(r.time_s for r in rows)
    assert total_duration == pytest.approx(sentinel_time, abs=0.01)
