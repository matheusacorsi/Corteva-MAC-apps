from __future__ import annotations

import pytest

from conftest import REAL_ANA, make_ana_bytes, requires_real_data
from epg_tool.io.ana import parse_ana_rows, rows_to_segments


def test_parse_rows_utf16_comma_decimal(tmp_path):
    data = make_ana_bytes(
        [
            (1, "0,00", 0),
            (2, "480,32", -113),
            (99, "600,0", 0),
        ]
    )
    path = tmp_path / "x_.ANA"
    path.write_bytes(data)
    rows = parse_ana_rows(path)
    assert [r.code for r in rows] == [1, 2, 99]
    assert rows[1].time_s == pytest.approx(480.32)
    assert rows[1].marker == -113


def test_rows_to_segments_drops_sentinel_and_closes_last_segment():
    from epg_tool.io.ana import AnaRow

    rows = [
        AnaRow(code=1, time_s=0.0, marker=0),
        AnaRow(code=2, time_s=10.0, marker=1),
        AnaRow(code=1, time_s=25.0, marker=2),
        AnaRow(code=99, time_s=40.0, marker=0),
    ]
    segments, recording_end = rows_to_segments(rows)
    assert recording_end == pytest.approx(40.0)
    assert len(segments) == 3
    assert segments[0].code == 1 and segments[0].start_s == 0.0 and segments[0].end_s == 10.0
    assert segments[-1].code == 1 and segments[-1].end_s == pytest.approx(40.0)
    assert all(s.code != 99 for s in segments)


def test_rows_to_segments_requires_at_least_two_rows():
    from epg_tool.io.ana import AnaRow

    with pytest.raises(ValueError):
        rows_to_segments([AnaRow(code=1, time_s=0.0, marker=0)])


@requires_real_data
def test_real_ana_file_matches_inspection():
    rows = parse_ana_rows(REAL_ANA)
    codes = {r.code for r in rows}
    assert codes == {1, 2, 3, 4, 5, 7, 99}

    segments, recording_end = rows_to_segments(rows)
    assert recording_end == pytest.approx(29160.0)
    assert all(s.code != 99 for s in segments)
    # 20 rows total, last one is the sentinel -> 19 real segments
    assert len(segments) == 19
