from __future__ import annotations

import numpy as np
import pytest

from conftest import REAL_ANA, REAL_D01, requires_real_data
from epg_tool.export.ana_writer import predictions_to_segments, write_ana_file
from epg_tool.features.windowing import make_windows
from epg_tool.io.ana import parse_ana_rows, rows_to_segments
from epg_tool.io.session import build_session
from epg_tool.species.profile import load_profile


def test_predictions_to_segments_merges_contiguous_same_label():
    segments = predictions_to_segments(
        window_start_idx=[0, 10, 20, 30],
        window_end_idx=[10, 20, 30, 40],
        pred_codes=[1, 1, 2, 2],
    )
    assert segments == [(1, 0, 20), (2, 20, 40)]


def test_predictions_to_segments_breaks_on_gap_even_if_same_label():
    segments = predictions_to_segments(
        window_start_idx=[0, 10, 25],
        window_end_idx=[10, 20, 35],
        pred_codes=[1, 1, 1],
    )
    # gap between idx 20 and 25 -> can't merge across it
    assert segments == [(1, 0, 20), (1, 25, 35)]


def test_predictions_to_segments_empty_input():
    assert predictions_to_segments([], [], []) == []


@requires_real_data
def test_write_ana_roundtrip_with_perfect_predictions(tmp_path):
    profile = load_profile("diaphorina_citri")
    session = build_session(REAL_D01, REAL_ANA, insect_id="PsilideoMudaT1-3-ch6")

    # Use fine-grained 1s windows so window boundaries can closely track
    # the original (irregular-length) ground-truth segment boundaries.
    windows = make_windows(session, window_s=1.0, step_s=1.0)
    pred_codes = np.array([w.label_code for w in windows])  # perfect "predictions"

    out_path = tmp_path / "predicted_.ANA"
    write_ana_file(out_path, windows, pred_codes, session, profile)

    rows = parse_ana_rows(out_path)
    written_segments, written_end = rows_to_segments(rows, sentinel_codes=profile.sentinel_codes)

    # Same waveform sequence as the original ground truth...
    original_codes = [s.code for s in session.segments]
    written_codes = [s.code for s in written_segments]
    assert written_codes == original_codes

    # ...and boundaries within one window (1s) of the original, since
    # predictions are only as precise as the window size.
    for orig, written in zip(session.segments, written_segments):
        assert written.start_s == pytest.approx(orig.start_s, abs=1.0)

    assert written_end == pytest.approx(session.recording_end_s, abs=1.0)


@requires_real_data
def test_write_ana_output_is_utf16_with_bom(tmp_path):
    profile = load_profile("diaphorina_citri")
    session = build_session(REAL_D01, REAL_ANA, insect_id="PsilideoMudaT1-3-ch6")
    windows = make_windows(session, window_s=1.0, step_s=1.0)
    pred_codes = np.array([w.label_code for w in windows])

    out_path = tmp_path / "predicted_.ANA"
    write_ana_file(out_path, windows, pred_codes, session, profile)

    raw = out_path.read_bytes()
    assert raw[:2] == b"\xff\xfe"
    text = raw.decode("utf-16")
    assert text.endswith("\r\n")
    assert text.splitlines()[-1].startswith("99\t")
