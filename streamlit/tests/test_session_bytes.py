from __future__ import annotations

import numpy as np
import pytest

from conftest import make_ana_bytes, make_d0x_bytes
from epg_tool.io.session import build_session_from_bytes


def test_build_session_from_bytes_concatenates_and_labels():
    samples_01 = [0.01 * i for i in range(10)]
    samples_02 = [0.01 * i for i in range(10, 20)]
    d0x_files = [
        ("insectA.D02", make_d0x_bytes(samples_02)),
        ("insectA.D01", make_d0x_bytes(samples_01)),  # deliberately out of order
    ]
    ana_bytes = make_ana_bytes([(1, "0,00", 0), (2, "0,10", 0), (99, "0,20", 0)])

    session = build_session_from_bytes(d0x_files, ana_bytes, insect_id="insectA")
    np.testing.assert_allclose(session.samples, samples_01 + samples_02, atol=1e-6)
    assert session.sample_rate_hz == pytest.approx(100.0)
    assert len(session.segments) == 2
    assert session.segments[0].code == 1
    assert session.recording_end_s == pytest.approx(0.20)


def test_build_session_from_bytes_without_ana_has_no_segments():
    d0x_files = [("x.D01", make_d0x_bytes([0.1, 0.2, 0.3]))]
    session = build_session_from_bytes(d0x_files, ana_bytes=None, insect_id="x")
    assert session.segments == []
    assert session.recording_end_s == pytest.approx(3 / 100.0)


def test_build_session_from_bytes_rejects_mismatched_rates():
    d0x_files = [
        ("x.D01", make_d0x_bytes([0.0], sample_rate_hz="100,000")),
        ("x.D02", make_d0x_bytes([0.0], sample_rate_hz="50,000")),
    ]
    with pytest.raises(ValueError, match="Inconsistent sample rates"):
        build_session_from_bytes(d0x_files)


def test_build_session_from_bytes_requires_at_least_one_file():
    with pytest.raises(ValueError, match="No .D0x files"):
        build_session_from_bytes([])
