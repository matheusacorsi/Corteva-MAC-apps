from __future__ import annotations

import numpy as np
import pytest

from conftest import REAL_ANA, REAL_D01, requires_real_data
from epg_tool.io.session import build_session, load_d0x_session


def test_load_d0x_session_concatenates_in_order(tmp_d0x_pair):
    p1, _p2, expected_samples = tmp_d0x_pair
    samples, rate, paths = load_d0x_session(p1)
    np.testing.assert_allclose(samples, expected_samples, atol=1e-6)
    assert rate == pytest.approx(100.0)
    assert [p.name for p in paths] == ["insectA.D01", "insectA.D02"]


def test_load_d0x_session_rejects_mismatched_rates(tmp_path):
    from conftest import make_d0x_bytes

    (tmp_path / "z.D01").write_bytes(make_d0x_bytes([0.0], sample_rate_hz="100,000"))
    (tmp_path / "z.D02").write_bytes(make_d0x_bytes([0.0], sample_rate_hz="50,000"))
    with pytest.raises(ValueError, match="Inconsistent sample rates"):
        load_d0x_session(tmp_path / "z.D01")


@requires_real_data
def test_build_session_against_real_files():
    session = build_session(REAL_D01, REAL_ANA, insect_id="PsilideoMudaT1-3-ch6")
    assert session.sample_rate_hz == pytest.approx(100.0)
    assert session.duration_s == pytest.approx(29160.0)
    assert len(session.segments) == 19

    # spot-check the first real segment: code=1 (Np), 0.00s -> 480.32s
    first = session.segments[0]
    assert first.code == 1
    assert first.start_s == pytest.approx(0.0)
    assert first.end_s == pytest.approx(480.32)
    assert first.start_idx == 0
    assert first.end_idx == 48032

    df = session.to_dataframe()
    assert len(df) == 19
    assert set(df["code"]) == {1, 2, 3, 4, 5, 7}
