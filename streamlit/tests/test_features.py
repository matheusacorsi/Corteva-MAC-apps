from __future__ import annotations

import numpy as np
import pytest

from epg_tool.features import available_extractors, extract_features, make_windows
from epg_tool.features.baseline import estimate_np_baseline
from epg_tool.io.session import EPGSession, LabeledSegment


def test_available_extractors_registered():
    names = available_extractors()
    assert {"amplitude", "spectral", "wavelet", "slope", "baseline"} <= set(names)


def test_amplitude_features_basic():
    window = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    feats = extract_features(window, sample_rate_hz=100.0, extractors=["amplitude"])
    assert feats["amp_max"] == pytest.approx(2.0)
    assert feats["amp_min"] == pytest.approx(-1.0)
    assert feats["amp_peak_to_peak"] == pytest.approx(3.0)
    assert feats["amp_pct_fullscale"] == pytest.approx(3.0 / 5.0 * 100)


def test_spectral_features_detects_dominant_frequency():
    sample_rate_hz = 100.0
    t = np.arange(200) / sample_rate_hz
    signal = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz sine, well within EPG range
    feats = extract_features(signal, sample_rate_hz, extractors=["spectral"])
    assert feats["spec_dominant_freq_hz"] == pytest.approx(10.0, abs=1.0)
    assert 0.0 <= feats["spec_band_5_10hz"] <= 1.0


def test_wavelet_features_keys_present():
    window = np.random.default_rng(0).normal(size=64)
    feats = extract_features(window, sample_rate_hz=100.0, extractors=["wavelet"])
    assert set(feats) == {f"wavelet_energy_L{i}" for i in range(1, 5)}
    assert all(0.0 <= v <= 1.0 for v in feats.values())


def test_slope_features_zero_crossing_rate():
    sample_rate_hz = 100.0
    t = np.arange(100) / sample_rate_hz
    signal = np.sin(2 * np.pi * 5.0 * t)  # 5 Hz -> 10 zero crossings/s
    feats = extract_features(signal, sample_rate_hz, extractors=["slope"])
    assert feats["slope_zero_crossing_rate"] == pytest.approx(10.0, abs=2.0)


def test_baseline_features_shift_from_context():
    window = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    feats = extract_features(
        window, sample_rate_hz=100.0, extractors=["baseline"], context={"np_baseline_v": 0.2}
    )
    assert feats["baseline_shift_v"] == pytest.approx(0.8)
    assert feats["baseline_abs_shift_v"] == pytest.approx(0.8)


def test_estimate_np_baseline_uses_np_mask():
    samples = np.array([0.1, 0.1, 5.0, 0.1], dtype=np.float32)
    np_mask = np.array([True, True, False, True])
    assert estimate_np_baseline(samples, np_mask) == pytest.approx(0.1)


def test_estimate_np_baseline_falls_back_to_full_median_if_no_np():
    samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np_mask = np.array([False, False, False])
    assert estimate_np_baseline(samples, np_mask) == pytest.approx(2.0)


def _toy_session() -> EPGSession:
    sample_rate_hz = 10.0
    samples = np.zeros(100, dtype=np.float32)
    segments = [
        LabeledSegment(code=1, start_s=0.0, end_s=5.0, start_idx=0, end_idx=50),
        LabeledSegment(code=2, start_s=5.0, end_s=10.0, start_idx=50, end_idx=100),
    ]
    return EPGSession(
        insect_id="toy",
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        source_files=[],
        segments=segments,
    )


def test_make_windows_labels_by_majority():
    session = _toy_session()
    windows = make_windows(session, window_s=1.0)  # 10-sample windows, 10 windows total
    assert len(windows) == 10
    assert all(w.label_code == 1 for w in windows[:5])
    assert all(w.label_code == 2 for w in windows[5:])
    assert all(w.label_purity == pytest.approx(1.0) for w in windows)


def test_make_windows_straddling_boundary_uses_majority_and_purity():
    session = _toy_session()
    # 1.2s windows landing across the 5.0s Np/C boundary
    windows = make_windows(session, window_s=1.2, step_s=1.2)
    boundary_window = next(w for w in windows if w.start_idx <= 50 < w.end_idx)
    assert boundary_window.label_purity < 1.0


def test_make_windows_respects_min_purity():
    session = _toy_session()
    windows = make_windows(session, window_s=1.2, step_s=1.2, min_purity=0.99)
    assert all(w.label_purity >= 0.99 for w in windows)
    assert len(windows) < 10
