"""FFT-based spectral features.

Frequency bands are chosen to bracket the ranges reported across the
Bonani et al. (2010) Table 1 waveforms (roughly 1-19 Hz) plus headroom on
both sides; they are not species-specific and apply to any insect's
DC-EPG signal sampled well above ~40 Hz.
"""

from __future__ import annotations

import numpy as np

from .base import register_feature

_BAND_EDGES_HZ = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]


@register_feature("spectral")
def spectral_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    n = len(window)
    if n < 4:
        return {
            "spec_dominant_freq_hz": 0.0,
            "spec_centroid_hz": 0.0,
            **{f"spec_band_{lo:g}_{hi:g}hz": 0.0 for lo, hi in zip(_BAND_EDGES_HZ, _BAND_EDGES_HZ[1:])},
        }

    windowed = window * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    power = spectrum**2

    # Ignore the DC bin (index 0) when finding the dominant oscillation.
    ac_power = power[1:]
    ac_freqs = freqs[1:]
    if ac_power.sum() > 0:
        dominant_freq = float(ac_freqs[np.argmax(ac_power)])
        centroid = float(np.sum(ac_freqs * ac_power) / np.sum(ac_power))
    else:
        dominant_freq = 0.0
        centroid = 0.0

    total_power = power.sum() or 1.0
    band_features = {}
    for lo, hi in zip(_BAND_EDGES_HZ, _BAND_EDGES_HZ[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        band_features[f"spec_band_{lo:g}_{hi:g}hz"] = float(power[mask].sum() / total_power)

    return {
        "spec_dominant_freq_hz": dominant_freq,
        "spec_centroid_hz": centroid,
        **band_features,
    }
