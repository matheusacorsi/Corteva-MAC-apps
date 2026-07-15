"""Amplitude-statistics features.

``pct_amplitude`` follows the convention in Bonani et al. (2010) Table 1
footnote 1: peak-to-peak amplitude expressed as a percentage of the 5V
full-scale EPG-DC range.
"""

from __future__ import annotations

import numpy as np

from .base import register_feature

_FULL_SCALE_V = 5.0


@register_feature("amplitude")
def amplitude_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    peak_to_peak = float(window.max() - window.min())
    return {
        "amp_mean": float(window.mean()),
        "amp_std": float(window.std()),
        "amp_min": float(window.min()),
        "amp_max": float(window.max()),
        "amp_peak_to_peak": peak_to_peak,
        "amp_rms": float(np.sqrt(np.mean(window**2))),
        "amp_pct_fullscale": peak_to_peak / _FULL_SCALE_V * 100.0,
    }
