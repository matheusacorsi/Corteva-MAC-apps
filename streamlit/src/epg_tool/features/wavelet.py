"""Discrete wavelet transform features (relative energy per decomposition level).

Captures transient/non-stationary structure that plain FFT band energy
smooths over -- useful for the sharp voltage steps at waveform transitions.
"""

from __future__ import annotations

import numpy as np
import pywt

from .base import register_feature

_WAVELET = "db4"
_MAX_LEVEL = 4


@register_feature("wavelet")
def wavelet_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    max_possible = pywt.dwt_max_level(len(window), pywt.Wavelet(_WAVELET).dec_len)
    level = max(1, min(_MAX_LEVEL, max_possible))
    if max_possible < 1:
        return {f"wavelet_energy_L{i}": 0.0 for i in range(1, _MAX_LEVEL + 1)}

    coeffs = pywt.wavedec(window, _WAVELET, level=level)
    energies = np.array([float(np.sum(c**2)) for c in coeffs])
    total = energies.sum() or 1.0
    relative = energies / total

    # coeffs[0] is the final approximation; coeffs[1:] are detail levels
    # from coarsest to finest. Report detail-level relative energies,
    # padding with 0 if fewer levels than _MAX_LEVEL were computed.
    detail_energies = relative[1:]
    features = {}
    for i in range(1, _MAX_LEVEL + 1):
        idx = i - 1
        features[f"wavelet_energy_L{i}"] = (
            float(detail_energies[idx]) if idx < len(detail_energies) else 0.0
        )
    return features
