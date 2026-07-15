"""Voltage-level baseline-shift features.

Approximates the extracellular/intracellular distinction used in manual
EPG classification (Bonani et al. 2010 Table 1): intracellular waveforms
(e.g. E1, E2) show a sustained DC shift relative to the insect's
non-probing baseline, while extracellular waveforms (e.g. C, D, G) sit
close to it. This is a coarse proxy, not a direct electrophysiological
measurement -- it needs the session's non-probing baseline voltage,
passed in via ``context["np_baseline_v"]`` (see
:func:`epg_tool.features.baseline.estimate_np_baseline`).
"""

from __future__ import annotations

import numpy as np

from .base import register_feature


def estimate_np_baseline(samples: np.ndarray, np_mask: np.ndarray) -> float:
    """Median voltage over all non-probing (Np) samples in a session,
    used as the zero-reference for baseline-shift features. Falls back
    to the median of the whole trace if no Np samples are labeled."""
    if np_mask.any():
        return float(np.median(samples[np_mask]))
    return float(np.median(samples))


@register_feature("baseline")
def baseline_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    np_baseline_v = context.get("np_baseline_v", 0.0)
    window_mean = float(window.mean())

    n = len(window)
    if n >= 2:
        x = np.arange(n, dtype=float)
        slope_v_per_s = float(np.polyfit(x, window, 1)[0] * sample_rate_hz)
    else:
        slope_v_per_s = 0.0

    return {
        "baseline_shift_v": window_mean - np_baseline_v,
        "baseline_abs_shift_v": abs(window_mean - np_baseline_v),
        "baseline_linear_trend_v_per_s": slope_v_per_s,
    }
