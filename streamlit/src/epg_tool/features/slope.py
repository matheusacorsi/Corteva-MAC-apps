"""Signal-derivative / slope features."""

from __future__ import annotations

import numpy as np

from .base import register_feature


@register_feature("slope")
def slope_features(window: np.ndarray, sample_rate_hz: float, context: dict) -> dict[str, float]:
    if len(window) < 2:
        return {"slope_mean_abs": 0.0, "slope_std": 0.0, "slope_zero_crossing_rate": 0.0}

    derivative = np.diff(window) * sample_rate_hz
    zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
    return {
        "slope_mean_abs": float(np.mean(np.abs(derivative))),
        "slope_std": float(np.std(derivative)),
        "slope_zero_crossing_rate": float(zero_crossings) / len(window) * sample_rate_hz,
    }
