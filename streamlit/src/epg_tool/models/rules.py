"""Transparent rule-based classifier driven entirely by a species profile's
``rule_based_thresholds`` config -- no per-species logic lives in this
module, only the generic scoring mechanics.

Each waveform's rule is scored against a window's extracted features
(dominant frequency, %amplitude, baseline shift); the waveform with the
highest score wins. Rules with a ``components`` sub-dict (e.g. E2's
waves/peaks in the Bonani et al. profile) are scored as the best of their
components. This is meant as an interpretable baseline/fallback -- expect
the trained ML models to outperform it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from epg_tool.species.profile import SpeciesProfile

_FREQ_FEATURE = "spec_dominant_freq_hz"
_AMP_FEATURE = "amp_pct_fullscale"
_BASELINE_FEATURE = "baseline_abs_shift_v"


def _match_range(value: float, rng: list | None) -> float:
    """1.0 inside [lo, hi], decaying linearly outside; 0.5 (neutral) if the
    rule doesn't constrain this feature at all."""
    if rng is None:
        return 0.5
    lo, hi = rng
    lo = -math.inf if lo is None else lo
    hi = math.inf if hi is None else hi
    if lo <= value <= hi:
        return 1.0
    span = (hi - lo) if math.isfinite(hi - lo) else max(abs(value), 1.0)
    dist = (lo - value) if value < lo else (value - hi)
    return max(0.0, 1.0 - dist / span) if span > 0 else 0.0


def _voltage_level_score(voltage_level: str | None, baseline_abs_shift: float) -> float:
    """Extracellular waveforms sit close to the non-probing baseline;
    intracellular ones (E1, E2) show a sustained shift away from it
    (physically, the ~1-2V step visible whenever the stylet tip crosses
    into a cell). 0.5V is an arbitrary but reasonable saturation point --
    tune per species if real intracellular shifts run smaller/larger."""
    if voltage_level == "extracellular":
        return max(0.0, 1.0 - baseline_abs_shift / 0.5)
    if voltage_level == "intracellular":
        return min(1.0, baseline_abs_shift / 0.5)
    return 0.5  # unspecified -> neutral, don't let it dominate the score


def _score_simple_rule(rule: dict, features: dict) -> float:
    baseline_abs_shift = features.get(_BASELINE_FEATURE, 0.0)

    if rule.get("voltage_level") == "baseline":
        # Np-style rule: flat trace close to its own baseline, i.e. both
        # low amplitude AND close to the reference voltage -- frequency
        # content alone can't distinguish it from a low-shift extracellular
        # waveform like D, so amplitude has to carry that half of the score.
        shift_score = max(0.0, 1.0 - baseline_abs_shift / 0.5)
        amp_score = max(0.0, 1.0 - features.get(_AMP_FEATURE, 0.0) / 10.0)
        return 0.5 * shift_score + 0.5 * amp_score

    freq_score = _match_range(features.get(_FREQ_FEATURE, 0.0), rule.get("frequency_hz"))
    amp_score = _match_range(features.get(_AMP_FEATURE, 0.0), rule.get("amplitude_pct"))
    level_score = _voltage_level_score(rule.get("voltage_level"), baseline_abs_shift)
    return 0.45 * freq_score + 0.25 * amp_score + 0.30 * level_score


def _score_rule(rule: dict, features: dict) -> float:
    if "components" in rule:
        return max(_score_simple_rule(sub, features) for sub in rule["components"].values())
    return _score_simple_rule(rule, features)


class RuleBasedClassifier:
    """Heuristic classifier using only ``profile.rule_based_thresholds`` --
    no trained parameters, no persistence needed."""

    def __init__(self, profile: SpeciesProfile):
        self.profile = profile

    def classify_one(self, features: dict) -> int:
        best_code: int | None = None
        best_score = -1.0
        for waveform in self.profile.waveforms:
            rule = self.profile.rule_based_thresholds.get(waveform.label)
            if rule is None:
                continue
            score = _score_rule(rule, features)
            if score > best_score:
                best_score = score
                best_code = waveform.code

        if best_code is None:
            raise ValueError(
                f"Species profile {self.profile.name!r} has no rule_based_thresholds entries"
            )
        return best_code

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.classify_one(row._asdict()) for row in X.itertuples(index=False)])
