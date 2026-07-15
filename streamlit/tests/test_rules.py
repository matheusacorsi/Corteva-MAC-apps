from __future__ import annotations

import pandas as pd

from epg_tool.models.rules import RuleBasedClassifier
from epg_tool.species.profile import load_profile


def test_rule_based_classifier_picks_c_for_c_like_features():
    profile = load_profile("diaphorina_citri")
    clf = RuleBasedClassifier(profile)
    # C: 45% amplitude, 11.5-19.0 Hz, extracellular/resistive
    features = {
        "spec_dominant_freq_hz": 15.0,
        "amp_pct_fullscale": 45.0,
        "baseline_abs_shift_v": 0.05,
    }
    assert clf.classify_one(features) == profile.label_to_code["C"]


def test_rule_based_classifier_picks_d_for_low_frequency_extracellular():
    profile = load_profile("diaphorina_citri")
    clf = RuleBasedClassifier(profile)
    # D: 1.0-3.5 Hz, extracellular/emf, no defined amplitude range
    features = {
        "spec_dominant_freq_hz": 2.0,
        "amp_pct_fullscale": 5.0,
        "baseline_abs_shift_v": 0.05,
    }
    assert clf.classify_one(features) == profile.label_to_code["D"]


def test_rule_based_classifier_picks_np_for_flat_baseline():
    profile = load_profile("diaphorina_citri")
    clf = RuleBasedClassifier(profile)
    features = {
        "spec_dominant_freq_hz": 0.5,
        "amp_pct_fullscale": 1.0,
        "baseline_abs_shift_v": 0.01,
    }
    assert clf.classify_one(features) == profile.label_to_code["Np"]


def test_rule_based_classifier_predict_over_dataframe():
    profile = load_profile("diaphorina_citri")
    clf = RuleBasedClassifier(profile)
    df = pd.DataFrame(
        [
            {"spec_dominant_freq_hz": 15.0, "amp_pct_fullscale": 45.0, "baseline_abs_shift_v": 0.05},
            {"spec_dominant_freq_hz": 0.5, "amp_pct_fullscale": 1.0, "baseline_abs_shift_v": 0.01},
        ]
    )
    preds = clf.predict(df)
    assert list(preds) == [profile.label_to_code["C"], profile.label_to_code["Np"]]
