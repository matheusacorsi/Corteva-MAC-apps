from __future__ import annotations

import numpy as np
import pytest

from epg_tool.species.profile import load_profile
from epg_tool.training.evaluate import (
    classification_report_df,
    confusion_matrix_df,
    evaluate,
    per_class_time_overlap,
    time_overlap_agreement,
)


def test_time_overlap_agreement_perfect_match():
    y = np.array([1, 2, 3, 1])
    assert time_overlap_agreement(y, y) == pytest.approx(1.0)


def test_time_overlap_agreement_partial_match():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 2, 2])
    assert time_overlap_agreement(y_true, y_pred) == pytest.approx(0.5)


def test_time_overlap_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        time_overlap_agreement([1, 2], [1])


def test_confusion_matrix_df_labels_by_waveform_name():
    profile = load_profile("diaphorina_citri")
    y_true = np.array([1, 1, 2, 4])  # Np, Np, C, E1
    y_pred = np.array([1, 2, 2, 4])
    cm = confusion_matrix_df(y_true, y_pred, profile)
    assert set(cm.index) == {"Np", "C", "E1"}
    assert cm.loc["Np", "C"] == 1  # one Np misclassified as C
    assert cm.loc["E1", "E1"] == 1


def test_per_class_time_overlap_values():
    profile = load_profile("diaphorina_citri")
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 2, 2, 2])
    result = per_class_time_overlap(y_true, y_pred, profile)
    assert result["Np"] == pytest.approx(0.5)
    assert result["C"] == pytest.approx(1.0)


def test_evaluate_bundles_all_metrics():
    profile = load_profile("diaphorina_citri")
    y_true = np.array([1, 1, 2, 2, 4])
    y_pred = np.array([1, 2, 2, 2, 4])
    result = evaluate(y_true, y_pred, profile)
    assert result.accuracy == pytest.approx(0.8)
    assert result.time_overlap_agreement == pytest.approx(0.8)
    assert isinstance(result.classification_report, type(classification_report_df(y_true, y_pred, profile)))
    assert "Np" in result.confusion_matrix.index
    assert "Np" in result.per_class_time_overlap.index
