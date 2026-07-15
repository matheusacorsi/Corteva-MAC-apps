"""Classification metrics plus the EPG-specific "% time-overlap agreement
with ground truth" metric used in the DiscoEPG/ML4Insects literature.

Because windows are fixed-length, window-level accuracy already *is* the
time-weighted agreement (every window counts for the same amount of
signal time) -- ``time_overlap_agreement`` and ``accuracy`` coincide here.
They're kept as separate named functions because they answer different
questions (ML metric vs. domain metric) and could diverge if this is ever
extended to variable-length predicted segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from epg_tool.species.profile import SpeciesProfile


def _label_for(code: int, profile: SpeciesProfile) -> str:
    return profile.label_for_code(code) or f"code {code}"


def classification_report_df(y_true, y_pred, profile: SpeciesProfile) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [_label_for(c, profile) for c in labels]
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).transpose()


def confusion_matrix_df(y_true, y_pred, profile: SpeciesProfile) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    names = [_label_for(c, profile) for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(names, name="true"), columns=pd.Index(names, name="predicted"))


def time_overlap_agreement(y_true, y_pred) -> float:
    """Fraction of (equal-length) windows where the prediction matches
    ground truth -- equivalently, the fraction of recording *time* the
    model got right."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def per_class_time_overlap(y_true, y_pred, profile: SpeciesProfile) -> pd.Series:
    """Of the time truly labeled waveform X, what fraction did the model
    also label X (per-class recall, expressed as time-overlap)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results = {}
    for code in sorted(set(y_true)):
        mask = y_true == code
        results[_label_for(code, profile)] = float(np.mean(y_pred[mask] == code))
    return pd.Series(results)


@dataclass
class EvaluationResult:
    accuracy: float
    time_overlap_agreement: float
    classification_report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    per_class_time_overlap: pd.Series


def evaluate(y_true, y_pred, profile: SpeciesProfile) -> EvaluationResult:
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        time_overlap_agreement=time_overlap_agreement(y_true, y_pred),
        classification_report=classification_report_df(y_true, y_pred, profile),
        confusion_matrix=confusion_matrix_df(y_true, y_pred, profile),
        per_class_time_overlap=per_class_time_overlap(y_true, y_pred, profile),
    )
