"""Tabular ML model wrappers (Random Forest / XGBoost) with a consistent
fit/predict/save/load interface, operating on extracted-feature DataFrames
keyed by waveform *code* (int), not label string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class TabularModel:
    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    label_encoder: LabelEncoder = field(default_factory=LabelEncoder)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabularModel":
        self.feature_names = list(X.columns)
        y_enc = self.label_encoder.fit_transform(y)
        self.estimator.fit(X[self.feature_names], y_enc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_enc = self.estimator.predict(X[self.feature_names])
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.feature_names])

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_

    def feature_importances(self) -> pd.Series | None:
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return None
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TabularModel":
        import joblib

        return joblib.load(Path(path))


def random_forest_model(**kwargs) -> TabularModel:
    from sklearn.ensemble import RandomForestClassifier

    params = dict(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    params.update(kwargs)
    return TabularModel(estimator=RandomForestClassifier(**params))


def xgboost_model(**kwargs) -> TabularModel:
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    params.update(kwargs)
    return TabularModel(estimator=XGBClassifier(**params))
