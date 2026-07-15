"""Common interface both the tabular ML models and (loosely) the rule-based
classifier follow, so training/evaluation code doesn't need to special-case
model type."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class Classifier(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...
