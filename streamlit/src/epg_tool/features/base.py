"""Pluggable feature-extraction registry.

Each extractor is a function ``(window, sample_rate_hz, context) -> dict[str, float]``
registered under a short name. Adding a new feature means writing one function
and decorating it -- nothing else in the codebase needs to change.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class FeatureExtractor(Protocol):
    def __call__(
        self, window: np.ndarray, sample_rate_hz: float, context: dict
    ) -> dict[str, float]: ...


_REGISTRY: dict[str, FeatureExtractor] = {}


def register_feature(name: str) -> Callable[[FeatureExtractor], FeatureExtractor]:
    def decorator(fn: FeatureExtractor) -> FeatureExtractor:
        if name in _REGISTRY:
            raise ValueError(f"Feature extractor {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_extractors() -> list[str]:
    return sorted(_REGISTRY)


def extract_features(
    window: np.ndarray,
    sample_rate_hz: float,
    extractors: list[str] | None = None,
    context: dict | None = None,
) -> dict[str, float]:
    """Run the given (or all registered) extractors over one window and
    merge their outputs into a single flat feature dict."""
    names = extractors if extractors is not None else available_extractors()
    context = context or {}
    features: dict[str, float] = {}
    for name in names:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown feature extractor {name!r}. Available: {available_extractors()}")
        features.update(_REGISTRY[name](window, sample_rate_hz, context))
    return features
