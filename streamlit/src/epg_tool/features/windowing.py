"""Fixed-length windowing of an EPGSession, with majority-vote ground-truth
labels per window for supervised feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    start_idx: int
    end_idx: int
    samples: np.ndarray
    label_code: int | None
    label_purity: float  # fraction of the window covered by the dominant label


def build_sample_labels(n_samples: int, segments) -> np.ndarray:
    """Expand (code, start_idx, end_idx) segments into one code per sample.
    Unlabeled samples (shouldn't normally occur) get -1."""
    labels = np.full(n_samples, -1, dtype=np.int32)
    for seg in segments:
        labels[seg.start_idx : seg.end_idx] = seg.code
    return labels


def make_windows(
    session,
    window_s: float,
    step_s: float | None = None,
    min_purity: float = 0.0,
) -> list[Window]:
    """Slide a fixed-length window over ``session.samples``. Each window's
    label is the ground-truth code covering the largest share of its
    samples; windows below ``min_purity`` (e.g. ones straddling a
    transition) are dropped."""
    step_s = window_s if step_s is None else step_s
    window_len = round(window_s * session.sample_rate_hz)
    step_len = round(step_s * session.sample_rate_hz)
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_s and step_s must resolve to at least 1 sample")

    labels = build_sample_labels(len(session.samples), session.segments)

    windows: list[Window] = []
    n = len(session.samples)
    start = 0
    while start + window_len <= n:
        end = start + window_len
        codes, counts = np.unique(labels[start:end], return_counts=True)
        dominant_idx = int(np.argmax(counts))
        dominant_code = int(codes[dominant_idx])
        purity = float(counts[dominant_idx]) / window_len

        if dominant_code != -1 and purity >= min_purity:
            windows.append(
                Window(
                    start_idx=start,
                    end_idx=end,
                    samples=session.samples[start:end],
                    label_code=dominant_code,
                    label_purity=purity,
                )
            )
        start += step_len
    return windows


def make_inference_windows(
    samples: np.ndarray,
    sample_rate_hz: float,
    window_s: float,
    step_s: float | None = None,
) -> list[Window]:
    """Same fixed-length slicing as :func:`make_windows`, for recordings
    with no ground-truth .ANA (real inference on unlabeled data) --
    every window is kept, with ``label_code=None``."""
    step_s = window_s if step_s is None else step_s
    window_len = round(window_s * sample_rate_hz)
    step_len = round(step_s * sample_rate_hz)
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_s and step_s must resolve to at least 1 sample")

    windows: list[Window] = []
    n = len(samples)
    start = 0
    while start + window_len <= n:
        end = start + window_len
        windows.append(
            Window(start_idx=start, end_idx=end, samples=samples[start:end], label_code=None, label_purity=0.0)
        )
        start += step_len
    return windows
