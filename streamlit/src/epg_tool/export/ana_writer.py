"""Writes classifier predictions back out as a Stylet+-compatible .ANA
file, so results can be loaded into Stylet+ for manual review/correction.

Mirrors the format verified in ``epg_tool.io.ana``: UTF-16LE with BOM,
CRLF line endings, tab-delimited ``code\\ttime(comma-decimal)\\tmarker``,
terminated by a sentinel row. The marker column (voltage in mV at the
segment's start sample) is redundant with the .D0x trace, same as in a
genuine Stylet+ export, but Stylet+ expects the column to be present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epg_tool.io.session import EPGSession, LabeledSegment
from epg_tool.species.profile import SpeciesProfile


def predictions_to_segments(
    window_start_idx: list[int], window_end_idx: list[int], pred_codes: list[int]
) -> list[tuple[int, int, int]]:
    """Merge consecutive same-label, contiguous windows into segments:
    (code, start_idx, end_idx)."""
    if not pred_codes:
        return []

    segments = []
    cur_code = pred_codes[0]
    cur_start = window_start_idx[0]
    cur_end = window_end_idx[0]
    for code, start_idx, end_idx in zip(pred_codes[1:], window_start_idx[1:], window_end_idx[1:]):
        if code == cur_code and start_idx == cur_end:
            cur_end = end_idx
        else:
            segments.append((cur_code, cur_start, cur_end))
            cur_code, cur_start, cur_end = code, start_idx, end_idx
    segments.append((cur_code, cur_start, cur_end))
    return segments


def _format_ana_time(seconds: float) -> str:
    return f"{seconds:.2f}".replace(".", ",")


def build_ana_bytes(
    segments_s: list[tuple[int, float, float]],
    samples: np.ndarray,
    sample_rate_hz: float,
    sentinel_code: int,
    recording_end_s: float,
) -> bytes:
    """Build .ANA file bytes from (code, start_s, end_s) segments. The
    marker column is filled with the real voltage (mV) at each segment's
    start sample, matching genuine Stylet+ output exactly."""
    n = len(samples)
    lines = []
    for code, start_s, _end_s in segments_s:
        idx = min(round(start_s * sample_rate_hz), n - 1)
        marker = round(float(samples[idx]) * 1000)
        lines.append(f"{code}\t{_format_ana_time(start_s)}\t{marker}")
    lines.append(f"{sentinel_code}\t{_format_ana_time(recording_end_s)}\t0")
    text = "\r\n".join(lines) + "\r\n"
    return b"\xff\xfe" + text.encode("utf-16-le")


def predictions_to_labeled_segments(
    windows: list, pred_codes: np.ndarray, sample_rate_hz: float
) -> list[LabeledSegment]:
    """Merge window-level predictions into ``LabeledSegment``s (same shape
    as ground-truth segments), so predictions can reuse the plotting and
    parameter-computation code written for ground truth."""
    window_start_idx = [w.start_idx for w in windows]
    window_end_idx = [w.end_idx for w in windows]
    raw_segments = predictions_to_segments(window_start_idx, window_end_idx, list(pred_codes))
    return [
        LabeledSegment(
            code=code,
            start_s=start_idx / sample_rate_hz,
            end_s=end_idx / sample_rate_hz,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        for code, start_idx, end_idx in raw_segments
    ]


def build_ana_from_windows(
    windows: list,
    pred_codes: np.ndarray,
    session: EPGSession,
    profile: SpeciesProfile,
) -> bytes:
    """High-level entry point: windows + their predicted codes -> .ANA
    file bytes, ready to write to disk or hand to a Streamlit download
    button."""
    sample_rate_hz = session.sample_rate_hz
    labeled_segments = predictions_to_labeled_segments(windows, pred_codes, sample_rate_hz)
    segments_s = [(seg.code, seg.start_s, seg.end_s) for seg in labeled_segments]

    sentinel_code = next(iter(profile.sentinel_codes))
    recording_end_s = (
        session.recording_end_s if session.recording_end_s is not None else len(session.samples) / sample_rate_hz
    )
    return build_ana_bytes(segments_s, session.samples, sample_rate_hz, sentinel_code, recording_end_s)


def write_ana_file(
    path: str | Path,
    windows: list,
    pred_codes: np.ndarray,
    session: EPGSession,
    profile: SpeciesProfile,
) -> None:
    data = build_ana_from_windows(windows, pred_codes, session, profile)
    Path(path).write_bytes(data)
