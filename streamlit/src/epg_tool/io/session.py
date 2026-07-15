"""Stitches multi-file .D0x recordings into one continuous trace and aligns
.ANA-labeled segments against it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ana import AnaSegment, parse_ana_bytes, parse_ana_file, rows_to_segments
from .d0x import D0xFile, find_d0x_series, parse_d0x_bytes, parse_d0x_file, sort_key


@dataclass
class LabeledSegment:
    code: int
    start_s: float
    end_s: float
    start_idx: int
    end_idx: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class EPGSession:
    insect_id: str
    samples: np.ndarray  # float32 volts, continuous across all .D0x files
    sample_rate_hz: float
    source_files: list[Path]
    segments: list[LabeledSegment] = field(default_factory=list)
    recording_end_s: float | None = None

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate_hz

    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.samples)) / self.sample_rate_hz

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "code": seg.code,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "duration_s": seg.duration_s,
                    "start_idx": seg.start_idx,
                    "end_idx": seg.end_idx,
                }
                for seg in self.segments
            ]
        )


def load_d0x_session(any_d0x_file: str | Path) -> tuple[np.ndarray, float, list[Path]]:
    """Load and concatenate every .D0x sibling of ``any_d0x_file`` in
    numeric-suffix order. Raises if sample rates disagree across files."""
    paths = find_d0x_series(any_d0x_file)
    if not paths:
        raise ValueError(f"No .D0x files found alongside {any_d0x_file}")

    files: list[D0xFile] = [parse_d0x_file(p) for p in paths]
    rates = {f.header.sample_rate_hz for f in files}
    if len(rates) > 1:
        raise ValueError(f"Inconsistent sample rates across {paths}: {rates}")

    samples = np.concatenate([f.samples for f in files])
    return samples, rates.pop(), paths


def _to_labeled_segments(ana_segments: list[AnaSegment], sample_rate_hz: float) -> list[LabeledSegment]:
    return [
        LabeledSegment(
            code=seg.code,
            start_s=seg.start_s,
            end_s=seg.end_s,
            start_idx=round(seg.start_s * sample_rate_hz),
            end_idx=round(seg.end_s * sample_rate_hz),
        )
        for seg in ana_segments
    ]


def build_session(
    any_d0x_file: str | Path,
    ana_file: str | Path,
    insect_id: str | None = None,
    sentinel_codes: frozenset[int] = frozenset({99}),
) -> EPGSession:
    samples, sample_rate_hz, source_files = load_d0x_session(any_d0x_file)
    ana_segments, recording_end_s = parse_ana_file(ana_file, sentinel_codes=sentinel_codes)

    return EPGSession(
        insect_id=insect_id or Path(any_d0x_file).stem,
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        source_files=source_files,
        segments=_to_labeled_segments(ana_segments, sample_rate_hz),
        recording_end_s=recording_end_s,
    )


def build_session_from_bytes(
    d0x_files: list[tuple[str, bytes]],
    ana_bytes: bytes | None = None,
    insect_id: str = "uploaded",
    sentinel_codes: frozenset[int] = frozenset({99}),
) -> EPGSession:
    """Same as :func:`build_session` but for in-memory file content (e.g.
    a Streamlit ``st.file_uploader`` buffer) instead of filesystem paths.
    ``d0x_files`` is a list of (filename, bytes); filenames only need
    their ``.D0x`` suffix to be sorted correctly, they don't need to
    exist on disk. If ``ana_bytes`` is omitted the session has no
    ground-truth segments (real inference on unlabeled data)."""
    if not d0x_files:
        raise ValueError("No .D0x files provided")

    ordered = sorted(d0x_files, key=lambda name_bytes: sort_key(Path(name_bytes[0])))
    parsed = [parse_d0x_bytes(data, source_name=name) for name, data in ordered]
    rates = {p.header.sample_rate_hz for p in parsed}
    if len(rates) > 1:
        raise ValueError(f"Inconsistent sample rates across uploaded files: {rates}")

    samples = np.concatenate([p.samples for p in parsed])
    sample_rate_hz = rates.pop()
    recording_end_s = len(samples) / sample_rate_hz

    segments: list[LabeledSegment] = []
    if ana_bytes is not None:
        rows = parse_ana_bytes(ana_bytes)
        ana_segments, recording_end_s = rows_to_segments(rows, sentinel_codes=sentinel_codes)
        segments = _to_labeled_segments(ana_segments, sample_rate_hz)

    return EPGSession(
        insect_id=insect_id,
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        source_files=[Path(name) for name, _ in ordered],
        segments=segments,
        recording_end_s=recording_end_s,
    )
