"""Parser for Stylet+ ``.ANA`` annotation files.

File layout (verified against a real Stylet+ export):

    UTF-16LE text (with BOM), CRLF line endings, tab-delimited:

        <waveform_code:int>\t<start_time_seconds:comma-decimal>\t<marker:int>\r\n

``start_time_seconds`` is elapsed time since the start of the whole
(possibly multi-file) recording session, not per-file. Each row marks the
*start* of a waveform event; its end is implicit -- the start time of the
next row. The file always ends with a sentinel row (code 99 in every
sample seen so far) that is not a real waveform, just Stylet+'s
"recording end" marker; its timestamp closes out the final real segment.
The marker column is the raw voltage in millivolts (voltage * 1000,
rounded) at the transition sample -- redundant with the .D0x trace and
not needed for classification, but kept for completeness/QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SENTINEL_CODES = frozenset({99})


@dataclass(frozen=True)
class AnaRow:
    code: int
    time_s: float
    marker: int


@dataclass(frozen=True)
class AnaSegment:
    code: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "."))


def parse_ana_bytes(data: bytes) -> list[AnaRow]:
    """Parse .ANA content already in memory (e.g. a Streamlit upload buffer)."""
    text = data.decode("utf-16")
    rows: list[AnaRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        code_s, time_s, marker_s = line.split("\t")
        rows.append(AnaRow(code=int(code_s), time_s=_to_float(time_s), marker=int(marker_s)))
    return rows


def parse_ana_rows(path: str | Path) -> list[AnaRow]:
    return parse_ana_bytes(Path(path).read_bytes())


def rows_to_segments(
    rows: list[AnaRow],
    sentinel_codes: frozenset[int] = DEFAULT_SENTINEL_CODES,
) -> tuple[list[AnaSegment], float]:
    """Convert start-time rows into (code, start, end) segments.

    Returns the segment list plus the overall recording end time (the
    timestamp of the final row, expected to be a sentinel).
    """
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows (one event + sentinel) to derive segments")

    rows_sorted = sorted(rows, key=lambda r: r.time_s)
    recording_end = rows_sorted[-1].time_s

    segments: list[AnaSegment] = []
    for current, nxt in zip(rows_sorted, rows_sorted[1:]):
        if current.code in sentinel_codes:
            continue
        segments.append(AnaSegment(code=current.code, start_s=current.time_s, end_s=nxt.time_s))
    return segments, recording_end


def parse_ana_file(
    path: str | Path,
    sentinel_codes: frozenset[int] = DEFAULT_SENTINEL_CODES,
) -> tuple[list[AnaSegment], float]:
    rows = parse_ana_rows(path)
    return rows_to_segments(rows, sentinel_codes=sentinel_codes)
