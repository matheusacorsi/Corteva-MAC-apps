"""Parser for EPG Systems / Stylet+ raw ``.D0x`` trace files.

File layout (verified against real Stylet+ output, see project docs):

    <ascii header>ok\r\n\r\n<float32 LE samples ...>

Header example::

    EPG: 13-10-2025 10:16:30/rec.time= 8,10/smpl.frq= 100,000Hz\r\nok\r\n\r\n

``rec.time`` is the total session duration in decimal hours and
``smpl.frq`` is the sampling rate in Hz; both use a comma as the decimal
separator. The header length is not fixed-width -- it is located by
searching for the ``ok\r\n\r\n`` terminator rather than assuming a byte
offset.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_HEADER_TERMINATOR = b"ok\r\n\r\n"

_HEADER_RE = re.compile(
    r"EPG:\s*(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})"
    r"/rec\.time=\s*(?P<rec_time>[\d,.]+)"
    r"/smpl\.frq=\s*(?P<freq>[\d,.]+)Hz",
    re.ASCII,
)


def _to_float(raw: str) -> float:
    """European-style decimal comma -> float."""
    return float(raw.replace(",", "."))


@dataclass(frozen=True)
class D0xHeader:
    recorded_at: datetime
    rec_time_hours: float
    sample_rate_hz: float
    header_end_offset: int


@dataclass
class D0xFile:
    path: Path | None
    header: D0xHeader
    samples: "np.ndarray"  # float32, volts

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.header.sample_rate_hz


def parse_d0x_header(data: bytes) -> D0xHeader:
    term_idx = data.find(_HEADER_TERMINATOR)
    if term_idx == -1:
        raise ValueError("Could not locate 'ok\\r\\n\\r\\n' header terminator in .D0x file")
    header_end = term_idx + len(_HEADER_TERMINATOR)

    header_text = data[:term_idx].decode("ascii", errors="strict")
    match = _HEADER_RE.search(header_text)
    if match is None:
        raise ValueError(f"Unrecognized .D0x header format: {header_text!r}")

    recorded_at = datetime.strptime(
        f"{match['date']} {match['time']}", "%d-%m-%Y %H:%M:%S"
    )
    return D0xHeader(
        recorded_at=recorded_at,
        rec_time_hours=_to_float(match["rec_time"]),
        sample_rate_hz=_to_float(match["freq"]),
        header_end_offset=header_end,
    )


def parse_d0x_bytes(data: bytes, source_name: str = "<bytes>") -> D0xFile:
    """Parse .D0x content already in memory (e.g. a Streamlit upload buffer)."""
    import numpy as np

    header = parse_d0x_header(data)

    body = data[header.header_end_offset :]
    n_samples, remainder = divmod(len(body), 4)
    if remainder != 0:
        raise ValueError(
            f"{source_name}: body size {len(body)} bytes is not a multiple of 4 "
            "(expected float32 samples)"
        )
    samples = np.frombuffer(body, dtype="<f4", count=n_samples)
    return D0xFile(path=None, header=header, samples=samples)


def parse_d0x_file(path: str | Path) -> D0xFile:
    path = Path(path)
    d0x = parse_d0x_bytes(path.read_bytes(), source_name=str(path))
    d0x.path = path
    return d0x


_SUFFIX_RE = re.compile(r"\.D(\d+)$", re.IGNORECASE)


def is_d0x_filename(name: str) -> bool:
    return _SUFFIX_RE.search(name) is not None


def sort_key(path: Path) -> int:
    match = _SUFFIX_RE.search(path.name)
    if match is None:
        raise ValueError(f"{path}: filename does not end in .D<digits> (e.g. .D01)")
    return int(match.group(1))


def find_d0x_series(any_file: str | Path) -> list[Path]:
    """Given one .D0x file of a recording, return all siblings (.D01, .D02, ...)
    for the same session, sorted by numeric suffix."""
    any_file = Path(any_file)
    stem_match = _SUFFIX_RE.search(any_file.name)
    if stem_match is None:
        raise ValueError(f"{any_file}: filename does not end in .D<digits>")
    prefix = any_file.name[: stem_match.start()]
    siblings = [
        p
        for p in any_file.parent.iterdir()
        if p.name.startswith(prefix) and _SUFFIX_RE.search(p.name)
    ]
    return sorted(siblings, key=sort_key)
