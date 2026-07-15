from __future__ import annotations

import struct
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
REAL_DATA_DIR = REPO_ROOT / "Arquivo Ondas EPG Diaphorina citri" / "arquivosDataAcquisition"
REAL_D01 = REAL_DATA_DIR / "PsilideoMudaT1-3-ch6.D01"
REAL_ANA = REAL_DATA_DIR / "DataANA" / "PsilideoMudaT1-3-ch6_.ANA"

requires_real_data = pytest.mark.skipif(
    not REAL_D01.exists() or not REAL_ANA.exists(),
    reason="Real example .D0x/.ANA files not present in this checkout",
)


def make_d0x_bytes(
    samples: list[float],
    recorded_at: str = "13-10-2025 10:16:30",
    rec_time_hours: str = "1,00",
    sample_rate_hz: str = "100,000",
) -> bytes:
    """Build a synthetic .D0x file body matching the verified Stylet+ layout."""
    date, time = recorded_at.split(" ")
    header = f"EPG: {date} {time}/rec.time= {rec_time_hours}/smpl.frq= {sample_rate_hz}Hz\r\nok\r\n\r\n"
    body = struct.pack(f"<{len(samples)}f", *samples)
    return header.encode("ascii") + body


def make_ana_bytes(rows: list[tuple[int, str, int]]) -> bytes:
    """Build a synthetic .ANA file: rows of (code, time_str_comma_decimal, marker)."""
    lines = [f"{code}\t{time_str}\t{marker}" for code, time_str, marker in rows]
    text = "\r\n".join(lines) + "\r\n"
    return text.encode("utf-16")


@pytest.fixture()
def tmp_d0x_pair(tmp_path):
    """Two consecutive .D0x files (D01, D02) with known sample values."""
    samples_01 = [0.01 * i for i in range(10)]
    samples_02 = [0.01 * i for i in range(10, 20)]
    p1 = tmp_path / "insectA.D01"
    p2 = tmp_path / "insectA.D02"
    p1.write_bytes(make_d0x_bytes(samples_01))
    p2.write_bytes(make_d0x_bytes(samples_02))
    return p1, p2, samples_01 + samples_02
