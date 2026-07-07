"""
epg_core.py
Núcleo de parsing, extração de features e classificação heurística de
sinais de EPG (Electrical Penetration Graph) em arquivos binários .DXX
(formato: cabeçalho ASCII + corpo de floats32 little-endian).

Classificação baseada nos critérios de frequência/amplitude descritos em:
Bonani, J.P. et al. (2010) "Characterization of electrical penetration
graphs of the Asian citrus psyllid, Diaphorina citri, in sweet orange
seedlings." Entomologia Experimentalis et Applicata, 134: 35-49.

IMPORTANTE: esta classificação é uma heurística automática de primeira
passagem (frequência dominante + amplitude pico-a-pico), não substitui
a anotação visual especializada feita em softwares como Stylet+a/PROBE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Informações de referência da literatura (Bonani et al. 2010, Tabela 1 e 2)
# ---------------------------------------------------------------------------

WAVEFORM_INFO: Dict[str, dict] = {
    "Np": {
        "label": "Np — Não-sondagem",
        "color": "#B0BEC5",
        "desc": "Inseto fora de sondagem ativa (caminhando, parado, antenando).",
        "lit_mean_event_min": 4.2,
        "lit_freq_range": "—",
    },
    "C": {
        "label": "C — Via intercelular",
        "color": "#42A5F5",
        "desc": "Trajeto intercelular do estilete / secreção de bainha salivar.",
        "lit_mean_event_min": 10.0,
        "lit_freq_range": "11.5–19.0 Hz",
    },
    "D": {
        "label": "D — Contato com floema",
        "color": "#AB47BC",
        "desc": "Primeiro contato dos estiletes com elementos do floema.",
        "lit_mean_event_min": 0.77,  # ~46s
        "lit_freq_range": "1.0–3.5 Hz",
    },
    "E1": {
        "label": "E1 — Salivação no floema",
        "color": "#FFA726",
        "desc": "Salivação nos tubos crivados do floema.",
        "lit_mean_event_min": 1.65,
        "lit_freq_range": "5.0–7.5 Hz",
    },
    "E2": {
        "label": "E2 — Ingestão de floema",
        "color": "#66BB6A",
        "desc": "Ingestão passiva de seiva do floema.",
        "lit_mean_event_min": 150.2,
        "lit_freq_range": "3.0–9.0 Hz",
    },
    "G": {
        "label": "G — Ingestão de xilema",
        "color": "#EF5350",
        "desc": "Ingestão ativa de seiva do xilema.",
        "lit_mean_event_min": 23.7,
        "lit_freq_range": "5.0–8.0 Hz",
    },
    "Indeterminado": {
        "label": "Indeterminado",
        "color": "#E0E0E0",
        "desc": "Não classificado com confiança pelos critérios atuais.",
        "lit_mean_event_min": None,
        "lit_freq_range": "—",
    },
}

DEFAULT_THRESHOLDS = {
    "np_amp_max": 0.05,
    "c_freq": (11.5, 19.5),
    "d_freq": (1.0, 3.5),
    "d_amp_max": 0.20,
    "e1_freq": (5.0, 7.5),
    "e1_mean_max": -0.05,
    "e2_freq": (3.0, 9.0),
    "e2_amp_min": 0.20,
    "g_freq": (5.0, 8.0),
    "g_amp_min": 0.50,
}


# ---------------------------------------------------------------------------
# Parsing de arquivos .DXX
# ---------------------------------------------------------------------------

@dataclass
class ParsedFile:
    filename: str
    header_text: str
    date: Optional[str]
    time: Optional[str]
    rec_time: Optional[str]
    sample_rate: float
    values: np.ndarray
    n_samples: int


def parse_epg_bytes(file_bytes: bytes, filename: str) -> ParsedFile:
    """Faz o parsing de um único arquivo .DXX (cabeçalho ASCII + floats32)."""
    marker_candidates = [b"ok\r\n\r\n", b"ok\n\n", b"ok\r\n"]
    header_end = -1
    for marker in marker_candidates:
        idx = file_bytes.find(marker)
        if idx != -1:
            header_end = idx + len(marker)
            break
    if header_end == -1:
        # fallback: assume primeiros 128 bytes como header se marcador não encontrado
        header_end = min(128, len(file_bytes))

    header_text = file_bytes[:header_end].decode("ascii", errors="replace").strip()
    body = file_bytes[header_end:]

    n_floats = len(body) // 4
    values = np.frombuffer(body[: 4 * n_floats], dtype="<f4").astype(np.float64)

    date_match = re.search(r"EPG:\s*([\d\-/]+)\s+([\d:]+)", header_text)
    freq_match = re.search(r"smpl\.frq\s*=\s*([\d.,]+)\s*Hz", header_text, re.IGNORECASE)
    rectime_match = re.search(r"rec\.time\s*=\s*([\d.,]+)", header_text, re.IGNORECASE)

    sample_rate = 100.0
    if freq_match:
        raw = freq_match.group(1)
        # formato brasileiro usa vírgula como separador decimal (ex: 100,000 = 100.000 Hz)
        raw = raw.replace(",", ".")
        try:
            sample_rate = float(raw)
        except ValueError:
            pass

    return ParsedFile(
        filename=filename,
        header_text=header_text,
        date=date_match.group(1) if date_match else None,
        time=date_match.group(2) if date_match else None,
        rec_time=rectime_match.group(1) if rectime_match else None,
        sample_rate=sample_rate,
        values=values,
        n_samples=n_floats,
    )


def natural_segment_number(filename: str) -> int:
    """Extrai o número do segmento a partir da extensão .D01, .D02, ... .D99"""
    m = re.search(r"\.D0*?(\d+)$", filename, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def extract_channel_key(filename: str) -> str:
    """
    Extrai a chave do canal/inseto a partir do nome do arquivo, removendo a
    extensão .DXX. Arquivos com a mesma chave são tratados como segmentos
    consecutivos da mesma gravação (mesmo canal/inseto) e concatenados.
    Ex: 'PsilideoMudaT1-3-ch6.D01' -> 'PsilideoMudaT1-3-ch6'
    """
    return re.sub(r"\.D0*?\d+$", "", filename, flags=re.IGNORECASE)


@dataclass
class StitchedChannel:
    channel_key: str
    sample_rate: float
    values: np.ndarray
    total_seconds: float
    segments: List[dict]  # [{filename, start_s, end_s, n_samples}]
    start_date: Optional[str]
    start_time: Optional[str]


def group_and_stitch(parsed_files: List[ParsedFile]) -> List[StitchedChannel]:
    """Agrupa arquivos pelo canal/inseto e concatena os segmentos em ordem."""
    groups: Dict[str, List[ParsedFile]] = {}
    for pf in parsed_files:
        key = extract_channel_key(pf.filename)
        groups.setdefault(key, []).append(pf)

    stitched_channels = []
    for key, files in groups.items():
        files_sorted = sorted(files, key=lambda p: natural_segment_number(p.filename))
        sample_rate = files_sorted[0].sample_rate
        all_values = np.concatenate([f.values for f in files_sorted])
        segments = []
        offset = 0
        for f in files_sorted:
            segments.append(
                {
                    "filename": f.filename,
                    "start_s": offset / sample_rate,
                    "end_s": (offset + f.n_samples) / sample_rate,
                    "n_samples": f.n_samples,
                }
            )
            offset += f.n_samples

        stitched_channels.append(
            StitchedChannel(
                channel_key=key,
                sample_rate=sample_rate,
                values=all_values,
                total_seconds=len(all_values) / sample_rate,
                segments=segments,
                start_date=files_sorted[0].date,
                start_time=files_sorted[0].time,
            )
        )

    return stitched_channels


# ---------------------------------------------------------------------------
# Extração de features por janela deslizante
# ---------------------------------------------------------------------------

def extract_features(
    signal: np.ndarray,
    fs: float,
    win_seconds: float = 5.0,
    overlap: float = 0.5,
    freq_band: tuple = (0.5, 25.0),
) -> pd.DataFrame:
    """
    Calcula, para cada janela deslizante do sinal:
      - tensão média (mean_v)
      - desvio padrão (std_v)
      - amplitude pico-a-pico robusta (percentil 95 - percentil 5) (amp_p2p)
      - frequência dominante via FFT, restrita a uma faixa biologicamente
        plausível para EPG (dom_freq_hz)
    """
    win_len = max(2, int(round(win_seconds * fs)))
    step = max(1, int(round(win_len * (1 - overlap))))
    n_win = max(0, (len(signal) - win_len) // step + 1)

    if n_win == 0:
        return pd.DataFrame(columns=["t_center_s", "mean_v", "std_v", "amp_p2p", "dom_freq_hz"])

    t_centers = np.empty(n_win)
    means = np.empty(n_win)
    stds = np.empty(n_win)
    amps = np.empty(n_win)
    domfreqs = np.empty(n_win)

    freqs_full = np.fft.rfftfreq(win_len, d=1.0 / fs)
    band_mask = (freqs_full >= freq_band[0]) & (freqs_full <= freq_band[1])
    has_band = band_mask.any()

    for i in range(n_win):
        start = i * step
        seg = signal[start : start + win_len]
        means[i] = seg.mean()
        stds[i] = seg.std()
        amps[i] = np.percentile(seg, 95) - np.percentile(seg, 5)
        if has_band:
            mag = np.abs(np.fft.rfft(seg - seg.mean()))
            domfreqs[i] = freqs_full[band_mask][np.argmax(mag[band_mask])]
        else:
            domfreqs[i] = 0.0
        t_centers[i] = (start + win_len / 2.0) / fs

    return pd.DataFrame(
        {
            "t_center_s": t_centers,
            "mean_v": means,
            "std_v": stds,
            "amp_p2p": amps,
            "dom_freq_hz": domfreqs,
        }
    )


# ---------------------------------------------------------------------------
# Classificação heurística
# ---------------------------------------------------------------------------

def classify_window(freq: float, amp: float, mean: float, th: dict) -> str:
    if amp < th["np_amp_max"]:
        return "Np"
    if th["c_freq"][0] <= freq <= th["c_freq"][1]:
        return "C"
    if th["d_freq"][0] <= freq <= th["d_freq"][1] and amp < th["d_amp_max"]:
        return "D"
    if th["e1_freq"][0] <= freq <= th["e1_freq"][1] and mean < th["e1_mean_max"]:
        return "E1"
    if th["e2_freq"][0] <= freq <= th["e2_freq"][1] and amp >= th["e2_amp_min"]:
        return "E2"
    if th["g_freq"][0] <= freq <= th["g_freq"][1] and amp >= th["g_amp_min"]:
        return "G"
    return "Indeterminado"


def classify_features(feat_df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    if feat_df.empty:
        feat_df = feat_df.copy()
        feat_df["waveform"] = []
        return feat_df
    freqs = feat_df["dom_freq_hz"].to_numpy()
    amps = feat_df["amp_p2p"].to_numpy()
    means = feat_df["mean_v"].to_numpy()
    labels = [classify_window(f, a, m, thresholds) for f, a, m in zip(freqs, amps, means)]
    out = feat_df.copy()
    out["waveform"] = labels
    return out


def compute_runs(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa janelas classificadas consecutivamente iguais em 'eventos' (runs)."""
    if feat_df.empty:
        return pd.DataFrame(columns=["waveform", "start_s", "end_s", "duration_s"])

    runs = []
    cur = feat_df.iloc[0]["waveform"]
    start = feat_df.iloc[0]["t_center_s"]
    for i in range(1, len(feat_df)):
        row = feat_df.iloc[i]
        if row["waveform"] != cur:
            end = feat_df.iloc[i - 1]["t_center_s"]
            runs.append({"waveform": cur, "start_s": start, "end_s": end, "duration_s": end - start})
            cur = row["waveform"]
            start = row["t_center_s"]
    end = feat_df.iloc[-1]["t_center_s"]
    runs.append({"waveform": cur, "start_s": start, "end_s": end, "duration_s": end - start})
    return pd.DataFrame(runs)


def summarize_runs(runs_df: pd.DataFrame, total_seconds: float) -> pd.DataFrame:
    """
    Calcula métricas por waveform análogas às usadas na literatura EPG
    (Backus et al. 2007): NWEI (nº de eventos), WDI (duração total, min),
    WDE (duração média por evento, min), % do tempo total.
    """
    if runs_df.empty:
        return pd.DataFrame()

    rows = []
    for wf, group in runs_df.groupby("waveform"):
        total_min = group["duration_s"].sum() / 60.0
        n_events = len(group)
        mean_event_min = total_min / n_events if n_events else 0.0
        rows.append(
            {
                "waveform": wf,
                "n_eventos (NWEI)": n_events,
                "duracao_total_min (WDI)": round(total_min, 2),
                "duracao_media_evento_min (WDE)": round(mean_event_min, 2),
                "%_do_tempo_total": round(100 * (total_min * 60) / total_seconds, 1) if total_seconds else 0,
            }
        )
    summary = pd.DataFrame(rows).sort_values("duracao_total_min (WDI)", ascending=False).reset_index(drop=True)
    return summary
