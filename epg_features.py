"""
epg_features.py
Extração de features "estilo DiscoEPG" para sinais de EPG, como substituto
robusto da extração baseada apenas em FFT + amplitude absoluta.

Combina:
  1. Pré-processamento: remoção de deriva de linha de base (baseline drift) +
     padronização robusta por gravação (mediana/IQR) -> torna as features
     independentes do ganho do amplificador e da posição do eletrodo.
  2. 13 features estatísticas/de complexidade por janela (conjunto DiscoEPG:
     Dinh et al., bioRxiv 2024.12.05.627099): média, RMS, desvio-padrão,
     variância, assimetria, quantis 0.05/0.25/0.5/0.75/0.95, taxa de cruzamento
     por zero, entropia de Shannon e entropia de permutação.
  3. Features espectrais informativas: centroide espectral, entropia espectral
     e razões de potência nas bandas biológicas do EPG (1-4 / 5-9 / 11-19 Hz).
  4. Features de wavelet (DWT db4): energia relativa e entropia por nível.
  5. Colunas de compatibilidade (mean_v, std_v, amp_p2p, dom_freq_hz) para que
     o classificador heurístico legado continue funcionando e permita o A/B.

Dependências: numpy, pandas, scipy, PyWavelets (pywt).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import pywt
from scipy import signal as sp_signal
from scipy.stats import skew


# ---------------------------------------------------------------------------
# 1. Pré-processamento
# ---------------------------------------------------------------------------

def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    baseline_window_s: float = 60.0,
    remove_baseline: bool = True,
    robust_standardize: bool = True,
) -> dict:
    """
    Remove deriva de linha de base e padroniza o sinal de forma robusta.

    Retorna um dicionário com:
      - 'signal_proc' : sinal processado (usado para extração de features)
      - 'baseline'    : linha de base estimada (mediana móvel)
      - 'np_center'   : nível de referência de não-sondagem (mediana global do
                        sinal processado) — usado para codificar deslocamentos
                        relativos de nível de voltagem (E1/E2 intracelular).
      - 'np_scale'    : escala robusta (IQR) do sinal processado.

    A padronização robusta (x - mediana) / IQR substitui thresholds em Volts
    absolutos por valores comparáveis entre gravações com ganhos diferentes.
    """
    x = np.asarray(signal, dtype=np.float64)

    baseline = np.zeros_like(x)
    if remove_baseline and len(x) > 3:
        # mediana móvel via filtro de mediana (kernel ímpar). Para janelas
        # longas isso pode ser custoso; usamos uma versão decimada + interp.
        win = max(3, int(round(baseline_window_s * fs)))
        if win % 2 == 0:
            win += 1
        # Estima a baseline em uma versão subamostrada para eficiência e
        # interpola de volta ao comprimento original.
        decim = max(1, win // 20)
        xs = x[::decim]
        kwin = max(3, win // decim)
        if kwin % 2 == 0:
            kwin += 1
        kwin = min(kwin, len(xs) - (1 - len(xs) % 2))  # não exceder tamanho
        if kwin >= 3:
            bs = sp_signal.medfilt(xs, kernel_size=kwin)
        else:
            bs = np.full_like(xs, np.median(xs))
        # interpola de volta
        idx_full = np.arange(len(x))
        idx_dec = np.arange(0, len(x), decim)[: len(bs)]
        baseline = np.interp(idx_full, idx_dec, bs)
        x_proc = x - baseline
    else:
        x_proc = x.copy()

    median = np.median(x_proc)
    q75, q25 = np.percentile(x_proc, [75, 25])
    iqr = (q75 - q25) or 1.0

    if robust_standardize:
        signal_proc = (x_proc - median) / iqr
        np_center = 0.0
        np_scale = 1.0
    else:
        signal_proc = x_proc
        np_center = median
        np_scale = iqr

    return {
        "signal_proc": signal_proc,
        "baseline": baseline,
        "np_center": np_center,
        "np_scale": np_scale,
    }


# ---------------------------------------------------------------------------
# 2. Features de complexidade
# ---------------------------------------------------------------------------

def _shannon_entropy(seg: np.ndarray, n_bins: int = 16) -> float:
    """Entropia de Shannon da distribuição de amplitudes (histograma)."""
    hist, _ = np.histogram(seg, bins=n_bins, density=False)
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p[p > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _permutation_entropy(seg: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Entropia de permutação (Bandt & Pompe). Mede a complexidade ordinal da
    série temporal — boa para separar waveforms 'agitados' (C) dos suaves (G/E2).
    Normalizada para [0, 1].
    """
    n = len(seg)
    m = n - delay * (order - 1)
    if m < 1:
        return 0.0
    # matriz de sub-janelas embutidas (m x order), vetorizada
    idx = np.arange(order) * delay
    embed = seg[np.arange(m)[:, None] + idx[None, :]]
    # padrão ordinal = argsort de cada linha; converte para código único
    perms = np.argsort(embed, axis=1, kind="quicksort")
    # codifica cada permutação como inteiro via base-fatorial simples (mixed radix)
    weights = (order ** np.arange(order)).astype(np.int64)
    codes = perms.astype(np.int64) @ weights
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    pe = -np.sum(p * np.log2(p))
    max_pe = np.log2(math.factorial(order))
    return float(pe / max_pe) if max_pe > 0 else 0.0


def _zero_crossing_rate(seg: np.ndarray) -> float:
    """Taxa de cruzamento por zero (em torno da média do segmento)."""
    centered = seg - seg.mean()
    signs = np.sign(centered)
    signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs)) > 0) / len(seg))


# ---------------------------------------------------------------------------
# 3. Features espectrais
# ---------------------------------------------------------------------------

def _spectral_features(seg: np.ndarray, fs: float, freq_band=(0.5, 25.0)) -> dict:
    """Frequência dominante, centroide espectral, entropia espectral e razões
    de potência nas bandas biológicas do EPG."""
    n = len(seg)
    seg_dm = seg - seg.mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(seg_dm)) ** 2

    band = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if not band.any() or psd[band].sum() == 0:
        return {
            "dom_freq_hz": 0.0,
            "spec_centroid_hz": 0.0,
            "spec_entropy": 0.0,
            "bandratio_1_4": 0.0,
            "bandratio_5_9": 0.0,
            "bandratio_11_19": 0.0,
        }

    f_b = freqs[band]
    p_b = psd[band]
    p_norm = p_b / p_b.sum()

    dom_freq = f_b[np.argmax(p_b)]
    centroid = float(np.sum(f_b * p_norm))
    spec_entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-12)))

    def band_ratio(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        denom = psd[band].sum()
        return float(psd[m].sum() / denom) if denom > 0 else 0.0

    return {
        "dom_freq_hz": float(dom_freq),
        "spec_centroid_hz": centroid,
        "spec_entropy": spec_entropy,
        "bandratio_1_4": band_ratio(1.0, 4.0),
        "bandratio_5_9": band_ratio(5.0, 9.0),
        "bandratio_11_19": band_ratio(11.5, 19.0),
    }


# ---------------------------------------------------------------------------
# 4. Features de wavelet
# ---------------------------------------------------------------------------

def _wavelet_features(seg: np.ndarray, wavelet: str = "db4", level: int = 4) -> dict:
    """Energia relativa e entropia por nível de decomposição wavelet (DWT)."""
    max_level = pywt.dwt_max_level(len(seg), pywt.Wavelet(wavelet).dec_len)
    use_level = min(level, max_level) if max_level >= 1 else 1
    try:
        coeffs = pywt.wavedec(seg, wavelet, level=use_level)
    except ValueError:
        coeffs = [seg]

    energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
    total = energies.sum() or 1.0
    rel_energy = energies / total

    feats = {}
    # garante sempre 'level+1' colunas (aprox + detalhes) preenchendo com 0
    for i in range(level + 1):
        feats[f"wav_energy_{i}"] = float(rel_energy[i]) if i < len(rel_energy) else 0.0
    # entropia da distribuição de energia entre níveis
    p = rel_energy[rel_energy > 0]
    feats["wav_entropy"] = float(-np.sum(p * np.log2(p))) if len(p) else 0.0
    return feats


# ---------------------------------------------------------------------------
# 5. Extração principal (drop-in replacement para extract_features)
# ---------------------------------------------------------------------------

def extract_features_advanced(
    signal: np.ndarray,
    fs: float,
    win_seconds: float = 5.0,
    overlap: float = 0.5,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    remove_baseline: bool = True,
    robust_standardize: bool = True,
    perm_entropy_order: int = 3,
) -> pd.DataFrame:
    """
    Extrai o conjunto completo de features por janela deslizante.

    Mantém compatibilidade com o pipeline heurístico legado ao incluir as
    colunas: t_center_s, mean_v, std_v, amp_p2p, dom_freq_hz.
    'mean_v' e 'amp_p2p' são calculados no sinal PROCESSADO (padronizado),
    de modo que os thresholds do classificador heurístico passam a operar em
    unidades robustas (comparáveis entre gravações), não mais em Volts crus.
    """
    prep = preprocess_signal(
        signal, fs, remove_baseline=remove_baseline, robust_standardize=robust_standardize
    )
    x = prep["signal_proc"]

    win_len = max(4, int(round(win_seconds * fs)))
    step = max(1, int(round(win_len * (1 - overlap))))
    n_win = max(0, (len(x) - win_len) // step + 1)
    if n_win == 0:
        return pd.DataFrame()

    rows = []
    for i in range(n_win):
        start = i * step
        seg = x[start : start + win_len]

        mean_v = float(seg.mean())
        rms = float(np.sqrt(np.mean(seg ** 2)))
        std_v = float(seg.std())
        var_v = float(seg.var())
        skew_v = float(skew(seg)) if seg.std() > 1e-12 else 0.0
        q05, q25, q50, q75, q95 = np.percentile(seg, [5, 25, 50, 75, 95])
        amp_p2p = float(q95 - q05)
        zcr = _zero_crossing_rate(seg)
        sh_ent = _shannon_entropy(seg)
        perm_ent = _permutation_entropy(seg, order=perm_entropy_order)

        spec = _spectral_features(seg, fs)
        wav = _wavelet_features(seg, wavelet=wavelet, level=wavelet_level)

        row = {
            "t_center_s": (start + win_len / 2.0) / fs,
            # --- compat. com heurística legada ---
            "mean_v": mean_v,
            "std_v": std_v,
            "amp_p2p": amp_p2p,
            "dom_freq_hz": spec["dom_freq_hz"],
            # --- 13 features estatísticas/complexidade (DiscoEPG) ---
            "rms": rms,
            "variance": var_v,
            "skewness": skew_v,
            "q05": float(q05),
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "q95": float(q95),
            "zero_cross_rate": zcr,
            "shannon_entropy": sh_ent,
            "perm_entropy": perm_ent,
            # --- espectrais ---
            "spec_centroid_hz": spec["spec_centroid_hz"],
            "spec_entropy": spec["spec_entropy"],
            "bandratio_1_4": spec["bandratio_1_4"],
            "bandratio_5_9": spec["bandratio_5_9"],
            "bandratio_11_19": spec["bandratio_11_19"],
        }
        row.update(wav)
        rows.append(row)

    return pd.DataFrame(rows)


# lista de colunas de features (para alimentar um modelo de ML)
def feature_columns(df: pd.DataFrame) -> list:
    """Retorna as colunas numéricas de feature (exclui tempo e rótulo)."""
    exclude = {"t_center_s", "waveform"}
    return [c for c in df.columns if c not in exclude]
