"""
epg_ana.py
Parsing e alinhamento de arquivos de anotação .ANA (Stylet+/Stylet+a) com o
sinal bruto .DXX, para construir datasets rotulados de treino.

Formato .ANA (Stylet+):
  - Arquivo texto tab-delimitado, codificado em UTF-16 com BOM (UTF-16LE).
  - 3 colunas por linha: código do waveform | tempo de início (s) | voltagem
    de saída (mV = ganho x tensão de entrada).
  - Cada linha é um EVENTO (um clique manual no Stylet+), não uma amostra:
    o código vale do seu tempo de início até o início do próximo evento.
  - Os códigos numéricos dos botões são DEFINIDOS PELO USUÁRIO no Stylet+
    (não há mapeamento universal) — por isso o parser aceita um dicionário
    de mapeamento código->waveform explícito.

Referências: manual Stylet+ (EPG Systems, Wageningen); Dinh et al., DiscoEPG,
bioRxiv 2024.12.05.627099 (descrição do formato .ANA).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Mapeamento de código->waveform confirmado pelo usuário para esta configuração
# de botões do Stylet+ (laboratório ESALQ, D. citri): note que o código 6 não é
# usado nesta configuração (pulado na numeração dos botões).
DEFAULT_CODE_MAP: Dict[int, str] = {
    1: "Np",
    2: "C",
    3: "D",   # também referido como E1e (fase extracelular / contato inicial com floema)
    4: "E1",
    5: "E2",
    7: "G",
}

# Códigos que são marcadores de controle do Stylet+ (ex: fim de arquivo/gravação),
# não waveforms reais -- devem ser excluídos da rotulagem de treino, mas seu
# start_s ainda delimita corretamente o fim do último evento real.
TERMINATOR_CODES = {99}


@dataclass
class AnaEvent:
    code: int
    start_s: float
    voltage_mv: float


def parse_ana_bytes(file_bytes: bytes) -> List[AnaEvent]:
    """
    Faz o parsing de um arquivo .ANA bruto (bytes) em uma lista de eventos.
    Tenta UTF-16 (com/sem BOM) primeiro (padrão Stylet+), com fallback para
    UTF-8/latin-1 caso o arquivo tenha sido re-salvo em outra codificação.
    """
    text = None
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            # heurística: um .ANA decodificado corretamente não deve conter
            # excesso de caracteres de controle/nulos
            if text.count("\x00") < len(text) * 0.05:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise ValueError("Não foi possível decodificar o arquivo .ANA com nenhuma codificação testada.")

    events: List[AnaEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()  # fallback para espaços em vez de tab
        if len(parts) < 2:
            continue
        try:
            code = int(float(parts[0].replace(",", ".")))
            start_s = float(parts[1].replace(",", "."))
            voltage_mv = float(parts[2].replace(",", ".")) if len(parts) > 2 else np.nan
        except ValueError:
            continue  # ignora linhas de cabeçalho ou mal formatadas
        events.append(AnaEvent(code=code, start_s=start_s, voltage_mv=voltage_mv))

    events.sort(key=lambda e: e.start_s)
    return events


def events_to_dataframe(events: List[AnaEvent], code_map: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """
    Converte a lista de eventos em um DataFrame com start_s, end_s, waveform.
    Códigos em TERMINATOR_CODES (ex: marcador de fim de arquivo) são usados
    para delimitar corretamente o fim do último evento real, mas são
    excluídos do resultado final (não geram uma linha/classe própria).
    """
    code_map = code_map or DEFAULT_CODE_MAP
    rows = []
    for i, ev in enumerate(events):
        end_s = events[i + 1].start_s if i + 1 < len(events) else np.inf
        if ev.code in TERMINATOR_CODES:
            continue  # marcador de controle, não é um waveform real
        waveform = code_map.get(ev.code, f"Codigo_{ev.code}")
        rows.append({"code": ev.code, "waveform": waveform, "start_s": ev.start_s, "end_s": end_s})
    return pd.DataFrame(rows)


def label_windows(feat_df: pd.DataFrame, events_df: pd.DataFrame, total_seconds: float) -> pd.DataFrame:
    """
    Atribui a cada janela de features (identificada por t_center_s) o
    waveform de anotação ativo naquele instante, expandindo os eventos
    (rótulo por evento) em um rótulo contínuo.

    Janelas cujo t_center_s cai depois do último evento anotado (ex: parte
    final não anotada) recebem NaN e devem ser descartadas do treino.
    """
    if events_df.empty:
        out = feat_df.copy()
        out["waveform"] = np.nan
        return out

    ev = events_df.sort_values("start_s").reset_index(drop=True)
    # substitui o 'end_s' do último evento pela duração total conhecida,
    # em vez de manter infinito, para fins de diagnóstico
    ev.loc[ev.index[-1], "end_s"] = min(ev["end_s"].iloc[-1], total_seconds) \
        if np.isfinite(ev["end_s"].iloc[-1]) else total_seconds

    starts = ev["start_s"].to_numpy()
    out = feat_df.copy()
    # para cada t_center, encontra o último evento com start_s <= t_center
    idx = np.searchsorted(starts, out["t_center_s"].to_numpy(), side="right") - 1
    labels = np.full(len(out), np.nan, dtype=object)
    valid = idx >= 0
    labels[valid] = ev["waveform"].to_numpy()[idx[valid]]
    out["waveform"] = labels
    return out


def build_labeled_dataset(
    feat_df: pd.DataFrame,
    ana_bytes: bytes,
    total_seconds: float,
    code_map: Optional[Dict[int, str]] = None,
    recording_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pipeline completo: parseia o .ANA, expande eventos e rotula as janelas
    de features já extraídas (ex: via extract_features_advanced).
    Adiciona a coluna 'recording_id' se fornecida, para permitir split
    agrupado por gravação/inseto no treino.
    """
    events = parse_ana_bytes(ana_bytes)
    events_df = events_to_dataframe(events, code_map=code_map)
    labeled = label_windows(feat_df, events_df, total_seconds)
    if recording_id is not None:
        labeled["recording_id"] = recording_id
    return labeled
