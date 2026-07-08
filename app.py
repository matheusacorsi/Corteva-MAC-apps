"""
EPG Classifier — Classificador automático de waveforms de EPG
(Electrical Penetration Graph) para Diaphorina citri e outros psilídeos.

Baseado nos critérios de frequência/amplitude de:
Bonani et al. (2010), Entomol. Exp. Appl. 134: 35-49.

Rode com: streamlit run app.py
"""

import io
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from epg_core import (
    DEFAULT_THRESHOLDS,
    DEFAULT_THRESHOLDS_STD,
    WAVEFORM_INFO,
    classify_features,
    compute_runs,
    extract_features,
    group_and_stitch,
    parse_epg_bytes,
    summarize_runs,
)
from epg_features import extract_features_advanced, feature_columns

st.set_page_config(
    page_title="EPG Classifier — Diaphorina citri",
    page_icon="🦟",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — upload e parâmetros
# ---------------------------------------------------------------------------

st.sidebar.title("🦟 EPG Classifier")
st.sidebar.caption("Classificação automática de waveforms de EPG a partir de critérios da literatura.")

uploaded_files = st.sidebar.file_uploader(
    "Envie os arquivos .DXX (ex: canal.D01, canal.D02, ...)",
    type=None,
    accept_multiple_files=True,
    help="Arquivos com o mesmo nome-base (antes de .D01, .D02...) são tratados "
    "como segmentos consecutivos do mesmo canal/inseto e concatenados automaticamente.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros de análise")

feature_method = st.sidebar.radio(
    "Método de extração de features",
    ["Avançado (estilo DiscoEPG)", "FFT + amplitude (legado)"],
    help="O modo avançado remove deriva de linha de base, padroniza o sinal "
    "de forma robusta (independente do ganho) e extrai 26 features "
    "(estatísticas, espectrais e de wavelet). O modo legado usa apenas "
    "frequência dominante (FFT) e amplitude absoluta em Volts.",
)
is_advanced = feature_method.startswith("Avançado")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Modelo treinado (opcional)")
uploaded_model = st.sidebar.file_uploader(
    "Envie um modelo treinado (.joblib)",
    type=["joblib"],
    help="Gerado por train_model.py a partir de arquivos .DXX + .ANA anotados. "
    "Se enviado, substitui a classificação heurística por predições do modelo.",
)
use_ml_model = False
ml_bundle = None
if uploaded_model is not None:
    try:
        ml_bundle = joblib.load(io.BytesIO(uploaded_model.getvalue()))
        use_ml_model = True
        st.sidebar.success(
            f"Modelo carregado ({len(ml_bundle['feature_columns'])} features, "
            f"classes: {', '.join(ml_bundle['label_encoder'].classes_)})"
        )
        if not is_advanced:
            st.sidebar.warning(
                "⚠️ O modelo foi treinado com o modo Avançado. Selecione "
                "'Avançado (estilo DiscoEPG)' acima para que as features "
                "sejam compatíveis."
            )
    except Exception as e:
        st.sidebar.error(f"Não foi possível carregar o modelo: {e}")

win_seconds = st.sidebar.slider("Tamanho da janela (s)", 1.0, 30.0, 5.0, 0.5)
overlap = st.sidebar.slider("Sobreposição entre janelas", 0.0, 0.9, 0.5, 0.05)

remove_baseline = True
if is_advanced:
    remove_baseline = st.sidebar.checkbox(
        "Remover deriva de linha de base", value=True,
        help="Subtrai uma mediana móvel lenta (~60s) para remover deriva do eletrodo.",
    )

st.sidebar.markdown("---")
_default_th = DEFAULT_THRESHOLDS_STD if is_advanced else DEFAULT_THRESHOLDS
_unit = "un. robustas (IQR)" if is_advanced else "V"
with st.sidebar.expander("⚙️ Thresholds de classificação (avançado)", expanded=not use_ml_model):
    if use_ml_model:
        st.caption(
            "🤖 Um modelo de ML está carregado e estes thresholds estão "
            "sendo **ignorados** — a classificação vem das predições do modelo."
        )
    st.caption(
        f"Faixas de frequência seguem Bonani et al. (2010). As amplitudes/nível "
        f"médio estão em **{_unit}**. "
        + (
            "No modo avançado o sinal é padronizado por IQR, tornando os "
            "thresholds comparáveis entre gravações com ganhos diferentes."
            if is_advanced
            else "No modo legado dependem do ganho do amplificador — calibre "
            "conforme seu equipamento."
        )
    )
    th = dict(_default_th)
    th["np_amp_max"] = st.number_input(
        f"Np: amplitude máxima ({_unit})", value=float(_default_th["np_amp_max"]), step=0.01, key="np_amp"
    )
    c1, c2 = st.columns(2)
    th["c_freq"] = (
        c1.number_input("C: freq. mínima (Hz)", value=float(_default_th["c_freq"][0]), key="c_lo"),
        c2.number_input("C: freq. máxima (Hz)", value=float(_default_th["c_freq"][1]), key="c_hi"),
    )
    d1, d2 = st.columns(2)
    th["d_freq"] = (
        d1.number_input("D: freq. mínima (Hz)", value=float(_default_th["d_freq"][0]), key="d_lo"),
        d2.number_input("D: freq. máxima (Hz)", value=float(_default_th["d_freq"][1]), key="d_hi"),
    )
    th["d_amp_max"] = st.number_input(
        f"D: amplitude máxima ({_unit})", value=float(_default_th["d_amp_max"]), step=0.01, key="d_amp"
    )
    e1a, e1b = st.columns(2)
    th["e1_freq"] = (
        e1a.number_input("E1: freq. mínima (Hz)", value=float(_default_th["e1_freq"][0]), key="e1_lo"),
        e1b.number_input("E1: freq. máxima (Hz)", value=float(_default_th["e1_freq"][1]), key="e1_hi"),
    )
    th["e1_mean_max"] = st.number_input(
        f"E1: nível médio máx. ({_unit})", value=float(_default_th["e1_mean_max"]), key="e1_mean"
    )
    e2a, e2b = st.columns(2)
    th["e2_freq"] = (
        e2a.number_input("E2: freq. mínima (Hz)", value=float(_default_th["e2_freq"][0]), key="e2_lo"),
        e2b.number_input("E2: freq. máxima (Hz)", value=float(_default_th["e2_freq"][1]), key="e2_hi"),
    )
    th["e2_amp_min"] = st.number_input(
        f"E2: amplitude mínima ({_unit})", value=float(_default_th["e2_amp_min"]), step=0.01, key="e2_amp"
    )
    ga, gb = st.columns(2)
    th["g_freq"] = (
        ga.number_input("G: freq. mínima (Hz)", value=float(_default_th["g_freq"][0]), key="g_lo"),
        gb.number_input("G: freq. máxima (Hz)", value=float(_default_th["g_freq"][1]), key="g_hi"),
    )
    th["g_amp_min"] = st.number_input(
        f"G: amplitude mínima ({_unit})", value=float(_default_th["g_amp_min"]), step=0.01, key="g_amp"
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Classificação heurística automática (1ª passagem). Não substitui "
    "anotação visual especializada em softwares como Stylet+a/PROBE."
)

# ---------------------------------------------------------------------------
# Corpo principal
# ---------------------------------------------------------------------------

st.title("🦟 Classificador automático de EPG")
st.markdown(
    "Envie um ou mais arquivos brutos de EPG (`.D01`, `.D02`, ...) para "
    "reconstituir a gravação completa, classificar os waveforms de acordo "
    "com critérios da literatura (Bonani et al. 2010) e visualizar os "
    "resultados interativamente."
)

if not uploaded_files:
    st.info("⬅️ Envie arquivos na barra lateral para começar.")
    with st.expander("📚 Referência de waveforms (Bonani et al. 2010)"):
        ref_rows = []
        for code, info in WAVEFORM_INFO.items():
            if code == "Indeterminado":
                continue
            ref_rows.append(
                {
                    "Waveform": info["label"],
                    "Faixa de frequência": info["lit_freq_range"],
                    "Duração média/evento (min, literatura)": info["lit_mean_event_min"],
                    "Descrição": info["desc"],
                }
            )
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
    st.stop()

# --- Parsing ---
parsed_files = []
parse_errors = []
for uf in uploaded_files:
    try:
        raw_bytes = uf.read()
        parsed = parse_epg_bytes(raw_bytes, uf.name)
        parsed_files.append(parsed)
    except Exception as e:
        parse_errors.append((uf.name, str(e)))

if parse_errors:
    st.error("Alguns arquivos não puderam ser lidos: " + ", ".join(f"{n} ({e})" for n, e in parse_errors))

if not parsed_files:
    st.stop()

channels = group_and_stitch(parsed_files)
channels = sorted(channels, key=lambda c: c.channel_key)

st.success(
    f"{len(parsed_files)} arquivo(s) lido(s), agrupados em {len(channels)} canal(is)/inseto(s)."
)

channel_keys = [c.channel_key for c in channels]
selected_key = st.selectbox("Selecione o canal/inseto para análise:", channel_keys)
channel = next(c for c in channels if c.channel_key == selected_key)

with st.expander("ℹ️ Detalhes dos segmentos deste canal"):
    seg_df = pd.DataFrame(channel.segments)
    seg_df["duracao_min"] = (seg_df["end_s"] - seg_df["start_s"]) / 60
    st.dataframe(seg_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Taxa de amostragem: {channel.sample_rate:.3f} Hz · "
        f"Início da gravação: {channel.start_date} {channel.start_time} · "
        f"Duração total: {channel.total_seconds/3600:.2f} h"
    )

# --- Processamento (com cache para não reprocessar ao mudar apenas thresholds visuais) ---


@st.cache_data(show_spinner="Extraindo features do sinal...")
def _extract_cached(values_bytes, fs, win_seconds, overlap, advanced, remove_baseline):
    values = np.frombuffer(values_bytes, dtype=np.float64)
    if advanced:
        return extract_features_advanced(
            values, fs, win_seconds=win_seconds, overlap=overlap,
            remove_baseline=remove_baseline,
        )
    return extract_features(values, fs, win_seconds=win_seconds, overlap=overlap)


feat_df = _extract_cached(
    channel.values.tobytes(), channel.sample_rate, win_seconds, overlap,
    is_advanced, remove_baseline,
)

if use_ml_model and is_advanced:
    model = ml_bundle["model"]
    le = ml_bundle["label_encoder"]
    model_feat_cols = ml_bundle["feature_columns"]
    missing_cols = [c for c in model_feat_cols if c not in feat_df.columns]
    if missing_cols:
        st.error(
            f"O modelo espera features que não foram encontradas: {missing_cols}. "
            "Verifique se o tamanho de janela/sobreposição são compatíveis com o "
            "treino, ou use a classificação heurística."
        )
        st.stop()
    X_pred = feat_df[model_feat_cols].to_numpy()
    y_pred = model.predict(X_pred)
    feat_df = feat_df.copy()
    feat_df["waveform"] = le.inverse_transform(y_pred)
    classification_mode_label = "🤖 Modelo de ML treinado"
else:
    feat_df = classify_features(feat_df, th)
    classification_mode_label = "📐 Heurística (thresholds)"
st.info(f"Modo de classificação ativo: **{classification_mode_label}**")

runs_df = compute_runs(feat_df)
summary_df = summarize_runs(runs_df, channel.total_seconds)

# tempo absoluto (se houver data/hora de início disponível)
start_dt = None
if channel.start_date and channel.start_time:
    for fmt in ("%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            start_dt = datetime.strptime(f"{channel.start_date} {channel.start_time}", fmt)
            break
        except ValueError:
            continue

# ---------------------------------------------------------------------------
# Tabs de visualização
# ---------------------------------------------------------------------------

tab_overview, tab_timeline, tab_trace, tab_features, tab_events, tab_export = st.tabs(
    [
        "📊 Visão geral",
        "🕒 Linha do tempo (Gantt)",
        "📈 Traço do sinal",
        "🔬 Explorador de features",
        "📋 Eventos detalhados",
        "⬇️ Exportar",
    ]
)

# --- Overview ---
with tab_overview:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Proporção de tempo por waveform")
        if not summary_df.empty:
            pie_df = summary_df.copy()
            pie_df["cor"] = pie_df["waveform"].map(lambda w: WAVEFORM_INFO[w]["color"])
            fig_pie = px.pie(
                pie_df,
                names="waveform",
                values="duracao_total_min (WDI)",
                color="waveform",
                color_discrete_map={w: WAVEFORM_INFO[w]["color"] for w in WAVEFORM_INFO},
                hole=0.45,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Sem dados suficientes para classificar.")

    with col2:
        st.subheader("Métricas por waveform")
        st.caption("NWEI = nº de eventos · WDI = duração total (min) · WDE = duração média por evento (min)")
        if not summary_df.empty:
            display_df = summary_df.copy()
            display_df["waveform"] = display_df["waveform"].map(lambda w: WAVEFORM_INFO[w]["label"])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Comparação com valores médios da literatura")
    st.caption("Bonani et al. (2010) — duração média por evento (min), gravações de 8h em D. citri")
    lit_rows = []
    for code, info in WAVEFORM_INFO.items():
        if code == "Indeterminado" or info["lit_mean_event_min"] is None:
            continue
        computed = summary_df[summary_df["waveform"] == code]["duracao_media_evento_min (WDE)"]
        lit_rows.append(
            {
                "Waveform": info["label"],
                "Literatura (min/evento)": info["lit_mean_event_min"],
                "Este arquivo (min/evento)": computed.iloc[0] if not computed.empty else "—",
            }
        )
    st.dataframe(pd.DataFrame(lit_rows), use_container_width=True, hide_index=True)

    n_indet = (feat_df["waveform"] == "Indeterminado").sum()
    pct_indet = 100 * n_indet / len(feat_df) if len(feat_df) else 0
    if pct_indet > 25:
        st.warning(
            f"⚠️ {pct_indet:.1f}% das janelas ficaram como 'Indeterminado'. "
            "Considere ajustar os thresholds na barra lateral ou aumentar o "
            "tamanho da janela para reduzir ruído de alta frequência."
        )

# --- Timeline Gantt ---
with tab_timeline:
    st.subheader("Linha do tempo dos waveforms (estilo Gantt)")
    if not runs_df.empty:
        gantt_df = runs_df.copy()
        if start_dt:
            gantt_df["start_dt"] = gantt_df["start_s"].apply(lambda s: start_dt + timedelta(seconds=s))
            gantt_df["end_dt"] = gantt_df["end_s"].apply(lambda s: start_dt + timedelta(seconds=s))
        else:
            epoch = datetime(2000, 1, 1)
            gantt_df["start_dt"] = gantt_df["start_s"].apply(lambda s: epoch + timedelta(seconds=s))
            gantt_df["end_dt"] = gantt_df["end_s"].apply(lambda s: epoch + timedelta(seconds=s))

        gantt_df["waveform_label"] = gantt_df["waveform"].map(lambda w: WAVEFORM_INFO[w]["label"])

        fig_gantt = px.timeline(
            gantt_df,
            x_start="start_dt",
            x_end="end_dt",
            y=["Canal: " + selected_key] * len(gantt_df),
            color="waveform",
            color_discrete_map={w: WAVEFORM_INFO[w]["color"] for w in WAVEFORM_INFO},
            hover_data={"duration_s": True, "waveform_label": True},
        )
        fig_gantt.update_yaxes(title="")
        fig_gantt.update_layout(height=250, legend_title_text="Waveform")
        st.plotly_chart(fig_gantt, use_container_width=True)

        st.caption(
            "Cada barra colorida representa um evento contínuo classificado. "
            "Passe o mouse para ver duração e tipo."
        )
    else:
        st.warning("Sem eventos para exibir.")

# --- Trace ---
with tab_trace:
    st.subheader("Traço do sinal bruto (com classificação sobreposta)")

    max_points = st.slider("Resolução do gráfico (nº máx. de pontos)", 1000, 20000, 5000, 1000)
    decim = max(1, len(feat_df) // max_points)
    plot_df = feat_df.iloc[::decim].copy()
    plot_df["t_min"] = plot_df["t_center_s"] / 60.0

    fig_trace = go.Figure()
    fig_trace.add_trace(
        go.Scattergl(
            x=plot_df["t_min"],
            y=plot_df["mean_v"],
            mode="lines",
            line=dict(color="rgba(30,30,30,0.7)", width=1),
            name="Tensão média (V)",
        )
    )
    # sombreamento de fundo por waveform (apenas para runs >= 10s para não poluir)
    if not runs_df.empty:
        for _, r in runs_df.iterrows():
            if r["duration_s"] < 10:
                continue
            fig_trace.add_vrect(
                x0=r["start_s"] / 60.0,
                x1=r["end_s"] / 60.0,
                fillcolor=WAVEFORM_INFO[r["waveform"]]["color"],
                opacity=0.25,
                line_width=0,
            )
    fig_trace.update_layout(
        xaxis_title="Tempo (min)",
        yaxis_title="Tensão (V)",
        height=450,
        showlegend=True,
    )
    st.plotly_chart(fig_trace, use_container_width=True)
    st.caption(
        "Linha: tensão média por janela. Faixas coloridas: eventos classificados "
        "com duração ≥ 10 s (eventos mais curtos são omitidos do sombreamento por clareza visual)."
    )

    st.markdown("##### Zoom em um intervalo específico")
    t_max_min = feat_df["t_center_s"].max() / 60.0 if not feat_df.empty else 1.0
    zoom_range = st.slider(
        "Intervalo (min)", 0.0, float(t_max_min), (0.0, min(10.0, float(t_max_min))), 0.5
    )
    zoom_mask = (feat_df["t_center_s"] / 60.0 >= zoom_range[0]) & (feat_df["t_center_s"] / 60.0 <= zoom_range[1])
    zoom_df = feat_df[zoom_mask]
    if not zoom_df.empty:
        fig_zoom = go.Figure()
        fig_zoom.add_trace(
            go.Scattergl(
                x=zoom_df["t_center_s"] / 60.0,
                y=zoom_df["mean_v"],
                mode="lines+markers",
                marker=dict(
                    size=5,
                    color=[WAVEFORM_INFO[w]["color"] for w in zoom_df["waveform"]],
                ),
                line=dict(color="rgba(30,30,30,0.4)", width=1),
            )
        )
        fig_zoom.update_layout(xaxis_title="Tempo (min)", yaxis_title="Tensão (V)", height=350)
        st.plotly_chart(fig_zoom, use_container_width=True)

# --- Explorador de features ---
with tab_features:
    st.subheader("Explorador de features por waveform")
    if not is_advanced:
        st.info(
            "As features avançadas (estatísticas, espectrais e de wavelet) só "
            "estão disponíveis no modo **Avançado (estilo DiscoEPG)**. "
            "Selecione-o na barra lateral para explorá-las e calibrar os thresholds."
        )
    else:
        feat_cols = [c for c in feature_columns(feat_df) if c not in ("waveform",)]
        st.caption(
            "Distribuição das features por waveform classificado. Útil para "
            "calibrar thresholds: observe onde cada waveform se separa e ajuste "
            "os limiares na barra lateral. Também é a base para treinar um "
            "classificador supervisionado (ML)."
        )
        sel_feat = st.selectbox("Feature para inspecionar:", feat_cols, index=feat_cols.index("amp_p2p") if "amp_p2p" in feat_cols else 0)

        # box plot da feature por waveform
        plot_df = feat_df[[sel_feat, "waveform"]].copy()
        fig_box = px.box(
            plot_df,
            x="waveform",
            y=sel_feat,
            color="waveform",
            color_discrete_map={w: WAVEFORM_INFO[w]["color"] for w in WAVEFORM_INFO},
            points="outliers",
        )
        fig_box.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title=sel_feat)
        st.plotly_chart(fig_box, use_container_width=True)

        # matriz de correlação entre features (amostra para performance)
        st.markdown("##### Correlação entre features")
        sample_n = min(3000, len(feat_df))
        corr = feat_df[feat_cols].sample(sample_n, random_state=0).corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "Features muito correlacionadas (|r|→1) são redundantes; para um "
            "modelo de ML, considere manter apenas uma de cada grupo redundante."
        )

# --- Eventos detalhados ---
with tab_events:
    st.subheader("Tabela de eventos (runs) classificados")
    if not runs_df.empty:
        display_runs = runs_df.copy()
        display_runs["waveform"] = display_runs["waveform"].map(lambda w: WAVEFORM_INFO[w]["label"])
        display_runs["start_min"] = (display_runs["start_s"] / 60).round(2)
        display_runs["end_min"] = (display_runs["end_s"] / 60).round(2)
        display_runs["duration_min"] = (display_runs["duration_s"] / 60).round(2)

        wf_filter = st.multiselect(
            "Filtrar por waveform:",
            options=sorted(display_runs["waveform"].unique()),
            default=sorted(display_runs["waveform"].unique()),
        )
        min_dur = st.slider("Duração mínima do evento (s)", 0.0, float(display_runs["duration_s"].max()), 0.0)

        filtered = display_runs[
            display_runs["waveform"].isin(wf_filter) & (display_runs["duration_s"] >= min_dur)
        ]
        st.dataframe(
            filtered[["waveform", "start_min", "end_min", "duration_min"]].sort_values("start_min"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(filtered)} de {len(display_runs)} eventos exibidos.")
    else:
        st.warning("Sem eventos para exibir.")

# --- Export ---
with tab_export:
    st.subheader("Exportar resultados")

    csv_feat = feat_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar features por janela (CSV)",
        data=csv_feat,
        file_name=f"{selected_key}_features.csv",
        mime="text/csv",
    )

    csv_runs = runs_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar eventos classificados / runs (CSV)",
        data=csv_runs,
        file_name=f"{selected_key}_eventos.csv",
        mime="text/csv",
    )

    csv_summary = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar resumo por waveform (CSV)",
        data=csv_summary,
        file_name=f"{selected_key}_resumo.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "Os arquivos exportados podem ser usados para análises estatísticas "
        "posteriores (ex: ANOVA comparando tratamentos, ou como dataset de "
        "treino para um classificador supervisionado mais robusto)."
    )

st.markdown("---")
st.caption(
    "Metodologia de classificação: heurística baseada em frequência dominante (FFT) "
    "e amplitude pico-a-pico por janela deslizante, seguindo as faixas descritas em "
    "Bonani, J.P. et al. (2010) Entomologia Experimentalis et Applicata, 134: 35-49. "
    "Resultados automáticos devem ser tratados como triagem exploratória, não como "
    "anotação definitiva de waveforms."
)
