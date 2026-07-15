"""Streamlit dashboard: upload a .D0x recording (+ optional .ANA ground
truth), classify it with the rule-based baseline or a trained model, and
review/download the results.

Training happens offline via the CLI (`epg-tool train`) on a growing
folder of labeled recordings -- that's a long-running, many-file batch
job that doesn't fit Streamlit's request/response model. This app is the
"serve a trained model" half of the pipeline: point it at a species
profile, upload a recording, get predictions back.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make epg_tool importable without a package install -- works whether
# this file sits at the repo root or nested in a monorepo subfolder,
# since it's resolved relative to this file's own location, not the
# installer's working directory (see requirements.txt for why that
# distinction matters here).
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from epg_tool.export.ana_writer import build_ana_from_windows, predictions_to_labeled_segments
from epg_tool.export.parameters import (
    nonsequential_parameters,
    sequential_parameters,
    transition_matrix,
)
from epg_tool.export.plotting import plot_session, plot_time_distribution_pies
from epg_tool.features import extract_features, make_inference_windows, make_windows
from epg_tool.features.baseline import estimate_np_baseline
from epg_tool.features.windowing import build_sample_labels
from epg_tool.io.session import EPGSession, build_session_from_bytes
from epg_tool.models.registry import has_trained_model, load_model
from epg_tool.models.rules import RuleBasedClassifier
from epg_tool.models.tabular import TabularModel
from epg_tool.species.profile import list_profiles, load_profile
from epg_tool.training.evaluate import evaluate

st.set_page_config(page_title="EPG Waveform Classifier", layout="wide")
st.title("EPG Waveform Classifier")
st.caption(
    "Upload a Stylet+ .D0x recording (all hourly files) and, optionally, its "
    ".ANA ground truth. Pick a species profile and classifier, then review "
    "and download the results."
)

with st.sidebar:
    st.header("Configuration")
    species_name = st.selectbox("Species profile", list_profiles())
    profile = load_profile(species_name)
    st.caption(profile.common_name)

    model_options = ["rule_based", "random_forest", "xgboost"]
    model_type = st.selectbox("Classifier", model_options)

    uploaded_model: TabularModel | None = None
    if model_type != "rule_based":
        model_upload = st.file_uploader(
            f"Optional: upload a trained {model_type} model (.joblib) instead of the bundled one",
            type=["joblib"],
            key=f"model_upload_{model_type}",
        )
        if model_upload is not None:
            model_bytes = model_upload.getvalue()
            try:
                uploaded_model = TabularModel.load_from_bytes(model_bytes)
                st.caption(f"Using uploaded model ({len(model_bytes) / 1e6:.1f} MB)")
            except Exception as exc:
                st.error(f"Couldn't load that model file: {exc}")
        elif not has_trained_model(profile, model_type):
            st.warning(
                f"No trained {model_type!r} model bundled for {species_name!r} and none "
                "uploaded -- falling back to the rule-based baseline. Train one with "
                f"`epg-tool train <data-folder> --species {species_name} --model {model_type}`, "
                "or upload a `.joblib` file above."
            )
            model_type = "rule_based"

    window_s = st.number_input("Window size (s)", min_value=0.1, value=1.0, step=0.5)


def get_classifier(model_type: str, uploaded_model: TabularModel | None, profile):
    if model_type == "rule_based":
        return RuleBasedClassifier(profile)
    if uploaded_model is not None:
        return uploaded_model
    return load_model(profile, model_type)

st.subheader("1. Upload recording")
col1, col2 = st.columns(2)
with col1:
    d0x_uploads = st.file_uploader(
        "All .D0x files for one recording (e.g. .D01, .D02, ...)",
        accept_multiple_files=True,
        type=None,
    )
with col2:
    ana_upload = st.file_uploader(
        "Optional: matching .ANA ground truth (enables QC comparison + accuracy metrics)",
        accept_multiple_files=False,
        type=None,
    )

if not d0x_uploads:
    st.info("Upload at least one .D0x file to continue.")
    st.stop()

default_insect_id = d0x_uploads[0].name.split(".")[0]
insect_id = st.text_input("Insect / recording ID", value=default_insect_id)

session = build_session_from_bytes(
    d0x_files=[(f.name, f.getvalue()) for f in d0x_uploads],
    ana_bytes=ana_upload.getvalue() if ana_upload else None,
    insect_id=insect_id,
    sentinel_codes=profile.sentinel_codes,
)
has_ground_truth = len(session.segments) > 0
st.success(
    f"Loaded {len(d0x_uploads)} file(s): {session.duration_s:.0f}s at {session.sample_rate_hz:.0f} Hz"
    + (" (with ground truth)" if has_ground_truth else " (no ground truth -- inference only)")
)

st.subheader("2. Classify")
if st.button("Run classification", type="primary"):
    if has_ground_truth:
        np_code = profile.label_to_code.get("Np")
        np_mask = (
            build_sample_labels(len(session.samples), session.segments) == np_code
            if np_code is not None
            else np.zeros(len(session.samples), dtype=bool)
        )
    else:
        np_mask = np.zeros(len(session.samples), dtype=bool)
    context = {"np_baseline_v": estimate_np_baseline(session.samples, np_mask)}

    windows = make_inference_windows(session.samples, session.sample_rate_hz, window_s=window_s)
    with st.spinner(f"Extracting features for {len(windows)} windows..."):
        X = pd.DataFrame([extract_features(w.samples, session.sample_rate_hz, context=context) for w in windows])

    clf = get_classifier(model_type, uploaded_model, profile)
    pred_codes = clf.predict(X)

    predicted_segments = predictions_to_labeled_segments(windows, pred_codes, session.sample_rate_hz)
    predicted_session = EPGSession(
        insect_id=insect_id,
        samples=session.samples,
        sample_rate_hz=session.sample_rate_hz,
        source_files=[],
        segments=predicted_segments,
        recording_end_s=session.recording_end_s,
    )
    st.session_state["result"] = {
        "windows": windows,
        "pred_codes": pred_codes,
        "predicted_session": predicted_session,
    }

if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
predicted_session = result["predicted_session"]

st.subheader("3. Review")
max_t = session.duration_s
start_s, end_s = st.slider(
    "Time range to display (s)", min_value=0.0, max_value=float(max_t), value=(0.0, float(min(max_t, 3600.0)))
)

plot_cols = st.columns(2) if has_ground_truth else [st.container()]
with plot_cols[0]:
    st.markdown("**Predicted**")
    fig, _ = plot_session(predicted_session, profile, start_s=start_s, end_s=end_s)
    st.pyplot(fig)
    plt.close(fig)

if has_ground_truth:
    with plot_cols[1]:
        st.markdown("**Ground truth**")
        fig, _ = plot_session(session, profile, start_s=start_s, end_s=end_s)
        st.pyplot(fig)
        plt.close(fig)

pie_sessions = {"Predicted": predicted_session}
if has_ground_truth:
    pie_sessions["Ground truth"] = session
fig = plot_time_distribution_pies(pie_sessions, profile)
st.pyplot(fig)
plt.close(fig)

pred_df = predicted_session.to_dataframe()
pred_df["label"] = pred_df["code"].map(profile.code_to_label)
st.markdown("**Predicted segments**")
st.dataframe(pred_df, use_container_width=True)

if has_ground_truth:
    st.subheader("4. Accuracy against ground truth")
    gt_windows = make_windows(session, window_s=window_s)
    gt_context = {"np_baseline_v": context["np_baseline_v"]}
    gt_X = pd.DataFrame(
        [extract_features(w.samples, session.sample_rate_hz, context=gt_context) for w in gt_windows]
    )
    clf = get_classifier(model_type, uploaded_model, profile)
    gt_pred = clf.predict(gt_X)
    gt_true = np.array([w.label_code for w in gt_windows])

    eval_result = evaluate(gt_true, gt_pred, profile)
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{eval_result.accuracy:.1%}")
    m2.metric("Time-overlap agreement", f"{eval_result.time_overlap_agreement:.1%}")
    st.dataframe(eval_result.classification_report, use_container_width=True)
    st.markdown("**Confusion matrix**")
    st.dataframe(eval_result.confusion_matrix, use_container_width=True)

st.subheader("5. Behavioral parameters")
param_session = session if has_ground_truth else predicted_session
nonseq = nonsequential_parameters(param_session, profile)
seq = sequential_parameters(param_session, profile)
trans = transition_matrix([param_session], profile)
st.caption(
    "Computed from ground truth" if has_ground_truth else "Computed from predictions (no ground truth uploaded)"
)
st.markdown("**Non-sequential (Backus et al. 2007 style)**")
st.dataframe(nonseq, use_container_width=True)
st.markdown("**Sequential (Sarria et al. 2009 style)**")
st.dataframe(pd.DataFrame([seq]), use_container_width=True)
st.markdown("**Transition matrix**")
st.dataframe(trans, use_container_width=True)

st.subheader("6. Downloads")
d1, d2 = st.columns(2)
with d1:
    ana_bytes = build_ana_from_windows(result["windows"], result["pred_codes"], session, profile)
    st.download_button(
        "Download predicted .ANA",
        data=ana_bytes,
        file_name=f"{insect_id}_predicted_.ANA",
        mime="application/octet-stream",
    )
with d2:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        nonseq.to_excel(writer, sheet_name="nonsequential", index=False)
        pd.DataFrame([seq]).to_excel(writer, sheet_name="sequential", index=False)
        trans.to_excel(writer, sheet_name="transition_matrix")
    st.download_button(
        "Download parameters (Excel)",
        data=buffer.getvalue(),
        file_name=f"{insect_id}_parameters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
