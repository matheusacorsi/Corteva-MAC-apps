"""EPG behavioral parameter computation.

Non-sequential parameters follow the Backus et al. (2007) convention
(number of events, total/mean duration per waveform, proportion of
individuals producing each waveform -- cf. Bonani et al. 2010 Table 2).
Sequential parameters follow Sarria et al. (2009) (number of probes,
time to first event, transition probabilities -- cf. Bonani et al. 2010
Table 3 and Figure 6). Which parameters get computed for a species is
driven by ``profile.parameters``, but the computation itself is generic
across waveform label sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from epg_tool.io.session import EPGSession
from epg_tool.species.profile import SpeciesProfile


def nonsequential_parameters(session: EPGSession, profile: SpeciesProfile) -> pd.DataFrame:
    """Per-waveform n_events / total_duration_s / mean_duration_s for one insect."""
    df = session.to_dataframe()
    if df.empty:
        return pd.DataFrame(columns=["insect_id", "label", "n_events", "total_duration_s", "mean_duration_s"])

    df["label"] = df["code"].map(profile.code_to_label)
    rows = []
    for label, group in df.groupby("label"):
        rows.append(
            {
                "insect_id": session.insect_id,
                "label": label,
                "n_events": len(group),
                "total_duration_s": float(group["duration_s"].sum()),
                "mean_duration_s": float(group["duration_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def nonsequential_parameters_multi(
    sessions: list[EPGSession], profile: SpeciesProfile
) -> tuple[pd.DataFrame, pd.Series]:
    """Same as :func:`nonsequential_parameters`, stacked across insects,
    plus PPW (proportion of individuals producing each waveform)."""
    per_insect = pd.concat(
        [nonsequential_parameters(s, profile) for s in sessions], ignore_index=True
    )
    n_insects = len(sessions)
    all_labels = [w.label for w in profile.waveforms]
    proportion = (
        per_insect.groupby("label")["insect_id"].nunique().reindex(all_labels, fill_value=0) / n_insects
    )
    proportion.name = "proportion_individuals"
    return per_insect, proportion


def sequential_parameters(session: EPGSession, profile: SpeciesProfile) -> dict:
    """Number of probes and time-to-first-event per waveform for one insect.

    A "probe" is a maximal run of consecutive non-Np segments (i.e. one
    continuous stylet insertion, however many waveform types it passes
    through before the insect withdraws back to Np)."""
    df = session.to_dataframe()
    result: dict[str, float] = {"insect_id": session.insect_id}

    np_code = profile.label_to_code.get("Np")
    if df.empty or np_code is None:
        result["n_probes"] = 0
    else:
        is_probe = (df["code"] != np_code).to_numpy()
        starts = is_probe & ~np.concatenate([[False], is_probe[:-1]])
        result["n_probes"] = int(starts.sum())

    for waveform in profile.waveforms:
        matches = df[df["code"] == waveform.code] if not df.empty else df
        result[f"time_to_first_{waveform.label}_s"] = (
            float(matches["start_s"].min()) if not matches.empty else float("nan")
        )
    return result


def sequential_parameters_multi(sessions: list[EPGSession], profile: SpeciesProfile) -> pd.DataFrame:
    return pd.DataFrame([sequential_parameters(s, profile) for s in sessions])


def transition_matrix(sessions: list[EPGSession], profile: SpeciesProfile) -> pd.DataFrame:
    """Row-normalized first-order Markov transition probabilities between
    waveform labels, aggregated across all given sessions (cf. Bonani et
    al. 2010 Figure 6)."""
    labels = [w.label for w in profile.waveforms]
    counts = pd.DataFrame(0.0, index=labels, columns=labels)

    for session in sessions:
        codes = [seg.code for seg in session.segments]
        for code_a, code_b in zip(codes, codes[1:]):
            label_a = profile.label_for_code(code_a)
            label_b = profile.label_for_code(code_b)
            if label_a in labels and label_b in labels:
                counts.loc[label_a, label_b] += 1

    row_sums = counts.sum(axis=1)
    return counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)


def export_parameters_excel(sessions: list[EPGSession], profile: SpeciesProfile, path: str | Path) -> None:
    nonseq, proportion = nonsequential_parameters_multi(sessions, profile)
    seq = sequential_parameters_multi(sessions, profile)
    trans = transition_matrix(sessions, profile)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        nonseq.to_excel(writer, sheet_name="nonsequential", index=False)
        proportion.to_frame().to_excel(writer, sheet_name="proportion_individuals")
        seq.to_excel(writer, sheet_name="sequential", index=False)
        trans.to_excel(writer, sheet_name="transition_matrix")
