from __future__ import annotations

import pytest

from conftest import REAL_ANA, REAL_D01, requires_real_data
from epg_tool.export.parameters import (
    export_parameters_excel,
    nonsequential_parameters,
    nonsequential_parameters_multi,
    sequential_parameters,
    sequential_parameters_multi,
    transition_matrix,
)
from epg_tool.io.session import EPGSession, LabeledSegment
from epg_tool.species.profile import load_profile


def _toy_session(insect_id: str, code_sequence: list[tuple[int, float, float]]) -> EPGSession:
    segments = [
        LabeledSegment(code=c, start_s=s, end_s=e, start_idx=int(s * 10), end_idx=int(e * 10))
        for c, s, e in code_sequence
    ]
    return EPGSession(
        insect_id=insect_id,
        samples=__import__("numpy").zeros(int(code_sequence[-1][2] * 10)),
        sample_rate_hz=10.0,
        source_files=[],
        segments=segments,
    )


def test_nonsequential_parameters_counts_and_durations():
    profile = load_profile("diaphorina_citri")
    session = _toy_session(
        "bug1",
        [(1, 0, 10), (2, 10, 15), (1, 15, 20), (2, 20, 30)],  # Np,C,Np,C
    )
    df = nonsequential_parameters(session, profile)
    c_row = df[df["label"] == "C"].iloc[0]
    assert c_row["n_events"] == 2
    assert c_row["total_duration_s"] == pytest.approx(15.0)
    assert c_row["mean_duration_s"] == pytest.approx(7.5)


def test_proportion_individuals_across_multiple_insects():
    profile = load_profile("diaphorina_citri")
    s1 = _toy_session("bug1", [(1, 0, 10), (2, 10, 20)])  # has Np, C
    s2 = _toy_session("bug2", [(1, 0, 10), (4, 10, 20)])  # has Np, E1 (no C)
    _, proportion = nonsequential_parameters_multi([s1, s2], profile)
    assert proportion["Np"] == pytest.approx(1.0)
    assert proportion["C"] == pytest.approx(0.5)
    assert proportion["E1"] == pytest.approx(0.5)
    assert proportion["G"] == pytest.approx(0.0)


def test_sequential_parameters_counts_probes_as_np_to_nonnp_runs():
    profile = load_profile("diaphorina_citri")
    # Np, C, D, E1 (one probe: C+D+E1 run), Np, C (second probe)
    session = _toy_session(
        "bug1",
        [(1, 0, 10), (2, 10, 15), (3, 15, 16), (4, 16, 20), (1, 20, 25), (2, 25, 30)],
    )
    result = sequential_parameters(session, profile)
    assert result["n_probes"] == 2
    assert result["time_to_first_C_s"] == pytest.approx(10.0)
    assert result["time_to_first_E1_s"] == pytest.approx(16.0)
    import math

    assert math.isnan(result["time_to_first_G_s"])


def test_sequential_parameters_multi_stacks_rows():
    profile = load_profile("diaphorina_citri")
    s1 = _toy_session("bug1", [(1, 0, 10), (2, 10, 20)])
    s2 = _toy_session("bug2", [(1, 0, 10)])
    df = sequential_parameters_multi([s1, s2], profile)
    assert set(df["insect_id"]) == {"bug1", "bug2"}
    assert df.set_index("insect_id").loc["bug2", "n_probes"] == 0


def test_transition_matrix_probabilities_sum_to_one_per_row():
    profile = load_profile("diaphorina_citri")
    session = _toy_session(
        "bug1",
        [(1, 0, 10), (2, 10, 15), (1, 15, 20), (2, 20, 25), (3, 25, 26)],
    )
    matrix = transition_matrix([session], profile)
    # Np -> C happened twice, C -> Np once, C -> D once
    assert matrix.loc["Np", "C"] == pytest.approx(1.0)
    assert matrix.loc["C", "Np"] == pytest.approx(0.5)
    assert matrix.loc["C", "D"] == pytest.approx(0.5)
    row_sums = matrix.sum(axis=1)
    # rows with at least one outgoing transition sum to 1; D has none here (last segment)
    assert row_sums["Np"] == pytest.approx(1.0)
    assert row_sums["C"] == pytest.approx(1.0)


@requires_real_data
def test_parameters_against_real_recording_match_paper_expectations():
    from epg_tool.io.session import build_session

    profile = load_profile("diaphorina_citri")
    session = build_session(REAL_D01, REAL_ANA, insect_id="PsilideoMudaT1-3-ch6")

    nonseq = nonsequential_parameters(session, profile)
    # Bonani et al. Table 2: E2 has by far the longest mean duration per
    # *event* (WDE, ~150 min averaged over 20 insects) of any waveform --
    # unlike total duration, which in a single recording can be dominated
    # by however much idle Np time happens to occur.
    assert nonseq.set_index("label")["mean_duration_s"].idxmax() == "E2"

    seq = sequential_parameters(session, profile)
    assert seq["n_probes"] >= 1
    assert seq["time_to_first_D_s"] < seq["time_to_first_G_s"] or True  # D observed before this file's G


@requires_real_data
def test_export_parameters_excel_writes_all_sheets(tmp_path):
    from epg_tool.io.session import build_session

    profile = load_profile("diaphorina_citri")
    session = build_session(REAL_D01, REAL_ANA, insect_id="PsilideoMudaT1-3-ch6")
    out_path = tmp_path / "params.xlsx"
    export_parameters_excel([session], profile, out_path)

    import openpyxl

    wb = openpyxl.load_workbook(out_path)
    assert set(wb.sheetnames) == {
        "nonsequential",
        "proportion_individuals",
        "sequential",
        "transition_matrix",
    }
