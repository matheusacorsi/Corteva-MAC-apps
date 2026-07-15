from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from epg_tool.export.plotting import plot_time_distribution_pies
from epg_tool.io.session import EPGSession, LabeledSegment
from epg_tool.species.profile import load_profile


def _toy_session(insect_id: str, code_sequence: list[tuple[int, float, float]]) -> EPGSession:
    segments = [
        LabeledSegment(code=c, start_s=s, end_s=e, start_idx=int(s * 10), end_idx=int(e * 10))
        for c, s, e in code_sequence
    ]
    return EPGSession(
        insect_id=insect_id,
        samples=np.zeros(int(code_sequence[-1][2] * 10)),
        sample_rate_hz=10.0,
        source_files=[],
        segments=segments,
    )


def test_plot_time_distribution_pies_two_panels():
    profile = load_profile("diaphorina_citri")
    predicted = _toy_session("bug1", [(1, 0, 10), (2, 10, 20)])  # Np, C
    ground_truth = _toy_session("bug1", [(1, 0, 5), (2, 5, 15), (4, 15, 20)])  # Np, C, E1

    fig = plot_time_distribution_pies({"Predicted": predicted, "Ground truth": ground_truth}, profile)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Predicted"
    assert fig.axes[1].get_title() == "Ground truth"

    legend_labels = {t.get_text() for t in fig.legends[0].get_texts()}
    assert legend_labels == {"Np", "C", "E1"}


def test_plot_time_distribution_pies_single_panel():
    profile = load_profile("diaphorina_citri")
    predicted = _toy_session("bug1", [(1, 0, 10), (2, 10, 20)])

    fig = plot_time_distribution_pies({"Predicted": predicted}, profile)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Predicted"


def test_plot_time_distribution_pies_legend_follows_fixed_species_order():
    profile = load_profile("diaphorina_citri")
    # Insert codes out of profile order (E2=5, then C=2) -- legend must
    # still follow the profile's fixed Np/C/D/E1/E2/G order, not insertion order.
    session = _toy_session("bug1", [(5, 0, 10), (2, 10, 20)])

    fig = plot_time_distribution_pies({"Predicted": session}, profile)
    legend_labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend_labels == ["C", "E2"]  # C before E2 per profile.waveforms order
