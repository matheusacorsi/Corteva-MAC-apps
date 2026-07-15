"""QC visualization: raw trace with predicted/ground-truth waveform segments
color-coded, in the spirit of Stylet+/DiscoEPG review plots."""

from __future__ import annotations

from epg_tool.io.session import EPGSession
from epg_tool.species.profile import SpeciesProfile

_MUTED_GRID = "#e1e0d9"
_MUTED_AXIS = "#c3c2b7"
_MUTED_TEXT = "#52514e"


def plot_session(
    session: EPGSession,
    profile: SpeciesProfile,
    start_s: float | None = None,
    end_s: float | None = None,
    ax=None,
):
    """Plot the raw voltage trace over [start_s, end_s) with background
    spans color-coded by waveform label (per the species profile)."""
    import matplotlib.pyplot as plt

    start_s = 0.0 if start_s is None else start_s
    end_s = session.duration_s if end_s is None else end_s

    start_idx = max(0, round(start_s * session.sample_rate_hz))
    end_idx = min(len(session.samples), round(end_s * session.sample_rate_hz))

    time = session.time_axis()[start_idx:end_idx]
    volts = session.samples[start_idx:end_idx]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    else:
        fig = ax.figure

    seen_labels: set[str] = set()
    for seg in session.segments:
        if seg.end_s <= start_s or seg.start_s >= end_s:
            continue
        label = profile.label_for_code(seg.code) or f"code {seg.code}"
        color = next(
            (w.color for w in profile.waveforms if w.code == seg.code), "#cccccc"
        )
        span_start = max(seg.start_s, start_s)
        span_end = min(seg.end_s, end_s)
        ax.axvspan(span_start, span_end, color=color, alpha=0.25, linewidth=0)
        seen_labels.add(label)

    ax.plot(time, volts, color="#0b0b0b", linewidth=0.6)

    ax.set_xlim(start_s, end_s)
    ax.set_xlabel("Time (s)", color=_MUTED_TEXT)
    ax.set_ylabel("Voltage (V)", color=_MUTED_TEXT)
    ax.grid(True, color=_MUTED_GRID, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(_MUTED_AXIS)
    ax.tick_params(colors=_MUTED_TEXT)

    if seen_labels:
        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(color=w.color, alpha=0.5, label=w.label)
            for w in profile.waveforms
            if w.label in seen_labels
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            ncols=min(len(handles), 6),
            fontsize=8,
        )

    ax.set_title(
        f"{session.insect_id} [{profile.common_name}] — {start_s:.0f}s–{end_s:.0f}s",
        color="#0b0b0b",
        fontsize=10,
    )

    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_time_distribution_pies(sessions: dict[str, EPGSession], profile: SpeciesProfile):
    """Side-by-side pie charts of % of recording time per waveform label,
    one panel per (title -> session) entry -- e.g. {"Predicted": ...,
    "Ground truth": ...}. Colors and label order follow the species
    profile, and a single shared legend covers every label seen in any
    panel (so a label a panel happens to lack doesn't shift its color)."""
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    from .parameters import nonsequential_parameters

    color_by_label = {w.label: w.color for w in profile.waveforms}
    label_order = [w.label for w in profile.waveforms]

    durations_by_title = {}
    labels_present_anywhere: set[str] = set()
    for title, session in sessions.items():
        nonseq = nonsequential_parameters(session, profile).set_index("label")["total_duration_s"]
        durations_by_title[title] = nonseq
        labels_present_anywhere |= set(nonseq.index[nonseq > 0])
    ordered_labels = [label for label in label_order if label in labels_present_anywhere]

    n = len(sessions)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), dpi=150)
    axes = [axes] if n == 1 else list(axes)

    for ax, (title, durations) in zip(axes, durations_by_title.items()):
        present = [label for label in ordered_labels if durations.get(label, 0.0) > 0]
        sizes = [durations[label] for label in present]
        colors = [color_by_label[label] for label in present]

        _wedges, _texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 1, "edgecolor": "#fcfcfb"},
            textprops={"fontsize": 9},
        )
        for label_text in autotexts:
            label_text.set_color("white")
            label_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground="#0b0b0b")])
        ax.set_title(title, fontsize=11, color="#0b0b0b")

    legend_handles = [mpatches.Patch(color=color_by_label[label], label=label) for label in ordered_labels]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Time distribution by waveform", fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    return fig
