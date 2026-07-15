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
