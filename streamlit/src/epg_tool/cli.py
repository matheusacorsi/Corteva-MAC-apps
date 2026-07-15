"""Command-line interface: parse/inspect, plot, train, predict, evaluate,
and export-parameters -- one entry point per stage of the pipeline
described in the project README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="EPG waveform classification toolkit", no_args_is_help=True)


def _load_species(species: str):
    from epg_tool.species.profile import load_profile

    return load_profile(species)


def _build_model(model_type: str):
    from epg_tool.models.tabular import random_forest_model, xgboost_model

    if model_type == "random_forest":
        return random_forest_model()
    if model_type == "xgboost":
        return xgboost_model()
    raise typer.BadParameter(f"Unknown model type {model_type!r}. Use random_forest or xgboost.")


@app.command()
def inspect(
    d0x: Path = typer.Argument(..., help="Any .D0x file in the recording's series"),
    ana: Path = typer.Argument(..., help="Matching .ANA annotation file"),
    species: str = typer.Option("diaphorina_citri", help="Species profile name"),
):
    """Parse a .D0x/.ANA pair and print the labeled-segment table."""
    from epg_tool.io.session import build_session

    profile = _load_species(species)
    session = build_session(d0x, ana, insect_id=d0x.stem)
    df = session.to_dataframe()
    df["label"] = df["code"].map(profile.code_to_label)
    pd.set_option("display.width", 120)
    typer.echo(df.to_string(index=False))
    typer.echo(f"\nDuration: {session.duration_s:.1f}s, sample rate: {session.sample_rate_hz} Hz")


@app.command()
def plot(
    d0x: Path = typer.Argument(..., help="Any .D0x file in the recording's series"),
    ana: Path = typer.Argument(..., help="Matching .ANA annotation file"),
    species: str = typer.Option("diaphorina_citri"),
    start: float = typer.Option(0.0, help="Window start (s)"),
    end: Optional[float] = typer.Option(None, help="Window end (s); defaults to full recording"),
    out: Path = typer.Option(Path("qc_plot.png")),
):
    """Render a QC plot (raw trace + color-coded ground-truth segments)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from epg_tool.export.plotting import plot_session
    from epg_tool.io.session import build_session

    profile = _load_species(species)
    session = build_session(d0x, ana, insect_id=d0x.stem)
    fig, _ax = plot_session(session, profile, start_s=start, end_s=end)
    fig.savefig(out)
    plt.close(fig)
    typer.echo(f"Wrote {out}")


@app.command()
def train(
    data_root: Path = typer.Argument(..., help="Folder to recursively search for .D0x/.ANA pairs"),
    species: str = typer.Option("diaphorina_citri"),
    model: str = typer.Option("random_forest", help="random_forest or xgboost"),
    window_s: float = typer.Option(1.0),
    step_s: Optional[float] = typer.Option(None),
    min_purity: float = typer.Option(0.5, help="Drop windows where the majority label covers less than this"),
    test_size: float = typer.Option(0.2),
    val_size: float = typer.Option(0.2),
    out: Optional[Path] = typer.Option(None, help="Where to save the model; defaults to the species registry path"),
):
    """Discover recordings under DATA_ROOT, build a windowed feature
    dataset split by insect individual, train a tabular model, and report
    validation metrics."""
    from epg_tool.models.registry import model_path_for, save_model
    from epg_tool.training.dataset import build_dataset, discover_recordings, group_train_val_test_split
    from epg_tool.training.evaluate import evaluate

    profile = _load_species(species)
    recordings = discover_recordings(data_root)
    if not recordings:
        raise typer.BadParameter(f"No .D0x/.ANA pairs found under {data_root}")
    typer.echo(f"Found {len(recordings)} recording(s): {[r.insect_id for r in recordings]}")

    X, y, groups = build_dataset(recordings, profile, window_s=window_s, step_s=step_s, min_purity=min_purity)
    typer.echo(f"Built {len(X)} windows across {len(set(groups))} insect(s)")

    split = group_train_val_test_split(X, y, groups, test_size=test_size, val_size=val_size)

    clf = _build_model(model)
    clf.fit(*split.train[:2])

    val_X, val_y, _ = split.val
    if len(val_X) > 0:
        val_result = evaluate(val_y, clf.predict(val_X), profile)
        typer.echo(f"\nValidation accuracy: {val_result.accuracy:.3f}")
        typer.echo(f"Validation time-overlap agreement: {val_result.time_overlap_agreement:.3f}")
        typer.echo(val_result.classification_report.to_string())

    if out is None:
        out_path = save_model(clf, profile, model)
    else:
        out_path = out
        clf.save(out_path)
    typer.echo(f"\nSaved model to {out_path}")


@app.command()
def evaluate(
    data_root: Path = typer.Argument(..., help="Folder to recursively search for .D0x/.ANA pairs"),
    species: str = typer.Option("diaphorina_citri"),
    model: str = typer.Option("random_forest"),
    window_s: float = typer.Option(1.0),
    step_s: Optional[float] = typer.Option(None),
    min_purity: float = typer.Option(0.5),
):
    """Evaluate an already-trained model (or the rule-based baseline)
    against every recording under DATA_ROOT."""
    from epg_tool.models.registry import load_model
    from epg_tool.models.rules import RuleBasedClassifier
    from epg_tool.training.dataset import build_dataset, discover_recordings
    from epg_tool.training.evaluate import evaluate as evaluate_predictions

    profile = _load_species(species)
    recordings = discover_recordings(data_root)
    X, y, groups = build_dataset(recordings, profile, window_s=window_s, step_s=step_s, min_purity=min_purity)

    clf = RuleBasedClassifier(profile) if model == "rule_based" else load_model(profile, model)
    result = evaluate_predictions(y, clf.predict(X), profile)

    typer.echo(f"Accuracy: {result.accuracy:.3f}")
    typer.echo(f"Time-overlap agreement: {result.time_overlap_agreement:.3f}")
    typer.echo(result.classification_report.to_string())
    typer.echo("\nConfusion matrix:")
    typer.echo(result.confusion_matrix.to_string())


@app.command()
def predict(
    d0x: Path = typer.Argument(..., help="Any .D0x file in the recording's series (no .ANA needed)"),
    species: str = typer.Option("diaphorina_citri"),
    model: str = typer.Option("random_forest", help="random_forest, xgboost, or rule_based"),
    window_s: float = typer.Option(1.0),
    step_s: Optional[float] = typer.Option(None),
    out: Path = typer.Option(Path("predicted_.ANA")),
):
    """Classify an unlabeled recording and write a Stylet+-compatible
    .ANA prediction file."""
    from epg_tool.export.ana_writer import write_ana_file
    from epg_tool.features import extract_features, make_inference_windows
    from epg_tool.features.baseline import estimate_np_baseline
    from epg_tool.io.session import EPGSession, load_d0x_session
    from epg_tool.models.registry import load_model
    from epg_tool.models.rules import RuleBasedClassifier

    profile = _load_species(species)
    samples, sample_rate_hz, source_files = load_d0x_session(d0x)
    session = EPGSession(
        insect_id=d0x.stem,
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        source_files=source_files,
        segments=[],
        recording_end_s=len(samples) / sample_rate_hz,
    )

    windows = make_inference_windows(samples, sample_rate_hz, window_s=window_s, step_s=step_s)
    context = {"np_baseline_v": estimate_np_baseline(samples, np.zeros(len(samples), dtype=bool))}
    X = pd.DataFrame([extract_features(w.samples, sample_rate_hz, context=context) for w in windows])

    clf = RuleBasedClassifier(profile) if model == "rule_based" else load_model(profile, model)
    pred_codes = clf.predict(X)

    write_ana_file(out, windows, pred_codes, session, profile)
    typer.echo(f"Wrote predictions for {len(windows)} windows to {out}")


@app.command(name="export-parameters")
def export_parameters_cmd(
    data_root: Path = typer.Argument(..., help="Folder to recursively search for .D0x/.ANA pairs"),
    species: str = typer.Option("diaphorina_citri"),
    out: Path = typer.Option(Path("parameters.xlsx")),
):
    """Compute non-sequential + sequential EPG behavioral parameters for
    every recording under DATA_ROOT and export them to an Excel workbook."""
    from epg_tool.export.parameters import export_parameters_excel
    from epg_tool.io.session import build_session
    from epg_tool.training.dataset import discover_recordings

    profile = _load_species(species)
    recordings = discover_recordings(data_root)
    sessions = [
        build_session(r.d0x_paths[0], r.ana_path, insect_id=r.insect_id, sentinel_codes=profile.sentinel_codes)
        for r in recordings
    ]
    export_parameters_excel(sessions, profile, out)
    typer.echo(f"Wrote parameters for {len(sessions)} insect(s) to {out}")


if __name__ == "__main__":
    app()
