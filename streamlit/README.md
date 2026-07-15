# EPG Waveform Classifier

Automated classification of insect probing/feeding behavior from
Electrical Penetration Graph (EPG-DC / Stylet+) recordings. Parses raw
`.D0x` traces and `.ANA` ground-truth annotations, extracts features,
classifies waveforms (rule-based or trained ML), and exports
Stylet+-compatible predictions plus Backus/Sarria-style behavioral
parameters.

The first populated species profile is *Diaphorina citri* (Asian citrus
psyllid) per Bonani et al. (2010, *Entomologia Experimentalis et
Applicata* 134:35-49) — a worked example, not a hardcoded assumption.
Every waveform label, threshold, and trained-model path lives in a
per-species YAML config (`src/epg_tool/species/profiles/`), so adding a
new insect (aphid, whitefly, leafhopper, ...) is a new config file, not
new code.

## Repo layout

```
src/epg_tool/
  io/            .D0x / .ANA parsers, multi-file session stitching
  species/       species profile schema + profiles/*.yaml
  features/      pluggable feature extraction + fixed-length windowing
  models/        rule-based classifier, tabular ML (RF/XGBoost), model registry
  training/      dataset discovery, group-based train/val/test split, metrics
  export/        .ANA writer, QC plotting, EPG behavioral parameters
  cli.py         typer CLI
streamlit_app.py Streamlit dashboard (upload -> classify -> review -> download)
tests/           pytest suite, including integration tests against real recordings
```

## Setup

```bash
conda create -n epg python=3.11 -y
conda activate epg
pip install -e .[dev]
pytest
```

**macOS + XGBoost:** the pip wheel needs an OpenMP runtime it doesn't
bundle. If you hit `Library not loaded: @rpath/libomp.dylib`, either
`conda install -c conda-forge libomp` (if your channel has it) or copy
an existing `libomp.dylib` (e.g. from a package that vendors one, such
as `sklearn/.dylibs/libomp.dylib`) into your env's `lib/` directory.
Not an issue on Streamlit Community Cloud (Linux).

## Species profiles

A profile is one YAML file: waveform codes/labels/colors, rule-based
thresholds (generalizing a Table-1-style expert chart), a model
registry (paths to trained artifacts), and which behavioral parameters
to compute. See `src/epg_tool/species/profiles/diaphorina_citri.yaml`
for the reference instance. To add a species: write a new YAML file
with that species' waveform vocabulary and thresholds — nothing in
`io/`, `features/`, `models/`, `cli.py`, or `streamlit_app.py` needs to
change.

Numeric waveform codes follow the Stylet+ mark convention exported in
your `.ANA` files; cross-check them against your annotation software's
legend (often an accompanying analysis worksheet) before writing a new
profile, the way `diaphorina_citri.yaml`'s header documents how codes
1/2/3/4/5/7 were resolved to Np/C/D/E1/E2/G for this dataset.

## CLI

```bash
epg-tool inspect  <d0x> <ana> --species diaphorina_citri     # print labeled segments
epg-tool plot     <d0x> <ana> --species diaphorina_citri --out qc.png
epg-tool train    <data_root> --species diaphorina_citri --model random_forest
epg-tool evaluate <data_root> --species diaphorina_citri --model random_forest
epg-tool predict  <d0x>       --species diaphorina_citri --model random_forest --out predicted_.ANA
epg-tool export-parameters <data_root> --species diaphorina_citri --out parameters.xlsx
```

`<data_root>` is searched recursively for `.ANA` files, each matched to
its `.D0x` series by shared filename stem (handles both "everything in
one folder" and the Stylet+ convention of annotations living in a
`DataANA/` subfolder next to the raw files).

**Group-based splitting:** `train` and dataset-building split by insect
individual, never randomly by window — adjacent windows of the same
probe are highly correlated, so a random split would leak. This needs
**at least 3 labeled insects** to guarantee no leakage in every split;
with fewer (today: 1), it falls back to a chronological split and warns
loudly. Treat those numbers as a pipeline sanity check, not a real
generalization estimate, until more insects are added.

## Streamlit app

```bash
streamlit run streamlit_app.py
```

Upload a recording's `.D0x` files (+ optionally its `.ANA` ground
truth), pick a species profile and classifier, and get back: a QC plot,
predicted segments, accuracy metrics (if ground truth was supplied),
behavioral parameters, and downloads (predicted `.ANA`, parameters
Excel). Training itself is a CLI job, not something the app does live —
it's a batch operation over many recordings, a poor fit for a
request/response UI.

### Deploying to Streamlit Community Cloud

1. `git init`, commit, push to a GitHub repo. Note `.gitignore` excludes
   the attached paper PDF (copyrighted) and raw recordings under
   `Arquivo Ondas EPG Diaphorina citri/` / `data/` (they'll only grow —
   don't let them live in git history; the app takes recordings via
   upload, not from a bundled folder).
2. On [share.streamlit.io](https://share.streamlit.io), point at the
   repo and `streamlit_app.py`. `requirements.txt` installs the package
   itself via `pip install -e .[app]`.
3. Trained model artifacts under `models/<species>/*.joblib` **are**
   committed (a few MB each) so the deployed app has something beyond
   the rule-based baseline out of the box. Retrain and commit again as
   more labeled data comes in; move to Git LFS if they grow large.

## Known limitations / next steps

- **Single-recording baseline today.** Metrics reported by `train` on
  this dataset are a chronological-split sanity check, not a real
  estimate — several classes (E1, G) are rare enough that they can end
  up entirely on one side of the split. Both the group-split and the
  warning are already in place; they'll kick in automatically once
  3+ insects are available.
- **Rule-based %amplitude calibration.** The Bonani et al. Table 1
  convention assumes voltage gained to within the 5V full-scale range;
  this dataset's raw `.D0x` values run much smaller (~0.1-0.3V), so the
  rule-based classifier's amplitude thresholds may need retuning for
  your acquisition gain settings — that's a YAML edit
  (`rule_based_thresholds` in the species profile), not a code change.
- **CNN upgrade path stubbed, not implemented** (`epg_tool.models.sequence`)
  behind the optional `torch` extra, for once there's enough data to
  benefit from a sequence model over hand-engineered features.
