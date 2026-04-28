# Brain-State-DashBoard

`Brain-State-DashBoard` is a FastAPI-based EEG research dashboard built around the **Shin2017 dataset family**. It trains traditional machine-learning models, evaluates them, and replays predicted brain states or motor-imagery intent over time inside a browser-based visualization and simulation UI.

The project currently supports **two models**:

- **Model 1: Relaxed vs Focused**
  Uses **Shin2017B** mental-arithmetic data:
  - `rest -> relaxed`
  - `subtraction -> focused`

- **Model 2: Left Hand vs Right Hand**
  Uses **Shin2017A** motor-imagery data:
  - `left_hand`
  - `right_hand`

The second model also drives a **simulation view** that visualizes predicted left/right intent during session playback.

## What Is Implemented

- FastAPI app with a dashboard at `/`
- Two trainable Shin2017-based models
- Session listing and session playback for both tasks
- EEG timeline playback with:
  - `- / Play / +` controls
  - slider scrubbing
  - band toggles
  - sliding time window
- Simulation panel for:
  - calm vs engaged playback for the first model
  - left-vs-right intent animation for the second model
- Poster/research figure generation from the Shin2017 imagery pipeline

## Tech Stack

- Python
- FastAPI
- Jinja2
- Chart.js
- NumPy / SciPy
- scikit-learn
- MOABB / MNE
- Matplotlib / Seaborn

## Project Structure

Core app files:

- [main.py](/Users/abishek/dev/Brain-State-DashBoard/main.py): FastAPI routes for status, training, sessions, playback, and the dashboard page
- [pipeline/eeg_loader.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/eeg_loader.py): Shin2017 session loading and trial extraction
- [pipeline/preprocess.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/preprocess.py): baseline removal and optional bandpass filtering
- [pipeline/feature_extractor.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/feature_extractor.py): bandpower features, sliding windows, and visualization traces
- [pipeline/state_classifier.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/state_classifier.py): model builders, training, persistence, and playback inference
- [pipeline/train_state_classifier.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/train_state_classifier.py): CLI training entrypoint
- [templates/brain_state_dashboard.html](/Users/abishek/dev/Brain-State-DashBoard/templates/brain_state_dashboard.html): dashboard layout
- [static/dashboard.js](/Users/abishek/dev/Brain-State-DashBoard/static/dashboard.js): UI state, playback controls, model switching, and simulation behavior
- [static/style.css](/Users/abishek/dev/Brain-State-DashBoard/static/style.css): dashboard styling

Research/presentation utilities:

- [scripts/generate_poster_figures.py](/Users/abishek/dev/Brain-State-DashBoard/scripts/generate_poster_figures.py): generates poster-ready Shin2017 imagery figures
- [poster_figures](/Users/abishek/dev/Brain-State-DashBoard/poster_figures): generated output figures for research presentations

## Running The App

```bash
cd Brain-State-DashBoard
source venv/bin/activate
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

## Training Models

You can train either model from the dashboard or the CLI.

### From the Website

- Open the dashboard
- Choose the model:
  - `Model 1: Relaxed vs Focused`
  - `Model 2: Left Hand vs Right Hand`
- Click the matching train button

### From the CLI

Train the first model:

```bash
venv/bin/python -m pipeline.train_state_classifier --task state
```

Train the second model:

```bash
venv/bin/python -m pipeline.train_state_classifier --task imagery
```

Optional examples:

```bash
venv/bin/python -m pipeline.train_state_classifier --task state --model svm
venv/bin/python -m pipeline.train_state_classifier --task state --model logreg
venv/bin/python -m pipeline.train_state_classifier --task imagery --model csp_lda
venv/bin/python -m pipeline.train_state_classifier --task imagery --model fbcsp_svm
venv/bin/python -m pipeline.train_state_classifier --task imagery --subject 1 --subject 2
```

Artifacts are saved under:

```text
artifacts/shin2017_first_model/
artifacts/shin2017_second_model/
```

## Current Models

### Model 1: Relaxed vs Focused

- Dataset: **Shin2017B**
- Pipeline:
  - preprocessing
  - theta/alpha/beta bandpower features
  - scikit-learn classifier pipeline
- Default classifier: `svm`

### Model 2: Left Hand vs Right Hand

- Dataset: **Shin2017A**
- Pipeline:
  - crop to `2s-6s`
  - filter bank:
    - `8-12 Hz`
    - `12-20 Hz`
    - `20-30 Hz`
  - Filter-Bank CSP
  - Linear Discriminant Analysis
- Default classifier: `csp_lda`
- Evaluation: **subject-aware**

## Current API

### Model Status

- `GET /api/model/status`
- `GET /api/model/imagery/status`

Returns whether the corresponding model has been trained and, if available, saved metadata.

### Training

- `POST /api/model/train?model=svm`
- `POST /api/model/imagery/train?model=csp_lda`

Returns evaluation metrics, saved artifact location, and metadata.

### Session Listing

- `GET /api/sessions?subject=1&kind=state`
- `GET /api/sessions?subject=1&kind=imagery`

Lists available Shin2017 sessions for the chosen subject and task.

### Playback

- `GET /api/session/playback?subject=1&session=1`
- `GET /api/session/imagery/playback?subject=1&session=1`

Returns:

- band signals
- playback timeline
- collapsed state/intent segments
- summary metadata
- simulation-ready labels

## Frontend Features

The dashboard currently supports:

- switching between the two Shin2017 models
- training either model from the UI
- subject/session selection with session lookup
- EEG playback with:
  - step backward
  - autoplay
  - step forward
  - slider control
- sliding EEG chart window with fine time ticks
- band toggles for `Theta`, `Alpha`, and `Beta`
- lower-panel simulation view

### Simulation Behavior

- **State model:** shifts between calm and engaged visual states
- **Imagery model:** animates left-vs-right intent using a simple avatar scene

## Poster Figure Generation

To regenerate the research figures:

```bash
MPLCONFIGDIR=/tmp/mpl MNE_DONTWRITE_HOME=true venv/bin/python scripts/generate_poster_figures.py
```

To generate only the text-heavy summary figures:

```bash
MPLCONFIGDIR=/tmp/mpl venv/bin/python scripts/generate_poster_figures.py --subset summary
```

Generated figures are written to:

```text
poster_figures/
```

Current outputs:

- `fig01_dataset_overview.png`
- `fig02_subject_distribution.png`
- `fig03_trial_examples.png`
- `fig04_pipeline_overview.png`
- `fig05_confusion_matrix.png`
- `fig06_results_summary.png`
- `fig07_playback_timeline.png`

## Dataset Notes

This project relies on the **Shin2017** dataset family through MOABB/MNE.

- **Shin2017B** is used for the relaxed-vs-focused model
- **Shin2017A** is used for the left-vs-right imagery model

On first use, MOABB/MNE may download dataset files into the MNE data directory. Training on the full dataset can take time.

## Research Story

The original long-term direction was to connect a real EEG device and classify intent in real time. For this phase of the project, the work was grounded on the Shin2017 dataset family to build a defensible baseline:

- validated data source
- traditional ML baselines
- subject-aware evaluation for imagery
- simulation-based playback for interpretability

This makes the project suitable both for research presentation and future extension into real-time EEG systems.

## Notes

This project is for research, visualization, and educational purposes only. It is not a medical tool and does not provide diagnosis or treatment.
