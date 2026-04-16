# Brain-State-DashBoard

Brain-State-DashBoard is a FastAPI-based EEG visualization project that trains a traditional machine-learning model and plays back predicted brain-state changes over time in the browser.

The current version is centered on the **first model** and the **Shin2017 dataset family**. The website can train the model, load a session, and animate the EEG timeline with playback controls.

## Current Scope

The first website model is a binary state classifier:
- `Relaxed`
- `Focused`

For this first model, the usable labels come from **Shin2017B**:
- `rest -> relaxed`
- `subtraction -> focused`

This is important because **Shin2017A** is motor imagery (`left_hand` vs `right_hand`), not relaxed vs focused.

## What Is Implemented

- FastAPI app with a dashboard at `/`
- First traditional ML model using:
  - preprocessing
  - frequency-domain bandpower features
  - scikit-learn classifier pipeline
- Training flow for the first model from the browser
- Session playback endpoint for the trained model
- Frontend EEG timeline with:
  - `- / Play / +` playback controls
  - slider scrubbing
  - band toggles
  - sliding chart window
  - predicted state updates during playback

## Tech Stack

- Python
- FastAPI
- Jinja2
- Chart.js
- NumPy / SciPy
- scikit-learn
- MOABB / MNE

## Project Structure

Key files in the current version:

- [main.py](/Users/abishek/dev/Brain-State-DashBoard/main.py): FastAPI routes for model status, training, session listing, playback, and the dashboard page
- [pipeline/eeg_loader.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/eeg_loader.py): dataset loaders, Shin2017 session access, and trial extraction
- [pipeline/preprocess.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/preprocess.py): baseline removal and bandpass filtering for trial-wise EEG
- [pipeline/feature_extractor.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/feature_extractor.py): bandpower features, sliding windows, and frontend band-signal extraction
- [pipeline/state_classifier.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/state_classifier.py): first model training, persistence, and session-level timeline inference
- [pipeline/train_state_classifier.py](/Users/abishek/dev/Brain-State-DashBoard/pipeline/train_state_classifier.py): CLI entrypoint for training the first model
- [templates/brain_state_dashboard.html](/Users/abishek/dev/Brain-State-DashBoard/templates/brain_state_dashboard.html): dashboard layout
- [static/dashboard.js](/Users/abishek/dev/Brain-State-DashBoard/static/dashboard.js): playback controls and EEG chart behavior
- [static/style.css](/Users/abishek/dev/Brain-State-DashBoard/static/style.css): dashboard styling

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

## Training The First Model

You can train the first model in two ways.

From the website:
- Open the dashboard
- Click `Train First Model`

From the CLI:

```bash
venv/bin/python -m pipeline.train_state_classifier
```

Optional CLI arguments:

```bash
venv/bin/python -m pipeline.train_state_classifier --model svm
venv/bin/python -m pipeline.train_state_classifier --model logreg
venv/bin/python -m pipeline.train_state_classifier --subject 1 --subject 2
```

Artifacts are saved under:

```text
artifacts/shin2017_first_model/
```

## Current API

- `GET /api/model/status`
  Returns whether the first model has been trained and, if available, saved metadata.

- `POST /api/model/train?model=svm`
  Trains the first model and returns evaluation metrics.

- `GET /api/sessions?subject=1`
  Lists available Shin2017B sessions for a subject.

- `GET /api/session/playback?subject=1&session=1`
  Returns band signals, timeline windows, summary data, and playback information for the chosen session.

## Frontend Playback

The current EEG timeline view supports:

- `-` to step back one playback window
- `Play` to autoplay the timeline
- `+` to step forward one playback window
- slider scrubbing for direct navigation
- a sliding EEG chart window with fine time ticks
- band toggles for `Theta`, `Alpha`, and `Beta`

The graph is currently designed for readability during playback rather than raw-signal inspection of the full session at once.

## Dataset Notes

The first model uses **Shin2017B** from MOABB.

On first use, MOABB/MNE may download the dataset files to your MNE data directory. Training on the full dataset can be large and may take time.

## Planned Next Steps

- Add the second model
- Add cleaner multi-model selection in the UI
- Improve session browsing and subject/session metadata in the dashboard
- Add dedicated `/about` and `/projects` pages or remove those nav links
- Continue refining the EEG playback visualization

## Notes

This project is for research, visualization, and educational purposes only. It is not a medical tool and does not provide diagnosis or treatment.
