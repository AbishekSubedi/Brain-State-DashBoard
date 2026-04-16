"""
Model lifecycle and session-level inference for the first brain-state classifier.

The first website model uses the Shin2017 study's mental-arithmetic split:
- `rest` -> relaxed
- `subtraction` -> focused

Shin2017A is motor imagery, so it is not used for this relaxed-vs-focused
state model. We keep the naming generic so a second model can be added later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline.eeg_loader import EEGTrialDataset, load_shin2017_focus_relax_trials, load_shin2017_session
from pipeline.evaluate import cross_validate_accuracy, evaluate_classifier
from pipeline.feature_extractor import compute_bandpower_features, extract_band_signal_timeseries, make_sliding_windows
from pipeline.preprocess import preprocess_trials

RELAXED_LABEL = 0
FOCUSED_LABEL = 1
LABEL_TO_NAME = {
    RELAXED_LABEL: "Relaxed",
    FOCUSED_LABEL: "Focused",
}
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "shin2017_first_model"
MODEL_PATH = ARTIFACT_DIR / "brain_state_classifier.joblib"
LABEL_MAPPING_PATH = ARTIFACT_DIR / "label_mapping.json"
METADATA_PATH = ARTIFACT_DIR / "training_metadata.json"


def build_classifier(model_name: str = "svm", random_state: int = 42) -> Pipeline:
    """Build the first model as a scaler + traditional classifier pipeline."""
    model_key = model_name.strip().lower()
    estimators: dict[str, Any] = {
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
        "logreg": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if model_key not in estimators:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {sorted(estimators)}.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimators[model_key]),
        ]
    )


def get_model_status() -> dict[str, Any]:
    """Return the training status and saved metadata for the first model."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists() or not LABEL_MAPPING_PATH.exists():
        return {"trained": False}
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    label_mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    return {
        "trained": True,
        "artifact_dir": str(ARTIFACT_DIR),
        "metadata": metadata,
        "label_mapping": label_mapping,
    }


def train_first_model(
    model_name: str = "svm",
    subjects: list[int] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and persist the first state model on Shin2017B."""
    dataset = load_shin2017_focus_relax_trials(subjects=subjects)
    processed_trials = preprocess_trials(
        dataset.trials,
        sfreq=dataset.sfreq,
        apply_baseline=True,
        baseline_samples=int(0.5 * dataset.sfreq),
    )
    X, feature_names = compute_bandpower_features(processed_trials, sfreq=dataset.sfreq)
    y = dataset.labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    model = build_classifier(model_name=model_name, random_state=random_state)
    model.fit(X_train, y_train)

    target_names = [label for label, _ in sorted(dataset.label_mapping.items(), key=lambda item: item[1])]
    metrics = evaluate_classifier(model, X_test, y_test, target_names=target_names)
    cv_scores = cross_validate_accuracy(model, X, y, cv=5)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    LABEL_MAPPING_PATH.write_text(json.dumps(dataset.label_mapping, indent=2), encoding="utf-8")

    metadata = {
        "dataset": "Shin2017B",
        "state_mapping": {"rest": "relaxed", "subtraction": "focused"},
        "model_name": model_name,
        "sfreq": dataset.sfreq,
        "n_trials": int(dataset.trials.shape[0]),
        "n_channels": int(dataset.trials.shape[1]),
        "n_times": int(dataset.trials.shape[2]),
        "subjects": subjects or "all",
        "feature_names": feature_names,
        "cross_val_accuracy_mean": float(np.mean(cv_scores)),
        "cross_val_accuracy_std": float(np.std(cv_scores)),
        "test_accuracy": float(metrics["accuracy"]),
        "class_names": target_names,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "metadata": metadata,
        "artifact_dir": str(ARTIFACT_DIR),
    }


def load_trained_model() -> Pipeline:
    """Load the persisted first model or raise a clear error if it is missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The first model has not been trained yet. Train it from the dashboard or run the training script first."
        )
    model = joblib.load(MODEL_PATH)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline at {MODEL_PATH}, got {type(model)}.")
    return model


def _predict_with_confidence(model: Pipeline, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(features).astype(int)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        confidences = probabilities[np.arange(len(predictions)), predictions]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if scores.ndim == 1:
            confidences = 1.0 / (1.0 + np.exp(-np.abs(scores)))
        else:
            confidences = np.max(scores, axis=1)
    else:
        confidences = np.ones(len(predictions), dtype=float)
    return predictions, confidences.astype(float)


def predict_session_timeline(
    subject: int,
    session: int,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    visualization_sfreq: float = 50.0,
) -> dict[str, Any]:
    """
    Score a Shin2017B session with the trained first model and return playback data.
    """
    model = load_trained_model()
    session_data = load_shin2017_session(subject=subject, session=session)
    raw_data = session_data["data"]
    sfreq = float(session_data["sfreq"])

    windows, start_times, end_times = make_sliding_windows(
        raw_data,
        sfreq=sfreq,
        window_sec=window_sec,
        step_sec=step_sec,
    )
    processed_windows = preprocess_trials(
        windows,
        sfreq=sfreq,
        apply_baseline=True,
        baseline_samples=int(0.5 * sfreq),
    )
    features, _ = compute_bandpower_features(processed_windows, sfreq=sfreq)
    predictions, confidences = _predict_with_confidence(model, features)

    timeline = []
    for index, (start, end, pred, confidence) in enumerate(
        zip(start_times, end_times, predictions, confidences, strict=True)
    ):
        midpoint = (start + end) / 2.0
        timeline.append(
            {
                "index": index,
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "midpoint": round(float(midpoint), 3),
                "label_id": int(pred),
                "label": LABEL_TO_NAME[int(pred)],
                "confidence": round(float(confidence), 4),
            }
        )

    segments = _collapse_timeline_segments(timeline)
    summary = _summarize_timeline(timeline)
    band_signals = extract_band_signal_timeseries(raw_data, sfreq=sfreq, target_sfreq=visualization_sfreq)

    return {
        "subject": subject,
        "session": session,
        "session_name": session_data["session_name"],
        "channel_names": session_data["channel_names"],
        "sampling_rate": band_signals["sfreq"],
        "time": band_signals["time"],
        "theta": band_signals["theta"],
        "alpha": band_signals["alpha"],
        "beta": band_signals["beta"],
        "timeline": timeline,
        "segments": segments,
        "summary": summary,
        "window_sec": window_sec,
        "step_sec": step_sec,
        "explanation": (
            "This first model treats Shin2017 mental arithmetic windows as focused and rest windows as relaxed."
        ),
    }


def _collapse_timeline_segments(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not timeline:
        return []
    segments = [
        {
            "label": timeline[0]["label"],
            "label_id": timeline[0]["label_id"],
            "start": timeline[0]["start"],
            "end": timeline[0]["end"],
            "confidence_values": [timeline[0]["confidence"]],
        }
    ]
    for item in timeline[1:]:
        current = segments[-1]
        if item["label_id"] == current["label_id"]:
            current["end"] = item["end"]
            current["confidence_values"].append(item["confidence"])
            continue
        segments.append(
            {
                "label": item["label"],
                "label_id": item["label_id"],
                "start": item["start"],
                "end": item["end"],
                "confidence_values": [item["confidence"]],
            }
        )
    for segment in segments:
        segment["confidence"] = round(float(np.mean(segment.pop("confidence_values"))), 4)
    return segments


def _summarize_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    if not timeline:
        return {
            "dominant_state": "Unknown",
            "focused_fraction": 0.0,
            "relaxed_fraction": 0.0,
            "avg_confidence": 0.0,
        }
    label_ids = np.asarray([item["label_id"] for item in timeline], dtype=int)
    confidences = np.asarray([item["confidence"] for item in timeline], dtype=float)
    focused_fraction = float(np.mean(label_ids == FOCUSED_LABEL))
    relaxed_fraction = float(np.mean(label_ids == RELAXED_LABEL))
    dominant_id = FOCUSED_LABEL if focused_fraction >= relaxed_fraction else RELAXED_LABEL
    return {
        "dominant_state": LABEL_TO_NAME[dominant_id],
        "focused_fraction": round(focused_fraction, 4),
        "relaxed_fraction": round(relaxed_fraction, 4),
        "avg_confidence": round(float(np.mean(confidences)), 4),
    }

