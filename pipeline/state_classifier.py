"""
Model lifecycle and session-level inference for the dashboard models.

Current models:
- first model: Shin2017B mental arithmetic (`rest` -> relaxed, `subtraction` -> focused)
- second model: Shin2017A motor imagery (`left_hand` vs `right_hand`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline.eeg_loader import (
    EEGTrialDataset,
    load_shin2017_focus_relax_trials,
    load_shin2017_left_right_trials,
    load_shin2017_session,
)
from pipeline.evaluate import cross_validate_accuracy, evaluate_classifier
from pipeline.feature_extractor import (
    compute_bandpower_features,
    extract_band_signal_timeseries,
    make_sliding_windows,
)
from pipeline.preprocess import preprocess_trials

RELAXED_LABEL = 0
FOCUSED_LABEL = 1
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1

FIRST_MODEL_LABEL_TO_NAME = {
    RELAXED_LABEL: "Relaxed",
    FOCUSED_LABEL: "Focused",
}
SECOND_MODEL_LABEL_TO_NAME = {
    LEFT_HAND_LABEL: "Left Hand",
    RIGHT_HAND_LABEL: "Right Hand",
}

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"
FIRST_MODEL_DIR = ARTIFACTS_ROOT / "shin2017_first_model"
SECOND_MODEL_DIR = ARTIFACTS_ROOT / "shin2017_second_model"
IMAGERY_FILTER_BANK = ((8.0, 12.0), (12.0, 20.0), (20.0, 30.0))
IMAGERY_TRAINING_WINDOW_SEC = (2.0, 6.0)


def _artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "artifact_dir": artifact_dir,
        "model_path": artifact_dir / "brain_state_classifier.joblib",
        "label_mapping_path": artifact_dir / "label_mapping.json",
        "metadata_path": artifact_dir / "training_metadata.json",
    }


FIRST_MODEL_PATHS = _artifact_paths(FIRST_MODEL_DIR)
SECOND_MODEL_PATHS = _artifact_paths(SECOND_MODEL_DIR)


def build_classifier(model_name: str = "svm", random_state: int = 42) -> Pipeline:
    """Build a scaler + traditional classifier pipeline."""
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


def build_imagery_classifier(model_name: str = "csp_lda", random_state: int = 42) -> Pipeline:
    """Build a motor-imagery classifier suited for left-vs-right hand decoding."""
    model_key = model_name.strip().lower()
    if model_key == "csp_lda":
        classifier = LinearDiscriminantAnalysis()
    elif model_key == "fbcsp_svm":
        classifier = SVC(kernel="linear", C=1.0, probability=True, random_state=random_state)
    else:
        raise ValueError("Unsupported imagery model. Choose from ['csp_lda', 'fbcsp_svm'].")
    return Pipeline(
        steps=[
            (
                "fbcsp",
                FilterBankCSP(
                    bands=IMAGERY_FILTER_BANK,
                    sfreq=200.0,
                    n_components=4,
                ),
            ),
            ("classifier", classifier),
        ]
    )


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP feature extractor for left-vs-right motor imagery."""

    def __init__(
        self,
        bands: tuple[tuple[float, float], ...],
        sfreq: float,
        n_components: int = 4,
        filter_order: int = 4,
    ) -> None:
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.filter_order = filter_order
        self._csps: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterBankCSP":
        from mne.decoding import CSP

        self._csps = []
        for low_freq, high_freq in self.bands:
            filtered = self._bandpass(X, low_freq, high_freq)
            csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
            csp.fit(filtered, y)
            self._csps.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._csps:
            raise ValueError("FilterBankCSP must be fitted before transform().")
        feature_blocks = []
        for (low_freq, high_freq), csp in zip(self.bands, self._csps, strict=True):
            filtered = self._bandpass(X, low_freq, high_freq)
            feature_blocks.append(csp.transform(filtered))
        return np.concatenate(feature_blocks, axis=1)

    def _bandpass(self, X: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        nyquist = 0.5 * self.sfreq
        sos = butter(
            self.filter_order,
            [low_freq / nyquist, high_freq / nyquist],
            btype="bandpass",
            output="sos",
        )
        return sosfiltfilt(sos, X, axis=-1)


def _get_model_status(paths: dict[str, Path]) -> dict[str, Any]:
    if not all(path.exists() for path in [paths["model_path"], paths["metadata_path"], paths["label_mapping_path"]]):
        return {"trained": False}
    metadata = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    label_mapping = json.loads(paths["label_mapping_path"].read_text(encoding="utf-8"))
    return {
        "trained": True,
        "artifact_dir": str(paths["artifact_dir"]),
        "metadata": metadata,
        "label_mapping": label_mapping,
    }


def get_model_status() -> dict[str, Any]:
    """Return the training status and saved metadata for the first model."""
    return _get_model_status(FIRST_MODEL_PATHS)


def get_second_model_status() -> dict[str, Any]:
    """Return the training status and saved metadata for the second model."""
    return _get_model_status(SECOND_MODEL_PATHS)


def _train_model(
    dataset: EEGTrialDataset,
    model_name: str,
    test_size: float,
    random_state: int,
    artifact_paths: dict[str, Path],
    metadata_base: dict[str, Any],
) -> dict[str, Any]:
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

    artifact_paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_paths["model_path"])
    artifact_paths["label_mapping_path"].write_text(json.dumps(dataset.label_mapping, indent=2), encoding="utf-8")

    metadata = {
        **metadata_base,
        "model_name": model_name,
        "sfreq": dataset.sfreq,
        "n_trials": int(dataset.trials.shape[0]),
        "n_channels": int(dataset.trials.shape[1]),
        "n_times": int(dataset.trials.shape[2]),
        "feature_names": feature_names,
        "cross_val_accuracy_mean": float(np.mean(cv_scores)),
        "cross_val_accuracy_std": float(np.std(cv_scores)),
        "test_accuracy": float(metrics["accuracy"]),
        "class_names": target_names,
    }
    artifact_paths["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "metadata": metadata,
        "artifact_dir": str(artifact_paths["artifact_dir"]),
    }

def _crop_trials_time_window(
    trials: np.ndarray,
    sfreq: float,
    start_sec: float,
    stop_sec: float,
) -> np.ndarray:
    start_idx = max(0, int(round(start_sec * sfreq)))
    stop_idx = min(trials.shape[-1], int(round(stop_sec * sfreq)))
    if stop_idx <= start_idx:
        raise ValueError(f"Invalid crop window {(start_sec, stop_sec)} for trial length {trials.shape[-1] / sfreq:.2f}s.")
    return trials[..., start_idx:stop_idx]


def _train_imagery_model(
    dataset: EEGTrialDataset,
    model_name: str,
    test_size: float,
    random_state: int,
    artifact_paths: dict[str, Path],
    metadata_base: dict[str, Any],
    crop_window_sec: tuple[float, float] = IMAGERY_TRAINING_WINDOW_SEC,
) -> dict[str, Any]:
    cropped_trials = _crop_trials_time_window(
        dataset.trials,
        sfreq=dataset.sfreq,
        start_sec=crop_window_sec[0],
        stop_sec=crop_window_sec[1],
    )
    y = dataset.labels.astype(int)
    groups = dataset.groups
    if groups is None:
        raise ValueError("Imagery training requires subject groups.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(cropped_trials, y, groups=groups))
    X_train = cropped_trials[train_index]
    X_test = cropped_trials[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model = build_imagery_classifier(model_name=model_name, random_state=random_state)
    model.named_steps["fbcsp"].sfreq = dataset.sfreq
    model.fit(X_train, y_train)

    target_names = [label for label, _ in sorted(dataset.label_mapping.items(), key=lambda item: item[1])]
    metrics = evaluate_classifier(model, X_test, y_test, target_names=target_names)
    unique_groups = np.unique(groups)
    n_splits = min(5, unique_groups.shape[0])
    if n_splits < 2:
        raise ValueError("Need at least two subject groups for imagery cross-validation.")
    group_kfold = GroupKFold(n_splits=n_splits)
    cv_scores_list = []
    for cv_train_idx, cv_test_idx in group_kfold.split(cropped_trials, y, groups=groups):
        fold_model = build_imagery_classifier(model_name=model_name, random_state=random_state)
        fold_model.named_steps["fbcsp"].sfreq = dataset.sfreq
        fold_model.fit(cropped_trials[cv_train_idx], y[cv_train_idx])
        cv_scores_list.append(float(fold_model.score(cropped_trials[cv_test_idx], y[cv_test_idx])))
    cv_scores = np.asarray(cv_scores_list, dtype=float)

    artifact_paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_paths["model_path"])
    artifact_paths["label_mapping_path"].write_text(json.dumps(dataset.label_mapping, indent=2), encoding="utf-8")

    n_components = model.named_steps["fbcsp"].n_components
    metadata = {
        **metadata_base,
        "model_name": model_name,
        "sfreq": dataset.sfreq,
        "n_trials": int(dataset.trials.shape[0]),
        "n_channels": int(dataset.trials.shape[1]),
        "n_times": int(dataset.trials.shape[2]),
        "training_window_sec": [crop_window_sec[0], crop_window_sec[1]],
        "filter_bank_hz": [list(band) for band in IMAGERY_FILTER_BANK],
        "feature_names": [
            f"band_{band_idx + 1}_csp_component_{component_idx + 1:02d}"
            for band_idx in range(len(model.named_steps["fbcsp"].bands))
            for component_idx in range(n_components)
        ],
        "cross_val_accuracy_mean": float(np.mean(cv_scores)),
        "cross_val_accuracy_std": float(np.std(cv_scores)),
        "test_accuracy": float(metrics["accuracy"]),
        "class_names": target_names,
        "evaluation": "subject-aware",
    }
    artifact_paths["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "metadata": metadata,
        "artifact_dir": str(artifact_paths["artifact_dir"]),
    }


def train_first_model(
    model_name: str = "svm",
    subjects: list[int] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and persist the first state model on Shin2017B."""
    dataset = load_shin2017_focus_relax_trials(subjects=subjects)
    return _train_model(
        dataset=dataset,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        artifact_paths=FIRST_MODEL_PATHS,
        metadata_base={
            "dataset": "Shin2017B",
            "task": "state",
            "state_mapping": {"rest": "relaxed", "subtraction": "focused"},
            "subjects": subjects or "all",
        },
    )


def train_second_model(
    model_name: str = "csp_lda",
    subjects: list[int] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and persist the second imagery model on Shin2017A."""
    dataset = load_shin2017_left_right_trials(subjects=subjects)
    return _train_imagery_model(
        dataset=dataset,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        artifact_paths=SECOND_MODEL_PATHS,
        metadata_base={
            "dataset": "Shin2017A",
            "task": "imagery",
            "movement_mapping": {"left_hand": "left_hand", "right_hand": "right_hand"},
            "subjects": subjects or "all",
        },
    )


def _load_trained_model(paths: dict[str, Path], missing_message: str) -> Pipeline:
    if not paths["model_path"].exists():
        raise FileNotFoundError(missing_message)
    model = joblib.load(paths["model_path"])
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline at {paths['model_path']}, got {type(model)}.")
    return model


def load_trained_model() -> Pipeline:
    """Load the persisted first model."""
    return _load_trained_model(
        FIRST_MODEL_PATHS,
        "The first model has not been trained yet. Train it from the dashboard or run the training script first.",
    )


def load_second_trained_model() -> Pipeline:
    """Load the persisted second model."""
    return _load_trained_model(
        SECOND_MODEL_PATHS,
        "The second model has not been trained yet. Train it from the backend or CLI first.",
    )


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


def _summarize_named_timeline(timeline: list[dict[str, Any]], label_to_name: dict[int, str]) -> dict[str, Any]:
    if not timeline:
        return {"dominant_label": "Unknown", "avg_confidence": 0.0, "fractions": {}}
    label_ids = np.asarray([item["label_id"] for item in timeline], dtype=int)
    confidences = np.asarray([item["confidence"] for item in timeline], dtype=float)
    fractions = {
        name: round(float(np.mean(label_ids == label_id)), 4)
        for label_id, name in label_to_name.items()
    }
    dominant_id = max(label_to_name, key=lambda label_id: fractions[label_to_name[label_id]])
    return {
        "dominant_label": label_to_name[dominant_id],
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "fractions": fractions,
    }


def _predict_session_timeline(
    model: Pipeline,
    subject: int,
    session: int,
    kind: str,
    label_to_name: dict[int, str],
    explanation: str,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    visualization_sfreq: float = 50.0,
) -> dict[str, Any]:
    session_data = load_shin2017_session(subject=subject, session=session, kind=kind)
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
                "label": label_to_name[int(pred)],
                "confidence": round(float(confidence), 4),
            }
        )

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
        "segments": _collapse_timeline_segments(timeline),
        "summary": _summarize_named_timeline(timeline, label_to_name),
        "window_sec": window_sec,
        "step_sec": step_sec,
        "explanation": explanation,
    }


def predict_session_timeline(
    subject: int,
    session: int,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    visualization_sfreq: float = 50.0,
) -> dict[str, Any]:
    """Score a Shin2017B session with the trained first model and return playback data."""
    return _predict_session_timeline(
        model=load_trained_model(),
        subject=subject,
        session=session,
        kind="state",
        label_to_name=FIRST_MODEL_LABEL_TO_NAME,
        explanation="This first model treats Shin2017 mental arithmetic windows as focused and rest windows as relaxed.",
        window_sec=window_sec,
        step_sec=step_sec,
        visualization_sfreq=visualization_sfreq,
    )


def predict_imagery_session_timeline(
    subject: int,
    session: int,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    visualization_sfreq: float = 50.0,
) -> dict[str, Any]:
    """Score a Shin2017A session with the trained second imagery model."""
    model = load_second_trained_model()
    session_data = load_shin2017_session(subject=subject, session=session, kind="imagery")
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
        apply_baseline=False,
        bandpass_range=None,
    )
    predictions, confidences = _predict_with_confidence(model, processed_windows)

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
                "label": SECOND_MODEL_LABEL_TO_NAME[int(pred)],
                "confidence": round(float(confidence), 4),
            }
        )

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
        "segments": _collapse_timeline_segments(timeline),
        "summary": _summarize_named_timeline(timeline, SECOND_MODEL_LABEL_TO_NAME),
        "window_sec": window_sec,
        "step_sec": step_sec,
        "explanation": "This second model uses a filter-bank CSP pipeline over motor-imagery bands for left-vs-right hand decoding.",
    }
