"""Public pipeline API for the current Shin2017-based dashboard."""

from pipeline.eeg_loader import (
    EEGTrialDataset,
    LABEL_MAPPING,
    load_shin2017_focus_relax_trials,
    load_shin2017_left_right_trials,
    load_shin2017_session,
    list_shin2017_sessions,
)
from pipeline.evaluate import cross_validate_accuracy, evaluate_classifier
from pipeline.feature_extractor import (
    BANDS,
    compute_bandpower_features,
    extract_band_signal_timeseries,
    make_sliding_windows,
)
from pipeline.preprocess import bandpass_filter_trials, preprocess_trials, remove_baseline_trials
from pipeline.state_classifier import (
    build_classifier,
    get_model_status,
    get_second_model_status,
    load_second_trained_model,
    load_trained_model,
    predict_imagery_session_timeline,
    predict_session_timeline,
    train_first_model,
    train_second_model,
)

__all__ = [
    "EEGTrialDataset",
    "LABEL_MAPPING",
    "load_shin2017_focus_relax_trials",
    "load_shin2017_left_right_trials",
    "load_shin2017_session",
    "list_shin2017_sessions",
    "BANDS",
    "compute_bandpower_features",
    "make_sliding_windows",
    "extract_band_signal_timeseries",
    "bandpass_filter_trials",
    "remove_baseline_trials",
    "preprocess_trials",
    "build_classifier",
    "get_model_status",
    "get_second_model_status",
    "load_second_trained_model",
    "load_trained_model",
    "predict_imagery_session_timeline",
    "predict_session_timeline",
    "train_first_model",
    "train_second_model",
    "evaluate_classifier",
    "cross_validate_accuracy",
]
