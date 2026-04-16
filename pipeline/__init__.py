# EEG pipeline: load data, extract features, classify mental state.
# See docs/DATASETS_AND_PIPELINE.md.

from pipeline.eeg_loader import (
    DEFAULT_DATA_DIR,
    EEGTrialDataset,
    LABEL_MAPPING,
    load_focus_relax_dataset,
    load_shin2017_focus_relax_trials,
    load_shin2017_session,
    list_shin2017_sessions,
    list_emotiv_edf_files,
    get_edf_path,
    load_band_time_series,
    load_band_time_series_for_subject_session,
)
from pipeline.evaluate import cross_validate_accuracy, evaluate_classifier
from pipeline.feature_extractor import (
    BANDS,
    compute_bandpower_features,
    extract_band_signal_timeseries,
    extract_features,
    make_sliding_windows,
)
from pipeline.preprocess import bandpass_filter_trials, preprocess_trials, remove_baseline_trials
from pipeline.state_classifier import build_classifier, get_model_status, load_trained_model, predict_session_timeline, train_first_model

__all__ = [
    "DEFAULT_DATA_DIR",
    "EEGTrialDataset",
    "LABEL_MAPPING",
    "load_focus_relax_dataset",
    "load_shin2017_focus_relax_trials",
    "load_shin2017_session",
    "list_shin2017_sessions",
    "list_emotiv_edf_files",
    "get_edf_path",
    "load_band_time_series",
    "load_band_time_series_for_subject_session",
    "extract_features",
    "BANDS",
    "compute_bandpower_features",
    "make_sliding_windows",
    "extract_band_signal_timeseries",
    "bandpass_filter_trials",
    "remove_baseline_trials",
    "preprocess_trials",
    "build_classifier",
    "get_model_status",
    "load_trained_model",
    "predict_session_timeline",
    "train_first_model",
    "evaluate_classifier",
    "cross_validate_accuracy",
]
