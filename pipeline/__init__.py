# EEG pipeline: load data, extract features, classify mental state.
# See docs/DATASETS_AND_PIPELINE.md.

from pipeline.eeg_loader import (
    DEFAULT_DATA_DIR,
    EEGTrialDataset,
    LABEL_MAPPING,
    load_focus_relax_dataset,
    list_emotiv_edf_files,
    get_edf_path,
    load_band_time_series,
    load_band_time_series_for_subject_session,
)
from pipeline.evaluate import cross_validate_accuracy, evaluate_classifier
from pipeline.feature_extractor import BANDS, compute_bandpower_features, extract_features
from pipeline.preprocess import bandpass_filter_trials, preprocess_trials, remove_baseline_trials
from pipeline.state_classifier import classify, STATES, STATE_KEYS

__all__ = [
    "DEFAULT_DATA_DIR",
    "EEGTrialDataset",
    "LABEL_MAPPING",
    "load_focus_relax_dataset",
    "list_emotiv_edf_files",
    "get_edf_path",
    "load_band_time_series",
    "load_band_time_series_for_subject_session",
    "extract_features",
    "BANDS",
    "compute_bandpower_features",
    "bandpass_filter_trials",
    "remove_baseline_trials",
    "preprocess_trials",
    "classify",
    "STATES",
    "STATE_KEYS",
    "evaluate_classifier",
    "cross_validate_accuracy",
]
