# EEG pipeline: load data, extract features, classify mental state.
# See docs/DATASETS_AND_PIPELINE.md.

from pipeline.eeg_loader import (
    DEFAULT_DATA_DIR,
    list_emotiv_edf_files,
    get_edf_path,
    load_band_time_series,
    load_band_time_series_for_subject_session,
)
from pipeline.feature_extractor import extract_features
from pipeline.state_classifier import classify, STATES, STATE_KEYS

__all__ = [
    "DEFAULT_DATA_DIR",
    "list_emotiv_edf_files",
    "get_edf_path",
    "load_band_time_series",
    "load_band_time_series_for_subject_session",
    "extract_features",
    "classify",
    "STATES",
    "STATE_KEYS",
]
