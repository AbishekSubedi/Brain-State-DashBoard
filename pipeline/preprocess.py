"""
EEG preprocessing utilities for trial-wise machine-learning inputs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _validate_trials(trials: np.ndarray) -> np.ndarray:
    arr = np.asarray(trials, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected EEG trials with shape (n_trials, n_channels, n_times), got {arr.shape}."
        )
    return arr


def bandpass_filter_trials(
    trials: np.ndarray,
    sfreq: float,
    low_freq: float = 1.0,
    high_freq: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Bandpass-filter each trial/channel using a zero-phase Butterworth filter."""
    arr = _validate_trials(trials)
    nyquist = 0.5 * sfreq
    if not 0 < low_freq < high_freq < nyquist:
        raise ValueError(
            f"Bandpass range must satisfy 0 < low < high < Nyquist ({nyquist:.2f}), got {(low_freq, high_freq)}."
        )
    sos = butter(order, [low_freq / nyquist, high_freq / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, arr, axis=-1)


def remove_baseline_trials(trials: np.ndarray, baseline_samples: int | None = None) -> np.ndarray:
    """
    Remove a per-trial/channel baseline mean.

    If `baseline_samples` is provided, use the leading samples as the baseline
    window. Otherwise subtract the mean over the whole trial.
    """
    arr = _validate_trials(trials)
    if baseline_samples is None:
        baseline = arr.mean(axis=-1, keepdims=True)
        return arr - baseline
    if baseline_samples <= 0 or baseline_samples > arr.shape[-1]:
        raise ValueError(
            f"baseline_samples must be between 1 and n_times ({arr.shape[-1]}), got {baseline_samples}."
        )
    baseline = arr[..., :baseline_samples].mean(axis=-1, keepdims=True)
    return arr - baseline


def preprocess_trials(
    trials: np.ndarray,
    sfreq: float,
    apply_baseline: bool = True,
    baseline_samples: int | None = None,
    bandpass_range: Iterable[float] = (1.0, 40.0),
    filter_order: int = 4,
) -> np.ndarray:
    """Apply baseline correction then bandpass filtering to trial-wise EEG."""
    arr = _validate_trials(trials)
    if apply_baseline:
        arr = remove_baseline_trials(arr, baseline_samples=baseline_samples)
    low_freq, high_freq = tuple(bandpass_range)
    return bandpass_filter_trials(arr, sfreq=sfreq, low_freq=low_freq, high_freq=high_freq, order=filter_order)
