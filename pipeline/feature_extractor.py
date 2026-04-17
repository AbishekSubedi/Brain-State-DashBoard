"""Feature extraction utilities for the current EEG classifiers and playback UI."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _validate_trial_array(trials: np.ndarray) -> np.ndarray:
    arr = np.asarray(trials, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected EEG trials with shape (n_trials, n_channels, n_times), got {arr.shape}."
        )
    return arr


def compute_bandpower_features(
    trials: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    nperseg: int | None = None,
    log_power: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute channel-wise bandpower features for trial-wise EEG.

    Returns:
    - feature matrix with shape (n_trials, n_channels * n_bands)
    - feature names aligned with the matrix columns
    """
    arr = _validate_trial_array(trials)
    selected_bands = bands or BANDS
    _, _, n_times = arr.shape
    if nperseg is None:
        nperseg = min(256, n_times)
    nperseg = min(max(8, nperseg), n_times)

    freqs, psd = welch(arr, fs=sfreq, axis=-1, nperseg=nperseg)
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    for band_name, (low_freq, high_freq) in selected_bands.items():
        mask = (freqs >= low_freq) & (freqs < high_freq)
        if not np.any(mask):
            raise ValueError(f"No Welch frequency bins found for band {band_name} in range {(low_freq, high_freq)}.")
        band_power = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
        if log_power:
            band_power = np.log10(band_power + 1e-12)
        feature_blocks.append(band_power)
        feature_names.extend([f"{band_name}_ch{channel_idx:02d}" for channel_idx in range(arr.shape[1])])

    features = np.concatenate(feature_blocks, axis=1)
    return features, feature_names


def make_sliding_windows(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice continuous EEG into trial-like windows for state timeline inference.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected continuous EEG with shape (n_channels, n_times), got {arr.shape}.")
    window_samples = max(1, int(round(window_sec * sfreq)))
    step_samples = max(1, int(round(step_sec * sfreq)))
    if window_samples > arr.shape[1]:
        raise ValueError("Window size is larger than the available signal length.")

    windows = []
    starts = []
    ends = []
    for start in range(0, arr.shape[1] - window_samples + 1, step_samples):
        stop = start + window_samples
        windows.append(arr[:, start:stop])
        starts.append(start / sfreq)
        ends.append(stop / sfreq)
    return np.stack(windows, axis=0), np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def extract_band_signal_timeseries(
    data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    target_sfreq: float | None = 50.0,
) -> dict[str, object]:
    """
    Create frontend-ready band-filtered mean signals for visualization.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected EEG array with shape (n_channels, n_times), got {arr.shape}.")

    selected_bands = bands or BANDS
    mean_signal = arr.mean(axis=0)
    time = np.arange(arr.shape[1], dtype=float) / sfreq
    output: dict[str, Any] = {}

    for band_name, (low_freq, high_freq) in selected_bands.items():
        sos = butter(4, [low_freq / (0.5 * sfreq), high_freq / (0.5 * sfreq)], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, mean_signal)
        filtered = filtered / (np.std(filtered) + 1e-9)
        output[band_name] = filtered

    out_sfreq = sfreq
    if target_sfreq is not None and sfreq > target_sfreq:
        step = max(1, int(round(sfreq / target_sfreq)))
        time = time[::step]
        for band_name in selected_bands:
            output[band_name] = output[band_name][::step]
        out_sfreq = sfreq / step

    return {
        "sfreq": float(out_sfreq),
        "time": np.round(time, 3).tolist(),
        **{band_name: np.round(output[band_name], 6).tolist() for band_name in selected_bands},
    }
