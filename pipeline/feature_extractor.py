"""
Extract EEG band features from band time series (mean power, relative power, ratios).
Input: dict with keys delta, theta, alpha, beta, gamma (lists or arrays).
Output: dict of scalar features for the rule-based (or future ML) classifier.
"""

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _to_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


def _band_power(band_series: np.ndarray) -> float:
    """Mean power (mean of squares) as a proxy for band power; stable across segments."""
    return float(np.mean(band_series ** 2))


def extract_features(band_series: dict[str, Any]) -> dict[str, float]:
    """
    Compute mean power per band, relative power, and ratios.
    band_series: dict with keys delta, theta, alpha, beta, gamma (each list or 1d array).
    Missing bands are treated as zero (handled gracefully).
    """
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    eps = 1e-12

    mean_powers = {}
    for b in bands:
        arr = band_series.get(b)
        if arr is None or (isinstance(arr, (list, np.ndarray)) and len(arr) == 0):
            mean_powers[b] = 0.0
        else:
            mean_powers[b] = _band_power(_to_array(arr))

    total = sum(mean_powers.values()) + eps
    relative = {f"relative_{b}": mean_powers[b] / total for b in bands}

    alpha_m = mean_powers.get("alpha", 0.0) + eps
    beta_m = mean_powers.get("beta", 0.0) + eps
    theta_m = mean_powers.get("theta", 0.0) + eps
    gamma_m = mean_powers.get("gamma", 0.0) + eps
    delta_m = mean_powers.get("delta", 0.0) + eps

    beta_alpha = beta_m / alpha_m
    theta_beta = theta_m / beta_m
    alpha_beta_gamma = alpha_m / (beta_m + gamma_m + eps)

    out = {
        "delta_mean": mean_powers.get("delta", 0.0),
        "theta_mean": mean_powers.get("theta", 0.0),
        "alpha_mean": mean_powers.get("alpha", 0.0),
        "beta_mean": mean_powers.get("beta", 0.0),
        "gamma_mean": mean_powers.get("gamma", 0.0),
        "relative_alpha": relative["relative_alpha"],
        "relative_beta": relative["relative_beta"],
        "relative_theta": relative["relative_theta"],
        "relative_delta": relative["relative_delta"],
        "relative_gamma": relative["relative_gamma"],
        "beta_alpha_ratio": beta_alpha,
        "theta_beta_ratio": theta_beta,
        "alpha_beta_gamma_ratio": alpha_beta_gamma,
    }
    return out


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
) -> dict[str, Any]:
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
