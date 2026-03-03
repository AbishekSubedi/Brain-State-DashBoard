"""
EEG band power and PSD utilities.
Uses scipy so it works with any (channels × time) array; no MNE required.
For EDF files, use MNE in loaders and then pass raw.get_data() here.
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

# Standard EEG bands (Hz)
DEFAULT_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def band_power_psd(data: np.ndarray, sfreq: float, bands: dict | None = None) -> dict[str, float]:
    """
    Compute average power in each frequency band from (n_channels, n_samples) data.
    Uses Welch PSD, then integrates power in each band and averages over channels.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    n_channels, n_samples = data.shape
    freqs, psd = welch(data, fs=sfreq, nperseg=min(256, n_samples // 4))
    result = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            result[name] = 0.0
            continue
        # (n_channels, n_freqs_in_band) -> integrate then mean over channels
        power_per_ch = simpson(psd[:, mask], freqs[mask], axis=1)
        result[name] = float(np.mean(power_per_ch))
    return result


def band_power_time_series(
    data: np.ndarray, sfreq: float, window_sec: float = 1.0, bands: dict | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Sliding-window band power so you get a time series per band (for graphing).
    Returns (time_axis, dict of band_name -> 1d array).
    """
    if bands is None:
        bands = DEFAULT_BANDS
    _, n_samples = data.shape
    step = max(1, int(sfreq * window_sec))
    n_windows = max(1, (n_samples - step) // step + 1)
    time_axis = np.arange(n_windows) * (step / sfreq)
    out = {name: np.zeros(n_windows) for name in bands}
    for i in range(n_windows):
        start = i * step
        end = min(start + step, n_samples)
        segment = data[:, start:end]
        if segment.size == 0:
            continue
        pw = band_power_psd(segment, sfreq, bands)
        for name, val in pw.items():
            out[name][i] = val
    return time_axis, out
