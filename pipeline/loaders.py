"""
EEG data loaders. Each loader returns a common format so the rest of the pipeline
and the API can stay dataset-agnostic.

Common return format for a single segment:
  - data: np.ndarray shape (n_channels, n_samples)
  - sfreq: float (Hz)
  - label: str | None, e.g. "relaxation", "concentration", "stress"
  - channel_names: list[str] | None

For full-dataset loading (for training), return list of segments or (X, y) with
X = band-power features and y = labels.
"""

import os
import numpy as np
from pathlib import Path
from typing import Any

try:
    import mne
except ImportError:
    mne = None  # type: ignore

# When you add another dataset, implement a loader and return the same structure.
# See docs/DATASETS_AND_PIPELINE.md.


# Default path to your Emotiv sample data (S001/S001E01.edf etc.)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Emotiv Sample Data"


def list_emotiv_edf_files(data_dir: Path | str | None = None) -> list[tuple[str, str, Path]]:
    """
    List available EDF files. Returns list of (subject_id, session_id, path).
    Example: [("S001", "E01", Path(".../S001/S001E01.edf")), ...]
    """
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    if not data_dir.is_dir():
        return []
    out = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("S"):
            continue
        subj_id = subj_dir.name
        for edf in sorted(subj_dir.glob("*.edf")) + sorted(subj_dir.glob("*.EDF")):
            # e.g. S001E01.edf -> session E01
            stem = edf.stem
            if stem.startswith(subj_id) and len(stem) > len(subj_id):
                sess = stem[len(subj_id) :]
                out.append((subj_id, sess, edf))
    return out


def load_emotiv_edf(
    edf_path: Path | str,
    preload: bool = True,
    pick_eeg: bool = True,
) -> dict[str, Any]:
    """
    Load one EDF file with MNE. Returns common segment format.
    """
    if mne is None:
        raise ImportError("Install mne: pip install mne")
    raw = mne.io.read_raw_edf(str(edf_path), preload=preload, verbose=False)
    if pick_eeg:
        try:
            raw.pick_types(eeg=True)
        except Exception:
            pass  # keep all if no standard types
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    times = raw.times.copy()
    return {
        "data": data,
        "sfreq": sfreq,
        "label": None,
        "channel_names": raw.ch_names,
        "time": times,
        "raw": raw,
    }


def emotiv_edf_to_band_time_series(
    edf_path: Path | str,
    max_duration_sec: float | None = 60.0,
    target_sfreq: int | None = 128,
) -> dict[str, Any]:
    """
    Load EDF, bandpass filter in delta/theta/alpha/beta/gamma, average over channels,
    and return the same shape as the dashboard API: time, alpha, beta, gamma, delta
    (and optionally theta). Optional downsampling to target_sfreq.
    """
    if mne is None:
        raise ImportError("Install mne: pip install mne")
    seg = load_emotiv_edf(edf_path, preload=True, pick_eeg=True)
    raw = seg["raw"]
    sfreq = seg["sfreq"]
    # Optional: crop to first N seconds to keep response small
    if max_duration_sec is not None:
        raw = raw.copy().crop(tmin=0, tmax=min(max_duration_sec, raw.times[-1]))
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    band_series = {}
    for name, (lo, hi) in bands.items():
        raw_band = raw.copy().filter(l_freq=lo, h_freq=hi, verbose=False)
        # (n_channels, n_times) -> mean over channels
        band_series[name] = raw_band.get_data().mean(axis=0)
    time = raw.times.copy()
    # Downsample if requested
    if target_sfreq is not None and sfreq > target_sfreq:
        step = int(sfreq / target_sfreq)
        time = time[::step]
        for k in band_series:
            band_series[k] = band_series[k][::step]
    return {
        "time": time.tolist(),
        "delta": band_series["delta"].tolist(),
        "theta": band_series["theta"].tolist(),
        "alpha": band_series["alpha"].tolist(),
        "beta": band_series["beta"].tolist(),
        "gamma": band_series["gamma"].tolist(),
    }


def _theta_signal(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.sin(2 * np.pi * 6 * t) * 0.45 + rng.normal(0, 0.05, t.shape[0])


def load_synthetic_segment(seconds: float = 10.0, sfreq: int = 128, seed: int = 42) -> dict[str, Any]:
    """
    Load a single segment of synthetic EEG-like data in the common format.
    Used by the dashboard API until real data is wired.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    t = np.linspace(0, seconds, n)
    delta = np.sin(2 * np.pi * 2 * t) * 0.6 + rng.normal(0, 0.06, n)
    theta = _theta_signal(t, rng)
    alpha = np.sin(2 * np.pi * 10 * t) * 0.5 + rng.normal(0, 0.05, n)
    beta = np.sin(2 * np.pi * 20 * t) * 0.4 + rng.normal(0, 0.05, n)
    gamma = np.sin(2 * np.pi * 38 * t) * 0.3 + rng.normal(0, 0.04, n)
    data = np.stack([delta, theta, alpha, beta, gamma], axis=0)
    return {
        "data": data,
        "sfreq": float(sfreq),
        "label": None,
        "channel_names": ["delta", "theta", "alpha", "beta", "gamma"],
        "time": t,
        "bands": {"delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma},
    }
