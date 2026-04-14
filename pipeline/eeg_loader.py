"""
Load EEG data for both the dashboard and the ML training pipeline.

This module keeps the existing EDF helpers used by the dashboard and also
provides a trial-wise dataset interface for traditional machine learning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Emotiv Sample Data"

LABEL_MAPPING = {"relaxed": 0, "focused": 1}


@dataclass(slots=True)
class EEGTrialDataset:
    """Container for trial-wise EEG data used by scikit-learn training."""

    trials: np.ndarray
    labels: np.ndarray
    sfreq: float
    channel_names: list[str]
    label_mapping: dict[str, int]

    def validate(self) -> "EEGTrialDataset":
        if self.trials.ndim != 3:
            raise ValueError(
                f"Expected trials with shape (n_trials, n_channels, n_times), got {self.trials.shape}."
            )
        if self.labels.ndim != 1:
            raise ValueError(f"Expected 1D labels, got shape {self.labels.shape}.")
        if self.trials.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Number of trials and labels must match: "
                f"{self.trials.shape[0]} != {self.labels.shape[0]}."
            )
        unique_labels = set(np.unique(self.labels).tolist())
        allowed_labels = set(self.label_mapping.values())
        if not unique_labels.issubset(allowed_labels):
            raise ValueError(f"Labels must be a subset of {sorted(allowed_labels)}, got {sorted(unique_labels)}.")
        return self


def list_emotiv_edf_files(data_dir: Path | str | None = None) -> list[tuple[str, str, Path]]:
    """List (subject_id, session_id, path) for each EDF under data_dir."""
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    if not data_dir.is_dir():
        return []
    out = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("S"):
            continue
        subj_id = subj_dir.name
        for edf in sorted(subj_dir.glob("*.edf")) + sorted(subj_dir.glob("*.EDF")):
            stem = edf.stem
            if stem.startswith(subj_id) and len(stem) > len(subj_id):
                sess = stem[len(subj_id):]
                out.append((subj_id, sess, edf))
    return out


def get_edf_path(subject: int, session: int, data_dir: Path | str | None = None) -> Path | None:
    """Return Path for subject/session (1-based), or None if not found."""
    files = list_emotiv_edf_files(data_dir)
    subj_id = f"S{subject:03d}"
    sess_id = f"E{session:02d}"
    for s, e, p in files:
        if s == subj_id and e == sess_id:
            return p
    return None


def load_emotiv_edf(edf_path: Path | str, preload: bool = True, pick_eeg: bool = True) -> dict[str, Any]:
    """Load one EDF with MNE. Returns data, sfreq, channel_names, raw, etc."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install mne: pip install mne") from exc
    raw = mne.io.read_raw_edf(str(edf_path), preload=preload, verbose=False)
    if pick_eeg:
        try:
            raw.pick_types(eeg=True)
        except Exception:
            pass
    return {
        "data": raw.get_data(),
        "sfreq": float(raw.info["sfreq"]),
        "channel_names": raw.ch_names,
        "time": raw.times.copy(),
        "raw": raw,
    }


def load_band_time_series(
    edf_path: Path | str,
    max_duration_sec: float | None = 60.0,
    target_sfreq: int | None = 128,
) -> dict[str, Any]:
    """
    Load EDF, bandpass filter per band, average over channels, optional downsampling.
    Returns dict with keys: time, delta, theta, alpha, beta, gamma (lists for JSON).
    """
    seg = load_emotiv_edf(edf_path, preload=True, pick_eeg=True)
    raw = seg["raw"]
    sfreq = seg["sfreq"]
    if max_duration_sec is not None:
        raw = raw.copy().crop(tmin=0, tmax=min(max_duration_sec, raw.times[-1]))
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    band_series = {}
    for name, (lo, hi) in bands.items():
        raw_band = raw.copy().filter(l_freq=lo, h_freq=hi, verbose=False)
        band_series[name] = raw_band.get_data().mean(axis=0)
    time = raw.times.copy()
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


def load_band_time_series_for_subject_session(
    subject: int,
    session: int,
    max_duration_sec: float = 60.0,
    target_sfreq: int = 128,
    data_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load band time series for subject/session (1-based). Returns None if file not found."""
    path = get_edf_path(subject, session, data_dir)
    if path is None:
        return None
    return load_band_time_series(path, max_duration_sec=max_duration_sec, target_sfreq=target_sfreq)


def load_trial_dataset_from_npz(npz_path: Path | str) -> EEGTrialDataset:
    """
    Load a trial-wise EEG dataset from an .npz file.

    Expected keys:
    - trials: shape (n_trials, n_channels, n_times)
    - labels: shape (n_trials,)
    - sfreq: scalar sampling frequency
    Optional keys:
    - channel_names: array-like of strings
    """
    data = np.load(Path(npz_path), allow_pickle=True)
    trials = np.asarray(data["trials"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    sfreq = float(np.asarray(data["sfreq"]).item())
    if "channel_names" in data:
        channel_names = [str(name) for name in np.asarray(data["channel_names"]).tolist()]
    else:
        channel_names = [f"ch_{idx:02d}" for idx in range(trials.shape[1])]
    return EEGTrialDataset(
        trials=trials,
        labels=labels,
        sfreq=sfreq,
        channel_names=channel_names,
        label_mapping=LABEL_MAPPING.copy(),
    ).validate()


def load_mock_focus_relax_dataset(
    n_trials_per_class: int = 60,
    n_channels: int = 8,
    n_times: int = 512,
    sfreq: float = 128.0,
    random_state: int = 42,
) -> EEGTrialDataset:
    """
    Generate a synthetic focused-vs-relaxed EEG dataset for pipeline testing.

    Relaxed trials are alpha-dominant; focused trials are beta-dominant with
    slightly lower alpha. This is not physiologically complete, but it gives
    the training pipeline a realistic enough baseline to run end-to-end.
    """
    rng = np.random.default_rng(random_state)
    time = np.arange(n_times, dtype=float) / sfreq

    def _make_class_trials(alpha_scale: float, beta_scale: float, noise_scale: float) -> np.ndarray:
        trials = np.zeros((n_trials_per_class, n_channels, n_times), dtype=float)
        for trial_idx in range(n_trials_per_class):
            for ch_idx in range(n_channels):
                alpha_freq = rng.uniform(9.0, 11.5)
                beta_freq = rng.uniform(16.0, 24.0)
                theta_freq = rng.uniform(4.5, 7.0)
                phase = rng.uniform(0.0, 2.0 * np.pi, size=3)

                alpha_component = alpha_scale * np.sin(2.0 * np.pi * alpha_freq * time + phase[0])
                beta_component = beta_scale * np.sin(2.0 * np.pi * beta_freq * time + phase[1])
                theta_component = 0.25 * np.sin(2.0 * np.pi * theta_freq * time + phase[2])
                slow_drift = 0.15 * np.sin(2.0 * np.pi * 1.0 * time + rng.uniform(0.0, 2.0 * np.pi))
                noise = rng.normal(0.0, noise_scale, size=n_times)
                channel_gain = rng.uniform(0.85, 1.15)

                trials[trial_idx, ch_idx, :] = channel_gain * (
                    alpha_component + beta_component + theta_component + slow_drift + noise
                )
        return trials

    relaxed_trials = _make_class_trials(alpha_scale=1.25, beta_scale=0.55, noise_scale=0.35)
    focused_trials = _make_class_trials(alpha_scale=0.55, beta_scale=1.3, noise_scale=0.35)
    trials = np.concatenate([relaxed_trials, focused_trials], axis=0)
    labels = np.concatenate(
        [
            np.full(n_trials_per_class, LABEL_MAPPING["relaxed"], dtype=int),
            np.full(n_trials_per_class, LABEL_MAPPING["focused"], dtype=int),
        ]
    )
    channel_names = [f"EEG{idx + 1:02d}" for idx in range(n_channels)]
    order = rng.permutation(trials.shape[0])
    return EEGTrialDataset(
        trials=trials[order],
        labels=labels[order],
        sfreq=sfreq,
        channel_names=channel_names,
        label_mapping=LABEL_MAPPING.copy(),
    ).validate()


def load_focus_relax_dataset(dataset: str = "mock", npz_path: Path | str | None = None) -> EEGTrialDataset:
    """
    Return a binary focused-vs-relaxed trial dataset.

    Use `dataset="mock"` to run the pipeline without external data.
    Use `dataset="npz"` with `npz_path` for a real trial-wise dataset export.
    """
    dataset_key = dataset.strip().lower()
    if dataset_key == "mock":
        return load_mock_focus_relax_dataset()
    if dataset_key == "npz":
        if npz_path is None:
            raise ValueError("npz_path is required when dataset='npz'.")
        return load_trial_dataset_from_npz(npz_path)
    raise ValueError(
        f"Unsupported dataset '{dataset}'. Use 'mock' now, or extend this function for Shin2017A/custom loaders."
    )
