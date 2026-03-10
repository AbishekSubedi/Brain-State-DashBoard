"""
Load EEG data from Emotiv Sample Data (EDF). Used by the dashboard and state pipeline.
Replace or extend for other datasets or real-time input.
"""

from pathlib import Path
from typing import Any

try:
    import mne
except ImportError:
    mne = None  # type: ignore

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Emotiv Sample Data"


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
    if mne is None:
        raise ImportError("Install mne: pip install mne")
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
    if mne is None:
        raise ImportError("Install mne: pip install mne")
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
        files = list_emotiv_edf_files(data_dir)
        if not files:
            return None
        path = files[0][2]
    return load_band_time_series(path, max_duration_sec=max_duration_sec, target_sfreq=target_sfreq)
