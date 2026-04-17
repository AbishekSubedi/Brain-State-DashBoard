"""Dataset loaders for the two Shin2017 dashboard models."""

from dataclasses import dataclass
from typing import Any

import numpy as np

LABEL_MAPPING = {"relaxed": 0, "focused": 1}
SHIN2017_STATE_LABEL_MAPPING = {"rest": 0, "subtraction": 1}
SHIN2017_IMAGERY_LABEL_MAPPING = {"left_hand": 0, "right_hand": 1}


@dataclass(slots=True)
class EEGTrialDataset:
    """Container for trial-wise EEG data used by scikit-learn training."""

    trials: np.ndarray
    labels: np.ndarray
    sfreq: float
    channel_names: list[str]
    label_mapping: dict[str, int]
    groups: np.ndarray | None = None

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
        if self.groups is not None:
            if self.groups.ndim != 1:
                raise ValueError(f"Expected 1D groups, got shape {self.groups.shape}.")
            if self.groups.shape[0] != self.trials.shape[0]:
                raise ValueError(
                    "Number of group entries must match number of trials: "
                    f"{self.groups.shape[0]} != {self.trials.shape[0]}."
                )
        return self


def _get_shin2017_dataset(kind: str = "state", subjects: list[int] | None = None):
    """
    Lazily instantiate the relevant Shin2017 dataset.

    `kind="state"` maps to Shin2017B (subtraction vs rest), which is the
    usable focused-vs-relaxed split for the first model.
    """
    try:
        from moabb.datasets import Shin2017A, Shin2017B
    except ImportError as exc:
        raise ImportError("Install moabb and mne to use the Shin2017 datasets.") from exc

    if kind == "state":
        return Shin2017B(accept=True, subjects=subjects)
    if kind == "imagery":
        return Shin2017A(accept=True, subjects=subjects)
    raise ValueError(f"Unsupported Shin2017 kind '{kind}'.")


def _sorted_session_keys(session_dict: dict[str, Any]) -> list[str]:
    return sorted(
        session_dict.keys(),
        key=lambda key: (int("".join(ch for ch in key if ch.isdigit()) or 0), key),
    )


def list_shin2017_sessions(subject: int, kind: str = "state") -> list[dict[str, Any]]:
    """Return user-facing session numbers for a Shin2017 subject."""
    dataset = _get_shin2017_dataset(kind=kind, subjects=[subject])
    subject_data = dataset.get_data(subjects=[subject]).get(subject, {})
    session_keys = _sorted_session_keys(subject_data)
    return [
        {
            "session": index + 1,
            "session_name": key,
        }
        for index, key in enumerate(session_keys)
    ]


def load_shin2017_session(subject: int, session: int, kind: str = "state") -> dict[str, Any]:
    """
    Load one Shin2017 session as continuous EEG for playback.

    `session` is a user-facing 1-based index into the available sessions for a
    subject in the chosen paradigm.
    """
    dataset = _get_shin2017_dataset(kind=kind, subjects=[subject])
    subject_data = dataset.get_data(subjects=[subject]).get(subject, {})
    session_keys = _sorted_session_keys(subject_data)
    if session < 1 or session > len(session_keys):
        raise ValueError(f"Session must be between 1 and {len(session_keys)}, got {session}.")

    session_name = session_keys[session - 1]
    runs = subject_data[session_name]
    run_keys = _sorted_session_keys(runs)
    raw = runs[run_keys[0]].copy()
    raw.pick("eeg")
    return {
        "subject": subject,
        "session": session,
        "session_name": session_name,
        "data": raw.get_data(),
        "sfreq": float(raw.info["sfreq"]),
        "channel_names": raw.ch_names,
        "time": raw.times.copy(),
    }


def load_shin2017_focus_relax_trials(subjects: list[int] | None = None) -> EEGTrialDataset:
    """
    Load trial-wise EEG from Shin2017B and map `rest`/`subtraction` to the
    first website model's `relaxed`/`focused` labels.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install mne to epoch the Shin2017 dataset.") from exc

    dataset = _get_shin2017_dataset(kind="state", subjects=subjects)
    subjects_to_load = subjects or dataset.subject_list
    data = dataset.get_data(subjects=subjects_to_load)
    all_trials: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    channel_names: list[str] | None = None
    sfreq: float | None = None
    event_code_to_name = {code: name for name, code in dataset.event_id.items()}

    for subject in subjects_to_load:
        subject_sessions = data.get(subject, {})
        for session_name in _sorted_session_keys(subject_sessions):
            runs = subject_sessions[session_name]
            for run_name in _sorted_session_keys(runs):
                raw = runs[run_name].copy()
                events = mne.find_events(raw, stim_channel="Stim", shortest_event=1, verbose=False)
                epochs = mne.Epochs(
                    raw,
                    events=events,
                    event_id=dataset.event_id,
                    tmin=dataset.interval[0],
                    tmax=dataset.interval[1],
                    baseline=None,
                    picks="eeg",
                    preload=True,
                    verbose=False,
                )
                X = epochs.get_data(copy=True)
                y = np.asarray(
                    [SHIN2017_STATE_LABEL_MAPPING[event_code_to_name[int(event_code)]] for event_code in epochs.events[:, 2]],
                    dtype=int,
                )
                all_trials.append(X)
                all_labels.append(y)
                all_groups.append(np.full(X.shape[0], subject, dtype=int))
                if channel_names is None:
                    channel_names = list(epochs.ch_names)
                if sfreq is None:
                    sfreq = float(epochs.info["sfreq"])

    if not all_trials or channel_names is None or sfreq is None:
        raise RuntimeError("No Shin2017B trials were loaded.")

    trials = np.concatenate(all_trials, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    return EEGTrialDataset(
        trials=trials,
        labels=labels,
        sfreq=sfreq,
        channel_names=channel_names,
        label_mapping=LABEL_MAPPING.copy(),
        groups=groups,
    ).validate()


def load_shin2017_left_right_trials(subjects: list[int] | None = None) -> EEGTrialDataset:
    """
    Load trial-wise EEG from Shin2017A and map `left_hand`/`right_hand` to
    integer labels for the second motor-imagery model.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install mne to epoch the Shin2017 dataset.") from exc

    dataset = _get_shin2017_dataset(kind="imagery", subjects=subjects)
    subjects_to_load = subjects or dataset.subject_list
    data = dataset.get_data(subjects=subjects_to_load)
    all_trials: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    channel_names: list[str] | None = None
    sfreq: float | None = None
    event_code_to_name = {code: name for name, code in dataset.event_id.items()}

    for subject in subjects_to_load:
        subject_sessions = data.get(subject, {})
        for session_name in _sorted_session_keys(subject_sessions):
            runs = subject_sessions[session_name]
            for run_name in _sorted_session_keys(runs):
                raw = runs[run_name].copy()
                events = mne.find_events(raw, stim_channel="Stim", shortest_event=1, verbose=False)
                epochs = mne.Epochs(
                    raw,
                    events=events,
                    event_id=dataset.event_id,
                    tmin=dataset.interval[0],
                    tmax=dataset.interval[1],
                    baseline=None,
                    picks="eeg",
                    preload=True,
                    verbose=False,
                )
                X = epochs.get_data(copy=True)
                y = np.asarray(
                    [SHIN2017_IMAGERY_LABEL_MAPPING[event_code_to_name[int(event_code)]] for event_code in epochs.events[:, 2]],
                    dtype=int,
                )
                all_trials.append(X)
                all_labels.append(y)
                all_groups.append(np.full(X.shape[0], subject, dtype=int))
                if channel_names is None:
                    channel_names = list(epochs.ch_names)
                if sfreq is None:
                    sfreq = float(epochs.info["sfreq"])

    if not all_trials or channel_names is None or sfreq is None:
        raise RuntimeError("No Shin2017A imagery trials were loaded.")

    trials = np.concatenate(all_trials, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    return EEGTrialDataset(
        trials=trials,
        labels=labels,
        sfreq=sfreq,
        channel_names=channel_names,
        label_mapping={"left_hand": 0, "right_hand": 1},
        groups=groups,
    ).validate()
