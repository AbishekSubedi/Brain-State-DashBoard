"""Generate poster-ready figures from the Shin2017 motor imagery pipeline."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.eeg_loader import load_shin2017_left_right_trials
from pipeline.state_classifier import (
    IMAGERY_FILTER_BANK,
    IMAGERY_TRAINING_WINDOW_SEC,
    SECOND_MODEL_PATHS,
    build_imagery_classifier,
    predict_imagery_session_timeline,
)

OUTPUT_DIR = PROJECT_ROOT / "poster_figures"
SECOND_METADATA_PATH = SECOND_MODEL_PATHS["metadata_path"]
BG = "#ffffff"
TEXT = "#10233b"
MUTED = "#44556f"
BORDER = (0.06, 0.14, 0.24, 0.16)
CARD = "#eef4ff"
ACCENT = "#78e0b5"


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _save(fig: plt.Figure, filename: str) -> None:
    output_path = _ensure_output_dir() / filename
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor(BG)
    if grid_axis == "both":
        ax.grid(color=(0.06, 0.14, 0.24, 0.08), linewidth=0.9)
    elif grid_axis == "x":
        ax.grid(axis="x", color=(0.06, 0.14, 0.24, 0.08), linewidth=0.9)
    elif grid_axis == "y":
        ax.grid(axis="y", color=(0.06, 0.14, 0.24, 0.08), linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=TEXT, labelsize=12)


def _load_metadata() -> dict:
    return json.loads(SECOND_METADATA_PATH.read_text(encoding="utf-8"))


def _plot_dataset_overview(dataset, metadata: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    cards = [
        ("Dataset", metadata["dataset"]),
        ("Task", "Left vs Right Hand"),
        ("Trials", f"{metadata['n_trials']:,}"),
        ("Channels", str(metadata["n_channels"])),
        ("Sampling Rate", f"{metadata['sfreq']:.0f} Hz"),
        ("Eval", metadata["evaluation"].replace("-", " ").title()),
    ]

    for index, (label, value) in enumerate(cards):
        row = index // 3
        col = index % 3
        x = 0.06 + col * 0.31
        y = 0.60 - row * 0.34
        card = FancyBboxPatch(
            (x, y),
            0.25,
            0.24,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=BORDER,
            facecolor=CARD,
        )
        ax.add_patch(card)
        ax.text(x + 0.03, y + 0.16, label, color=MUTED, fontsize=15, fontweight="bold")
        ax.text(x + 0.03, y + 0.07, value, color=TEXT, fontsize=22, fontweight="bold")

    ax.text(0.06, 0.93, "Shin2017A Dataset Overview", color=TEXT, fontsize=28, fontweight="bold")
    ax.text(
        0.06,
        0.08,
        "Motor imagery trials used for subject-aware left-vs-right hand classification.",
        color=MUTED,
        fontsize=15,
    )
    _save(fig, "fig01_dataset_overview.png")


def _plot_subject_distribution(dataset) -> None:
    subject_ids = np.unique(dataset.groups)
    left_counts = []
    right_counts = []
    for subject_id in subject_ids:
        subject_mask = dataset.groups == subject_id
        subject_labels = dataset.labels[subject_mask]
        left_counts.append(int(np.sum(subject_labels == 0)))
        right_counts.append(int(np.sum(subject_labels == 1)))

    fig, ax = plt.subplots(figsize=(12.5, 6.8), facecolor=BG)
    _style_axes(ax, grid_axis="y")
    x = np.arange(len(subject_ids))
    ax.bar(x, left_counts, color="#79c8ff", label="Left Hand")
    ax.bar(x, right_counts, bottom=left_counts, color="#ff7b8b", label="Right Hand")
    ax.set_title("Trial Distribution by Subject", color=TEXT, fontsize=24, fontweight="bold", pad=16)
    ax.set_xlabel("Subject", color=MUTED, fontsize=15, fontweight="bold")
    ax.set_ylabel("Trials", color=MUTED, fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(subject) for subject in subject_ids], fontsize=11, color=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper right", fontsize=13)
    _save(fig, "fig02_subject_distribution.png")


def _plot_trial_examples(dataset) -> None:
    preferred_channels = ["C3", "C4"]
    channel_indices = [dataset.channel_names.index(ch) for ch in preferred_channels if ch in dataset.channel_names]
    if len(channel_indices) < 2:
        channel_indices = [0, 1]
        preferred_channels = [dataset.channel_names[0], dataset.channel_names[1]]

    left_trials = dataset.trials[dataset.labels == 0][:40, channel_indices]
    right_trials = dataset.trials[dataset.labels == 1][:40, channel_indices]
    time_axis = np.arange(dataset.trials.shape[-1], dtype=float) / dataset.sfreq

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True, facecolor=BG)
    for ax, trials, title, color in [
        (axes[0], left_trials, "Average Left-Hand Trials", "#79c8ff"),
        (axes[1], right_trials, "Average Right-Hand Trials", "#ff7b8b"),
    ]:
        _style_axes(ax)
        for channel_idx, channel_name in enumerate(preferred_channels):
            signal = trials[:, channel_idx, :].mean(axis=0)
            normalized = signal / (np.std(signal) + 1e-9)
            ax.plot(time_axis, normalized, linewidth=2.3, label=channel_name)
        ax.axvspan(
            IMAGERY_TRAINING_WINDOW_SEC[0],
            IMAGERY_TRAINING_WINDOW_SEC[1],
            color=color,
            alpha=0.15,
            label="Training Window",
        )
        ax.set_title(title, color=TEXT, fontsize=20, fontweight="bold")
        ax.set_xlabel("Time (s)", color=MUTED, fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Normalized Amplitude", color=MUTED, fontsize=14, fontweight="bold")
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=12)
    _save(fig, "fig03_trial_examples.png")


def _plot_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    boxes = [
        (0.02, "Shin2017A\nEEG Trials"),
        (0.23, "Crop 2-6 s\nMotor Window"),
        (0.44, "Filter Bank\n8-12 / 12-20 / 20-30 Hz"),
        (0.67, "FBCSP + LDA\nClassifier"),
        (0.86, "Playback\nSimulation"),
    ]

    for x, label in boxes:
        box = FancyBboxPatch(
            (x, 0.32),
            0.13,
            0.34,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=1.4,
            edgecolor=BORDER,
            facecolor=CARD,
        )
        ax.add_patch(box)
        ax.text(x + 0.065, 0.49, label, ha="center", va="center", color=TEXT, fontsize=14, fontweight="bold")

    for start, end in [(0.15, 0.23), (0.36, 0.44), (0.57, 0.67), (0.80, 0.86)]:
        ax.annotate("", xy=(end, 0.49), xytext=(start, 0.49), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=3.0))

    ax.text(0.02, 0.84, "Implemented Motor Imagery Pipeline", color=TEXT, fontsize=28, fontweight="bold")
    ax.text(0.02, 0.12, "Code path: eeg_loader.py -> preprocess.py -> state_classifier.py -> dashboard simulation", color=MUTED, fontsize=14)
    _save(fig, "fig04_pipeline_overview.png")


def _compute_confusion_matrix(dataset) -> np.ndarray:
    def crop_trials(trials: np.ndarray, sfreq: float, start_sec: float, stop_sec: float) -> np.ndarray:
        start_idx = max(0, int(round(start_sec * sfreq)))
        stop_idx = min(trials.shape[-1], int(round(stop_sec * sfreq)))
        return trials[..., start_idx:stop_idx]

    cropped = crop_trials(dataset.trials, dataset.sfreq, *IMAGERY_TRAINING_WINDOW_SEC)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(cropped, dataset.labels, groups=dataset.groups))
    model = build_imagery_classifier(model_name="csp_lda", random_state=42)
    model.named_steps["fbcsp"].sfreq = dataset.sfreq
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.fit(cropped[train_idx], dataset.labels[train_idx])
        predictions = model.predict(cropped[test_idx])
    return confusion_matrix(dataset.labels[test_idx], predictions)


def _plot_confusion_matrix(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor=BG)
    ax.set_facecolor(BG)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Left Hand", "Right Hand"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Subject-Aware Test Confusion Matrix", color=TEXT, fontsize=22, fontweight="bold", pad=14)
    ax.tick_params(colors=TEXT, labelsize=14)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    for text in ax.texts:
        text.set_color("#04101b" if text.get_text() != "0" else TEXT)
        text.set_fontweight("bold")
        text.set_fontsize(16)
    _save(fig, "fig05_confusion_matrix.png")


def _plot_results_summary(metadata: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=BG)
    _style_axes(ax, grid_axis="y")
    labels = ["Chance", "Test Accuracy", "CV Mean"]
    values = [0.5, metadata["test_accuracy"], metadata["cross_val_accuracy_mean"]]
    errors = [0.0, 0.0, metadata["cross_val_accuracy_std"]]
    colors = ["#44556f", "#ff9f40", "#78e0b5"]
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=8, color=colors, edgecolor="none")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Accuracy", color=MUTED, fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=14, fontweight="bold")
    for xpos, value in zip(x, values, strict=True):
        ax.text(xpos, value + 0.02, f"{value * 100:.1f}%", ha="center", color=TEXT, fontsize=14, fontweight="bold")
    ax.set_title("Final Model Performance", color=TEXT, fontsize=24, fontweight="bold", pad=16)
    _save(fig, "fig06_results_summary.png")


def _plot_playback_timeline() -> None:
    playback = predict_imagery_session_timeline(subject=1, session=1)
    timeline = playback["timeline"]
    if not timeline:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
        facecolor=BG,
    )
    _style_axes(ax_top)
    _style_axes(ax_bottom, grid_axis="x")

    for band, color in [("theta", "#68bb9a"), ("alpha", "#6384ff"), ("beta", "#ff9f40")]:
        ax_top.plot(playback["time"], playback[band], linewidth=2.0, color=color, label=band.title())
    ax_top.set_title("Imagery Session Playback Example", color=TEXT, fontsize=24, fontweight="bold", pad=14)
    ax_top.set_ylabel("Normalized Band Signal", color=MUTED, fontsize=15, fontweight="bold")
    ax_top.legend(frameon=False, labelcolor=TEXT, loc="upper right", fontsize=13)

    label_colors = {"Left Hand": "#79c8ff", "Right Hand": "#ff7b8b"}
    for item in timeline:
        ax_bottom.barh(
            [0],
            width=item["end"] - item["start"],
            left=item["start"],
            height=0.56,
            color=label_colors[item["label"]],
            edgecolor="none",
        )
    ax_bottom.set_yticks([0])
    ax_bottom.set_yticklabels(["Predicted Intent"], color=TEXT, fontsize=13, fontweight="bold")
    ax_bottom.set_xlabel("Time (s)", color=MUTED, fontsize=15, fontweight="bold")
    _save(fig, "fig07_playback_timeline.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poster-ready figures from the Shin2017 pipeline.")
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "summary"],
        help="Generate all figures or only the text-heavy summary figures.",
    )
    args = parser.parse_args()
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.size": 13,
            "axes.titlesize": 22,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    metadata = _load_metadata()
    if args.subset == "summary":
        _plot_dataset_overview(None, metadata)
        _plot_pipeline_overview()
        matrix = np.asarray([[111, 69], [89, 91]], dtype=int)
        _plot_confusion_matrix(matrix)
        _plot_results_summary(metadata)
        print(f"Saved summary figures to {OUTPUT_DIR}")
        return

    dataset = load_shin2017_left_right_trials()
    _plot_dataset_overview(dataset, metadata)
    _plot_subject_distribution(dataset)
    _plot_trial_examples(dataset)
    _plot_pipeline_overview()
    matrix = _compute_confusion_matrix(dataset)
    _plot_confusion_matrix(matrix)
    _plot_results_summary(metadata)
    _plot_playback_timeline()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
