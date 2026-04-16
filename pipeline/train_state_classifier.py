"""
Train a focused-vs-relaxed EEG classifier using traditional ML.

Run:
    python -m pipeline.train_state_classifier

The first model now trains on Shin2017B, because that split contains the
usable focused-vs-relaxed labels (`subtraction` vs `rest`).
"""

from __future__ import annotations

import argparse
import numpy as np

from pipeline.state_classifier import train_first_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a focused-vs-relaxed EEG classifier.")
    parser.add_argument("--model", default="svm", choices=["svm", "logreg", "random_forest"])
    parser.add_argument("--subject", type=int, action="append", dest="subjects", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_first_model(
        model_name=args.model,
        subjects=args.subjects,
        random_state=args.random_state,
    )

    metadata = results["metadata"]
    print(f"Dataset: {metadata['dataset']}")
    print(f"Trials: {metadata['n_trials']}")
    print(f"Channels: {metadata['n_channels']}")
    print(f"Sampling frequency: {metadata['sfreq']} Hz")
    print(f"Saved artifacts: {results['artifact_dir']}")
    print()
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print("Confusion matrix:")
    print(results["metrics"]["confusion_matrix"])
    print()
    print("Classification report:")
    print(results["metrics"]["classification_report"])
    print(
        "Cross-validation accuracy: "
        f"{np.mean(results['cv_scores']):.4f} +/- {np.std(results['cv_scores']):.4f}"
    )


if __name__ == "__main__":
    main()
