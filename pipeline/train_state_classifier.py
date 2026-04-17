"""
Train a focused-vs-relaxed EEG classifier using traditional ML.

Run:
    python -m pipeline.train_state_classifier

The first model now trains on Shin2017B, because that split contains the
usable focused-vs-relaxed labels (`subtraction` vs `rest`).

Use `--task imagery` for the second model based on Shin2017A
(`left_hand` vs `right_hand`).
"""

from __future__ import annotations

import argparse
import numpy as np

from pipeline.state_classifier import train_first_model, train_second_model

STATE_MODELS = ("svm", "logreg", "random_forest")
IMAGERY_MODELS = ("csp_lda", "fbcsp_svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a focused-vs-relaxed EEG classifier.")
    parser.add_argument("--task", default="state", choices=["state", "imagery"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--subject", type=int, action="append", dest="subjects", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    valid_models = STATE_MODELS if args.task == "state" else IMAGERY_MODELS
    if args.model is not None and args.model not in valid_models:
        parser.error(
            f"--model={args.model!r} is invalid for task {args.task!r}. "
            f"Choose from {', '.join(valid_models)}."
        )
    return args


def main() -> None:
    args = parse_args()
    if args.task == "state":
        results = train_first_model(
            model_name=args.model or "svm",
            subjects=args.subjects,
            random_state=args.random_state,
        )
    else:
        results = train_second_model(
            model_name=args.model or "csp_lda",
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
