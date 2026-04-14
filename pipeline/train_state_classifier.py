"""
Train a focused-vs-relaxed EEG classifier using traditional ML.

Run:
    python -m pipeline.train_state_classifier --dataset mock

Later, connect a real dataset by extending `pipeline.eeg_loader.load_focus_relax_dataset`
or by exporting trial-wise data to an .npz file and using `--dataset npz`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline.eeg_loader import EEGTrialDataset, load_focus_relax_dataset
from pipeline.evaluate import cross_validate_accuracy, evaluate_classifier
from pipeline.feature_extractor import compute_bandpower_features
from pipeline.preprocess import preprocess_trials


def build_classifier(model_name: str = "svm", random_state: int = 42) -> Pipeline:
    """Build a scikit-learn pipeline with scaling and a swap-friendly classifier."""
    model_key = model_name.strip().lower()
    estimators: dict[str, Any] = {
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "logreg": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if model_key not in estimators:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {sorted(estimators)}.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimators[model_key]),
        ]
    )


def train_and_evaluate(
    dataset: EEGTrialDataset,
    model_name: str,
    test_size: float,
    random_state: int,
    apply_baseline: bool,
    baseline_samples: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Run preprocessing, feature extraction, training, evaluation, and persistence."""
    processed_trials = preprocess_trials(
        dataset.trials,
        sfreq=dataset.sfreq,
        apply_baseline=apply_baseline,
        baseline_samples=baseline_samples,
    )
    X, feature_names = compute_bandpower_features(processed_trials, sfreq=dataset.sfreq)
    y = dataset.labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = build_classifier(model_name=model_name, random_state=random_state)
    model.fit(X_train, y_train)

    target_names = [label for label, _ in sorted(dataset.label_mapping.items(), key=lambda item: item[1])]
    metrics = evaluate_classifier(model, X_test, y_test, target_names=target_names)
    cv_scores = cross_validate_accuracy(model, X, y, cv=5)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "brain_state_classifier.joblib"
    label_mapping_path = output_dir / "label_mapping.json"
    metadata_path = output_dir / "training_metadata.json"

    joblib.dump(model, model_path)
    label_mapping_path.write_text(json.dumps(dataset.label_mapping, indent=2), encoding="utf-8")
    metadata = {
        "model_name": model_name,
        "sfreq": dataset.sfreq,
        "n_trials": int(dataset.trials.shape[0]),
        "n_channels": int(dataset.trials.shape[1]),
        "n_times": int(dataset.trials.shape[2]),
        "feature_names": feature_names,
        "cross_val_accuracy_mean": float(np.mean(cv_scores)),
        "cross_val_accuracy_std": float(np.std(cv_scores)),
        "test_accuracy": metrics["accuracy"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "feature_names": feature_names,
        "model_path": model_path,
        "label_mapping_path": label_mapping_path,
        "metadata_path": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a focused-vs-relaxed EEG classifier.")
    parser.add_argument("--dataset", default="mock", choices=["mock", "npz"], help="Dataset source to use.")
    parser.add_argument("--npz-path", default=None, help="Path to an .npz trial dataset when --dataset=npz.")
    parser.add_argument("--model", default="svm", choices=["svm", "logreg", "random_forest"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--baseline-samples", type=int, default=None)
    parser.add_argument(
        "--disable-baseline",
        action="store_true",
        help="Skip baseline correction before filtering.",
    )
    parser.add_argument("--output-dir", default="artifacts/brain_state_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_focus_relax_dataset(dataset=args.dataset, npz_path=args.npz_path)
    results = train_and_evaluate(
        dataset=dataset,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        apply_baseline=not args.disable_baseline,
        baseline_samples=args.baseline_samples,
        output_dir=Path(args.output_dir),
    )

    print(f"Dataset trials: {dataset.trials.shape}")
    print(f"Sampling frequency: {dataset.sfreq} Hz")
    print(f"Saved model: {results['model_path']}")
    print(f"Saved label mapping: {results['label_mapping_path']}")
    print(f"Saved metadata: {results['metadata_path']}")
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
