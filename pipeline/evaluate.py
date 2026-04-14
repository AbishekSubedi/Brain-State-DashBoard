"""
Evaluation helpers for the focused-vs-relaxed state classifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def evaluate_classifier(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    """Run standard classification metrics on a held-out test set."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names, digits=4)
    return {
        "accuracy": float(accuracy),
        "confusion_matrix": matrix,
        "classification_report": report,
        "predictions": predictions,
    }


def cross_validate_accuracy(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> np.ndarray:
    """Return cross-validation accuracy scores."""
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy")
