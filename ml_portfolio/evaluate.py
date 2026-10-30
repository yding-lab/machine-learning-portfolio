"""Evaluation utilities for classification models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Serializable evaluation metrics."""

    roc_auc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float


def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a fitted binary classifier using probability-aware metrics."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = ClassificationMetrics(
        roc_auc=float(roc_auc_score(y, probabilities)),
        average_precision=float(average_precision_score(y, probabilities)),
        accuracy=float(accuracy_score(y, predictions)),
        balanced_accuracy=float(
            balanced_accuracy_score(y, predictions)
        ),
        precision=float(
            precision_score(y, predictions, zero_division=0)
        ),
        recall=float(recall_score(y, predictions, zero_division=0)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(y, probabilities)),
    )

    return {
        "metrics": asdict(metrics),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "prediction_summary": {
            "mean_probability": float(np.mean(probabilities)),
            "min_probability": float(np.min(probabilities)),
            "max_probability": float(np.max(probabilities)),
            "positive_prediction_rate": float(np.mean(predictions)),
        },
    }
