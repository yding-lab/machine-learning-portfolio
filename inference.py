"""Safe model loading and batch/single-record inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from config import ARTIFACT_DIR


@dataclass(frozen=True)
class Prediction:
    """Human-readable prediction object."""

    predicted_class: int
    probability_positive: float
    threshold: float


class ModelArtifactError(RuntimeError):
    """Raised when a trained model artifact is unavailable or invalid."""


def load_model(path: str | Path | None = None):
    """Load the serialized sklearn Pipeline."""
    model_path = Path(path) if path else ARTIFACT_DIR / "best_model.joblib"

    if not model_path.exists():
        raise ModelArtifactError(
            f"Model artifact not found at {model_path}. "
            "Run `python src/train.py` first."
        )

    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise ModelArtifactError(
            f"Unable to load model artifact: {exc}"
        ) from exc


def predict_dataframe(
    model,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return class probabilities and thresholded predictions."""
    if X.empty:
        raise ValueError("Inference DataFrame cannot be empty.")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = X.copy()
    result["probability_positive"] = probabilities
    result["predicted_class"] = predictions
    result["decision_threshold"] = threshold
    return result


def predict_one(
    model,
    features: dict[str, float],
    threshold: float = 0.5,
) -> Prediction:
    """Predict one record represented as a feature dictionary."""
    frame = pd.DataFrame([features])
    probability = float(model.predict_proba(frame)[0, 1])
    predicted_class = int(probability >= threshold)

    return Prediction(
        predicted_class=predicted_class,
        probability_positive=probability,
        threshold=threshold,
    )
