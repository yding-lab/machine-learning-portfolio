"""Safe model loading and batch/single-record inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH, POSITIVE_CLASS


@dataclass(frozen=True)
class Prediction:
    """Human-readable prediction object."""

    predicted_class: int
    predicted_label: str
    probability_positive: float  # probability of the positive class (benign)
    threshold: float


class ModelArtifactError(RuntimeError):
    """Raised when a trained model artifact is unavailable or invalid."""


class FeatureValidationError(ValueError):
    """Raised when an inference payload doesn't match the model's features."""

    def __init__(self, missing: list[str], unexpected: list[str], non_finite: list[str]):
        self.missing, self.unexpected, self.non_finite = missing, unexpected, non_finite
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} feature(s): {missing}")
        if unexpected:
            parts.append(f"unexpected feature(s): {unexpected}")
        if non_finite:
            parts.append(f"non-finite value(s) for: {non_finite}")
        super().__init__("; ".join(parts))

    def to_dict(self) -> dict:
        return {"missing": self.missing, "unexpected": self.unexpected, "non_finite": self.non_finite}


def load_model(path: str | Path | None = None):
    """Load the serialized sklearn Pipeline."""
    model_path = Path(path) if path else MODEL_PATH

    if not model_path.exists():
        raise ModelArtifactError(
            f"Model artifact not found at {model_path}. "
            "Run `python -m ml_portfolio.train` first."
        )

    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise ModelArtifactError(
            f"Unable to load model artifact: {exc}"
        ) from exc


def load_metadata(path: str | Path | None = None) -> dict:
    """Metadata written next to the model by training (empty if absent)."""
    p = Path(path) if path else METADATA_PATH
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def expected_features(model) -> list[str]:
    """Feature names the fitted Pipeline was trained on, in order."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ModelArtifactError("Model artifact does not record its training feature names.")
    return list(names)


def validate_features(features: dict[str, float], expected: list[str]) -> pd.DataFrame:
    """Check a single record against the model's features and return it as a 1-row frame.

    Nothing is silently filled in: a missing feature is an error, not a zero.
    """
    missing = [f for f in expected if f not in features]
    unexpected = sorted(set(features) - set(expected))
    non_finite = [f for f in expected if f in features and not _finite(features[f])]
    if missing or unexpected or non_finite:
        raise FeatureValidationError(missing, unexpected, non_finite)
    return pd.DataFrame([[float(features[f]) for f in expected]], columns=expected)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    frame = validate_features(features, expected_features(model))
    probability = float(model.predict_proba(frame)[0, 1])
    predicted_class = int(probability >= threshold)

    return Prediction(
        predicted_class=predicted_class,
        predicted_label=POSITIVE_CLASS if predicted_class == 1 else "malignant",
        probability_positive=probability,
        threshold=threshold,
    )
