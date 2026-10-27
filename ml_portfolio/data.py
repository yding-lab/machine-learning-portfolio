"""Dataset loading and validation utilities.

This portfolio uses scikit-learn's breast-cancer dataset so the repository
is fully reproducible without requiring credentials or private data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import DEFAULT_CONFIG, TrainingConfig


@dataclass
class DatasetBundle:
    """Train/test bundle used by model development code."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


class DatasetValidationError(ValueError):
    """Raised when data fails required quality checks."""


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the public dataset as pandas objects."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    y.name = "target"
    return X, y


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate core assumptions before training."""
    if X.empty or y.empty:
        raise DatasetValidationError("Dataset cannot be empty.")

    if len(X) != len(y):
        raise DatasetValidationError(
            f"Feature and target lengths differ: {len(X)} != {len(y)}"
        )

    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()
        raise DatasetValidationError(
            f"Duplicate feature names detected: {duplicates}"
        )

    null_counts = X.isna().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise DatasetValidationError(
            f"Missing feature values detected: {bad}"
        )

    if not set(y.unique()).issubset({0, 1}):
        raise DatasetValidationError(
            f"Expected binary target labels 0/1; found {sorted(y.unique())}"
        )

    if y.nunique() != 2:
        raise DatasetValidationError(
            "Both classes must be present for binary classification."
        )


def make_train_test_split(
    config: TrainingConfig = DEFAULT_CONFIG,
) -> DatasetBundle:
    """Create a stratified train/test split."""
    X, y = load_dataset()
    validate_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X.columns.tolist(),
    )
