"""Model definitions and hyperparameter search spaces."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ModelCandidate:
    """A candidate estimator and its GridSearchCV parameter space."""

    name: str
    estimator: object
    param_grid: dict[str, list[object]]


def build_model_candidates(
    random_state: int = 42,
) -> list[ModelCandidate]:
    """Return interpretable and nonlinear baseline candidates."""
    logistic = ModelCandidate(
        name="logistic_regression",
        estimator=LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        param_grid={
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        },
    )

    forest = ModelCandidate(
        name="random_forest",
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        param_grid={
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7],
        },
    )

    return [logistic, forest]
