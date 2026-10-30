"""Model definitions and hyperparameter search spaces."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import DEFAULT_CONFIG, TrainingConfig


@dataclass(frozen=True)
class ModelCandidate:
    """A candidate estimator and its GridSearchCV parameter space."""

    name: str
    estimator: object
    param_grid: dict[str, list[object]]


def build_model_candidates(
    config: TrainingConfig = DEFAULT_CONFIG,
) -> list[ModelCandidate]:
    """Return interpretable and nonlinear baseline candidates.

    The main search values come from TrainingConfig so a single config controls the run.
    """
    random_state = config.random_state
    logistic = ModelCandidate(
        name="logistic_regression",
        estimator=LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        param_grid={
            "model__C": list(config.logistic_c_values),
            # 0 = L2 (ridge), 1 = L1 (lasso); replaces `penalty`, deprecated in scikit-learn 1.8.
            "model__l1_ratio": [0.0, 1.0],
        },
    )

    forest = ModelCandidate(
        name="random_forest",
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=config.n_jobs,
        ),
        param_grid={
            "model__n_estimators": list(config.rf_estimators),
            "model__max_depth": list(config.rf_max_depth),
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7],
        },
    )

    return [logistic, forest]
