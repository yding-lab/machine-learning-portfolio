"""Project configuration for the production-style ML portfolio.

The project intentionally keeps configuration in one place so that training,
evaluation, inference, and serving use the same defaults.

Dataset note: scikit-learn's Breast Cancer Wisconsin (Diagnostic) data is used as an
educational benchmark only. Nothing here is clinically validated or suitable for
medical decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", PROJECT_ROOT / "reports"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACT_DIR / "best_model.joblib"))
METADATA_PATH = MODEL_PATH.with_name("model_metadata.json")

# scikit-learn encodes the target as 0 = malignant, 1 = benign. "Positive" in every
# metric and in the API therefore means *benign*. This is easy to get backwards.
TARGET_NAMES = {0: "malignant", 1: "benign"}
POSITIVE_CLASS = TARGET_NAMES[1]


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration shared across the model-development workflow."""

    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    scoring: str = "roc_auc"
    n_jobs: int = -1

    # Candidate values for model selection.
    logistic_c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    rf_estimators: tuple[int, ...] = (200, 400)
    rf_max_depth: tuple[int | None, ...] = (None, 8, 16)

    # Threshold used to turn probabilities into labels.
    decision_threshold: float = 0.50


DEFAULT_CONFIG = TrainingConfig()
