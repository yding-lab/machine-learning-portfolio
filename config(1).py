"""Project configuration for the production-style ML portfolio.

The project intentionally keeps configuration in one place so that training,
evaluation, inference, and serving use the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


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
