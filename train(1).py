"""End-to-end model training and model selection."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import ARTIFACT_DIR, REPORT_DIR, DEFAULT_CONFIG, TrainingConfig
from data import make_train_test_split
from evaluate import evaluate_classifier
from features import build_preprocessor
from models import build_model_candidates


def train_and_select(
    config: TrainingConfig = DEFAULT_CONFIG,
) -> dict:
    """Train candidate models and persist the strongest model by CV ROC-AUC."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = make_train_test_split(config)
    preprocessor = build_preprocessor(bundle.feature_names)

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    candidate_results: list[dict] = []
    best_search = None
    best_name = None
    best_cv_score = float("-inf")

    for candidate in build_model_candidates(config.random_state):
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", candidate.estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=candidate.param_grid,
            scoring=config.scoring,
            cv=cv,
            n_jobs=config.n_jobs,
            refit=True,
            return_train_score=True,
        )
        search.fit(bundle.X_train, bundle.y_train)

        candidate_results.append(
            {
                "model": candidate.name,
                "best_cv_score": float(search.best_score_),
                "best_params": search.best_params_,
            }
        )

        if search.best_score_ > best_cv_score:
            best_cv_score = float(search.best_score_)
            best_search = search
            best_name = candidate.name

    if best_search is None:
        raise RuntimeError("No model candidate completed training.")

    holdout_results = evaluate_classifier(
        best_search.best_estimator_,
        bundle.X_test,
        bundle.y_test,
        threshold=config.decision_threshold,
    )

    model_path = ARTIFACT_DIR / "best_model.joblib"
    joblib.dump(best_search.best_estimator_, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_name,
        "best_cv_score": best_cv_score,
        "best_params": best_search.best_params_,
        "feature_names": bundle.feature_names,
        "training_config": asdict(config),
        "candidate_results": candidate_results,
        "holdout_results": holdout_results,
        "artifact": str(model_path.relative_to(Path(__file__).resolve().parents[1])),
    }

    report_path = REPORT_DIR / "training_report.json"
    report_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


if __name__ == "__main__":
    result = train_and_select()
    print(json.dumps(result, indent=2))
