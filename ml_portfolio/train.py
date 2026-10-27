"""End-to-end model training and model selection.

    python -m ml_portfolio.train            # full grid (about a minute on a laptop)
    python -m ml_portfolio.train --quick    # tiny grid for smoke tests

Workflow: load and validate data -> stratified train/test split -> for each candidate,
GridSearchCV with stratified k-fold on the TRAINING split only -> pick the candidate with
the best mean CV ROC-AUC -> evaluate that single model once on the untouched test split ->
save the fitted Pipeline, its metadata, and a JSON + Markdown report.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_CONFIG,
    METADATA_PATH,
    MODEL_PATH,
    POSITIVE_CLASS,
    PROJECT_ROOT,
    REPORT_DIR,
    TARGET_NAMES,
    TrainingConfig,
)
from .data import make_train_test_split
from .evaluate import evaluate_classifier
from .features import build_preprocessor
from .models import build_model_candidates

QUICK_CONFIG = replace(DEFAULT_CONFIG, cv_folds=3, logistic_c_values=(1.0,), rf_estimators=(50,),
                       rf_max_depth=(8,))


def train_and_select(
    config: TrainingConfig = DEFAULT_CONFIG,
) -> dict:
    """Train candidate models and persist the strongest model by CV ROC-AUC."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
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

    for candidate in build_model_candidates(config):
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

        i = search.best_index_
        candidate_results.append(
            {
                "model": candidate.name,
                "best_cv_score": float(search.best_score_),
                "best_cv_score_std": float(search.cv_results_["std_test_score"][i]),
                "best_cv_train_score": float(search.cv_results_["mean_train_score"][i]),
                "grid_size": len(search.cv_results_["params"]),
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

    joblib.dump(best_search.best_estimator_, MODEL_PATH)

    y_all = list(bundle.y_train) + list(bundle.y_test)
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_name,
        "selection_rule": f"highest mean {config.cv_folds}-fold CV {config.scoring} on the training split",
        "best_cv_score": best_cv_score,
        "best_params": best_search.best_params_,
        "feature_names": bundle.feature_names,
        "target_encoding": {str(k): v for k, v in TARGET_NAMES.items()},
        "positive_class": POSITIVE_CLASS,
        "data_summary": {
            "rows": len(y_all),
            "train_rows": len(bundle.y_train),
            "test_rows": len(bundle.y_test),
            "class_counts": {TARGET_NAMES[c]: y_all.count(c) for c in (0, 1)},
        },
        "training_config": asdict(config),
        "candidate_results": candidate_results,
        "holdout_results": holdout_results,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "artifact": _relative(MODEL_PATH),
    }

    METADATA_PATH.write_text(json.dumps(_slim(metadata), indent=2), encoding="utf-8")
    (REPORT_DIR / "training_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "training_report.md").write_text(report_markdown(metadata), encoding="utf-8")
    return metadata


def _relative(path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _slim(metadata: dict) -> dict:
    """What the API needs to describe the loaded model."""
    keys = ("trained_at_utc", "selected_model", "selection_rule", "best_cv_score", "best_params",
            "feature_names", "target_encoding", "positive_class", "environment")
    out = {k: metadata[k] for k in keys}
    out["decision_threshold"] = metadata["training_config"]["decision_threshold"]
    out["test_metrics"] = metadata["holdout_results"]["metrics"]
    return out


def report_markdown(m: dict) -> str:
    met = m["holdout_results"]["metrics"]
    cm = m["holdout_results"]["confusion_matrix"]
    rep = m["holdout_results"]["classification_report"]
    d = m["data_summary"]
    lines = [
        "# Training report",
        "",
        f"Generated {m['trained_at_utc']} with scikit-learn {m['environment']['scikit_learn']} "
        f"(Python {m['environment']['python']}). Educational benchmark only, not for medical use.",
        "",
        f"Data: {d['rows']} rows ({d['class_counts']['malignant']} malignant, {d['class_counts']['benign']} benign); "
        f"stratified split {d['train_rows']} train / {d['test_rows']} test. "
        f"Target encoding: 0 = malignant, 1 = benign (positive class = {m['positive_class']}).",
        "",
        f"## Model selection ({m['selection_rule']})",
        "",
        "| Candidate | CV ROC-AUC (mean ± sd) | Train ROC-AUC | Grid size | Best params |",
        "|---|---|---|---|---|",
    ]
    for c in m["candidate_results"]:
        mark = " **(selected)**" if c["model"] == m["selected_model"] else ""
        lines.append(f"| {c['model']}{mark} | {c['best_cv_score']:.4f} ± {c['best_cv_score_std']:.4f} | "
                     f"{c['best_cv_train_score']:.4f} | {c['grid_size']} | `{c['best_params']}` |")
    lines += [
        "",
        f"## Held-out test set (evaluated once, threshold {m['training_config']['decision_threshold']})",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in met.items()],
        f"| recall, malignant class | {rep['0']['recall']:.4f} |",
        f"| precision, malignant class | {rep['0']['precision']:.4f} |",
        "",
        "Confusion matrix (rows = true, columns = predicted; order malignant, benign):",
        "",
        "| | pred malignant | pred benign |",
        "|---|---|---|",
        f"| true malignant | {cm[0][0]} | {cm[0][1]} |",
        f"| true benign | {cm[1][0]} | {cm[1][1]} |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny search grid, for smoke tests")
    args = parser.parse_args(argv)
    result = train_and_select(QUICK_CONFIG if args.quick else DEFAULT_CONFIG)
    print(report_markdown(result))
    print(f"Saved model to {result['artifact']} and reports to {_relative(REPORT_DIR)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
