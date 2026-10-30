"""End-to-end model training and model selection.

    python -m ml_portfolio.train            # full grid (about a minute on a laptop)
    python -m ml_portfolio.train --quick    # tiny grid for smoke tests

Workflow: load and validate data -> stratified train/test split -> for each candidate,
GridSearchCV with stratified k-fold on the TRAINING split only -> pick the candidate with
the best mean CV ROC-AUC -> out-of-fold predictions on the TRAINING split for the threshold
trade-off and the operating-point rule -> evaluate the selected model once on the untouched
test split (with intervals) -> save the fitted Pipeline, metadata, JSON + Markdown report
and diagnostic figures.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

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
from . import diagnostics as dg
from .data import make_train_test_split
from .evaluate import evaluate_classifier
from .features import build_preprocessor
from .models import build_model_candidates

QUICK_CONFIG = replace(DEFAULT_CONFIG, cv_folds=3, logistic_c_values=(1.0,), rf_estimators=(50,),
                       rf_max_depth=(8,))


def train_and_select(
    config: TrainingConfig = DEFAULT_CONFIG,
    n_bootstrap: int = dg.N_BOOTSTRAP,
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

    # Threshold analysis on the training split only: each row predicted by a fold model
    # that didn't see it, with the selected hyperparameters and the selection folds.
    p_oof = dg.oof_probabilities(best_search.best_estimator_, bundle.X_train, bundle.y_train, cv)
    chosen = dg.choose_threshold(bundle.y_train, p_oof)
    # The test split, once, at two thresholds fixed before looking at it.
    p_test = best_search.best_estimator_.predict_proba(bundle.X_test)[:, 1]
    default_t = config.decision_threshold
    diagnostics = {
        "probability_meaning": "P(benign); a case is called malignant when P(benign) < threshold",
        "oof_training": {
            "roc_auc": float(dg.roc_auc_score(bundle.y_train, p_oof)),
            "brier_score": float(dg.brier_score_loss(bundle.y_train, p_oof)),
            "threshold_table": dg.threshold_table(bundle.y_train, p_oof),
            "calibration_bins": dg.calibration_bins(bundle.y_train, p_oof),
        },
        "chosen_threshold": chosen,
        "test_at_default": dg.test_summary(bundle.y_test, p_test, default_t, n_bootstrap),
        "test_at_chosen": dg.test_summary(bundle.y_test, p_test, chosen["threshold"], n_bootstrap),
        "test_calibration_bins": dg.calibration_bins(bundle.y_test, p_test),
        "bootstrap_replicates": n_bootstrap,
    }
    caption = (f"Breast Cancer Wisconsin (Diagnostic), scikit-learn copy; {len(bundle.y_train)} train / "
               f"{len(bundle.y_test)} test rows; model {best_name}; generated by python -m ml_portfolio.train")
    figures = dg.save_figures(REPORT_DIR / "figures", bundle.y_test, p_test, bundle.y_train, p_oof, default_t,
                              chosen, caption)
    diagnostics["figures"] = {k: _relative(Path(v)) for k, v in figures.items()}

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
        "diagnostics": diagnostics,
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
    dgn = m.get("diagnostics")
    if dgn:
        lines += _diagnostics_md(dgn, m)
    return "\n".join(lines)


def _ci(x: dict, pct: bool = False) -> str:
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
    counts = f" ({x['k']}/{x['n']})" if "k" in x else ""
    return f"{f(x['value'])}{counts} | {f(x['ci95'][0])} – {f(x['ci95'][1])} | {x['method']}"


def _bins(bins: list[dict]) -> str:
    return "; ".join("%.2f → %.2f (%d)" % (b["mean_predicted"], b["observed_benign_rate"], b["n"]) for b in bins)


def _diagnostics_md(d: dict, m: dict) -> list[str]:
    td, tc, ch = d["test_at_default"], d["test_at_chosen"], d["chosen_threshold"]
    lines = [
        "## Uncertainty on the test set",
        "",
        f"Probabilities are {d['probability_meaning']}. Test set: {td['n_test']} rows "
        f"({td['n_malignant']} malignant, {td['n_benign']} benign). Intervals are 95%: Wilson score for "
        f"count proportions, and a stratified bootstrap ({d['bootstrap_replicates']} resamples of the test set, "
        "model held fixed) for the rest. They reflect test-set sampling only, not variation from "
        "re-training, and with 42 malignant cases one error moves malignant recall by about 2.4 points.",
        "",
        f"| Metric (threshold {td['threshold']:g}) | Value | 95% CI | Method |",
        "|---|---|---|---|",
        f"| ROC-AUC | {_ci(td['roc_auc'])} |",
        f"| Average precision, malignant | {_ci(td['average_precision_malignant'])} |",
        f"| Brier score (P(benign)) | {_ci(td['brier_score'])} |",
        f"| Accuracy | {_ci(td['accuracy'], True)} |",
        f"| Malignant recall (sensitivity) | {_ci(td['malignant_recall'], True)} |",
        f"| Malignant precision | {_ci(td['malignant_precision'], True)} |",
        f"| Benign recall (specificity for malignant) | {_ci(td['benign_recall'], True)} |",
        "",
        "## Threshold trade-off (training split, out-of-fold)",
        "",
        "Computed from out-of-fold predictions on the training split, never from the test set.",
        "",
        "| Threshold on P(benign) | Malignant caught | Malignant missed | Benign flagged | Malignant recall | Malignant precision |",
        "|---|---|---|---|---|---|",
        *[f"| {r['threshold']:g} | {r['malignant_caught']} | {r['malignant_missed']} | {r['benign_flagged']} | "
          f"{r['malignant_recall']:.3f} | {r['malignant_precision']:.3f} |" for r in d["oof_training"]["threshold_table"]],
        "",
        f"Operating-point rule: {ch['rule']}. It gives threshold **{ch['threshold']:.3f}** "
        f"(out-of-fold: recall {ch['malignant_recall']:.3f}, precision {ch['malignant_precision']:.3f}). "
        "Evaluated once on the test set:",
        "",
        "| Threshold | Malignant recall | 95% CI | | Malignant precision | 95% CI | |",
        "|---|---|---|---|---|---|---|",
        f"| {td['threshold']:g} (default) | {_ci(td['malignant_recall'], True)} | {_ci(td['malignant_precision'], True)} |",
        f"| {tc['threshold']:.3f} (CV rule) | {_ci(tc['malignant_recall'], True)} | {_ci(tc['malignant_precision'], True)} |",
        "",
        f"The saved model and the API use the default threshold {td['threshold']:g}.",
        "",
        "## Calibration",
        "",
        "No calibrator is fitted; calibration is checked, not adjusted. Mean predicted P(benign) vs observed "
        "benign rate in 5 quantile bins:",
        "",
        "| Split | Bins (predicted → observed, n) | Brier |",
        "|---|---|---|",
        f"| training, out-of-fold | {_bins(d['oof_training']['calibration_bins'])} | {d['oof_training']['brier_score']:.3f} |",
        f"| test | {_bins(d['test_calibration_bins'])} | {td['brier_score']['value']:.3f} |",
        "",
        "Figures: " + ", ".join(f"`{v}`" for v in d.get("figures", {}).values()),
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny search grid, for smoke tests")
    args = parser.parse_args(argv)
    result = train_and_select(QUICK_CONFIG, n_bootstrap=200) if args.quick else train_and_select(DEFAULT_CONFIG)
    print(report_markdown(result))
    print(f"Saved model to {result['artifact']} and reports to {_relative(REPORT_DIR)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
