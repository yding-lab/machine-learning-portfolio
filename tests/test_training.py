import json


def test_training_selects_by_cv_and_writes_artifacts(trained):
    m, tmp = trained["metadata"], trained["dir"]
    assert m["selected_model"] in {"logistic_regression", "random_forest"}
    best = max(m["candidate_results"], key=lambda c: c["best_cv_score"])
    assert best["model"] == m["selected_model"]  # selection uses CV, never the test split
    assert (tmp / "artifacts" / "best_model.joblib").exists()
    meta = json.loads((tmp / "artifacts" / "model_metadata.json").read_text())
    assert meta["positive_class"] == "benign" and len(meta["feature_names"]) == 30
    assert (tmp / "reports" / "training_report.md").read_text().startswith("# Training report")
    assert m["holdout_results"]["metrics"]["roc_auc"] > 0.9  # sanity floor, not a claim
