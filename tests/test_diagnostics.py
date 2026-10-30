"""Diagnostics: intervals, thresholds and figures, and that none of it is tuned on the test set."""

import numpy as np
import pytest

from ml_portfolio import diagnostics as dg


def test_confusion_counts_treat_malignant_as_class_of_interest():
    y = np.array([0, 0, 1, 1])  # 0 = malignant, 1 = benign
    p = np.array([0.1, 0.7, 0.2, 0.9])  # P(benign)
    c = dg.confusion_at(y, p, 0.5)
    assert (c["malignant_caught"], c["malignant_missed"], c["benign_flagged"], c["benign_cleared"]) == (1, 1, 1, 1)


def test_wilson_matches_known_values():
    lo, hi = dg.wilson(41, 42)
    assert lo == pytest.approx(0.877, abs=1e-3) and hi == pytest.approx(0.996, abs=1e-3)
    assert dg.wilson(0, 10)[0] == 0.0


def test_threshold_rule_reaches_target_and_is_lowest():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    ch = dg.choose_threshold(y, p, 0.95)
    assert ch["malignant_recall"] >= 0.95
    lower = dg.confusion_at(y, p, ch["threshold"] - 1e-3)
    assert lower["malignant_recall"] < 0.95 or lower["benign_flagged"] == ch["benign_flagged"]


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], [40, 70])
    p = np.clip(y * 0.7 + rng.normal(0.15, 0.15, y.size), 0, 1)
    s = dg.test_summary(y, p, 0.5, n_boot=300)
    for k in ("roc_auc", "brier_score", "accuracy", "average_precision_malignant"):
        lo, hi = s[k]["ci95"]
        assert lo <= s[k]["value"] <= hi


def test_training_writes_diagnostics_and_figures(trained):
    d, tmp = trained["metadata"]["diagnostics"], trained["dir"]
    for name in ("confusion_matrix", "roc_pr_curves", "calibration", "threshold_tradeoff"):
        assert (tmp / "reports" / "figures" / f"{name}.png").stat().st_size > 10_000
    assert d["test_at_default"]["n_test"] == 114 and d["test_at_default"]["n_malignant"] == 42
    cm = trained["metadata"]["holdout_results"]["confusion_matrix"]
    rec = d["test_at_default"]["malignant_recall"]
    assert (rec["k"], rec["n"]) == (cm[0][0], cm[0][0] + cm[0][1])  # consistent with sklearn's matrix
    md = (tmp / "reports" / "training_report.md").read_text()
    assert "## Threshold trade-off (training split, out-of-fold)" in md and "Wilson" in md


def test_threshold_choice_never_sees_test_labels(monkeypatch):
    """Scrambling the test labels must not change the threshold or the OOF table."""
    from ml_portfolio import data, train

    real = data.make_train_test_split

    def scrambled(config):
        b = real(config)
        b.y_test = b.y_test.sample(frac=1, random_state=3).set_axis(b.y_test.index)
        return b

    cfg = train.QUICK_CONFIG
    a = train.train_and_select(cfg, n_bootstrap=50)["diagnostics"]
    monkeypatch.setattr(train, "make_train_test_split", scrambled)
    b = train.train_and_select(cfg, n_bootstrap=50)["diagnostics"]
    assert a["chosen_threshold"] == b["chosen_threshold"]
    assert a["oof_training"] == b["oof_training"]
    assert a["test_at_default"]["roc_auc"]["value"] != b["test_at_default"]["roc_auc"]["value"]
