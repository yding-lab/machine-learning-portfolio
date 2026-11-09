import numpy as np
import pytest

from mortgage_approval.evaluate import at_threshold, best_f1_threshold, wilson
from mortgage_approval.models import MLPModel


def test_wilson_interval():
    lo, hi = wilson(41, 42)
    assert lo == pytest.approx(0.877, abs=1e-3) and hi == pytest.approx(0.996, abs=1e-3)


def test_threshold_counts():
    a = at_threshold(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (a["tp"], a["fn"], a["fp"], a["tn"]) == (1, 1, 1, 1) and a["fpr"] == 0.5


def test_best_f1_threshold_separates_easy_data():
    y = np.repeat([0, 1], [900, 100])
    p = np.where(y == 1, 0.8, 0.1) + np.random.default_rng(0).normal(0, 0.01, 1000)
    t = best_f1_threshold(y, p)
    assert 0.1 < t <= 0.8 and at_threshold(y, p, t)["f1"] == 1.0


def test_mlp_learns_and_keeps_best_epoch():
    rng = np.random.default_rng(0)
    num = rng.normal(size=(3000, 3)).astype(np.float32)
    cat = rng.integers(0, 3, size=(3000, 1)).astype(np.int64)
    y = ((num[:, 0] + (cat[:, 0] == 2)) > 1).astype(int)
    val = (num[:600], cat[:600], y[:600])
    m = MLPModel([3], hidden=(16,), dropout=0.0, batch_size=64, max_epochs=15, patience=4, seed=0).fit(num[600:], cat[600:], y[600:], val=val)
    p = m.predict_proba(*val[:2])
    assert ((p >= 0) & (p <= 1)).all()
    best = max(r["val_average_precision"] for r in m.history)
    assert best > 0.9 and m.history[m.best_epoch - 1]["val_average_precision"] == best
