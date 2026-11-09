import numpy as np

from mars_classifier.evaluate import bootstrap, by_group, metrics


def test_metrics_only_count_classes_present():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 2])
    m = metrics(y, pred, None, [0, 1])
    assert m["accuracy"] == 0.75 and m["balanced_accuracy"] == 0.75
    assert abs(m["macro_f1"] - np.mean([1.0, 2 / 3])) < 1e-9  # class 2 isn't in y, so it isn't averaged in


def test_top3_accuracy():
    prob = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
    assert metrics(np.array([2, 0]), np.array([0, 3]), prob, [0, 2])["top3_accuracy"] == 0.5


def test_bootstrap_interval_contains_point_and_groups_add_up():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.7, y, rng.integers(0, 4, 400))
    b = bootstrap(y, {"a": pred, "b": rng.integers(0, 4, 400)}, "a", [0, 1, 2, 3], 200)
    lo, hi = b["selected"]["accuracy"]
    assert lo <= float(np.mean(y == pred)) <= hi and b["macro_f1_difference_vs"]["b"]["ci95"][0] > 0
    g = by_group(y, pred, np.array(["x", "y"] * 200), [0, 1, 2, 3])
    assert sum(r["n"] for r in g) == 400
