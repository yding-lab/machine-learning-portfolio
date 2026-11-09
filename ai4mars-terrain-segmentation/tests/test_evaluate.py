import numpy as np

from terrain_seg.evaluate import bootstrap, confusion, scores


def test_iou_ignores_unlabelled_pixels_and_absent_classes():
    y = np.array([[0, 0, 1, 255]], np.uint8)
    p = np.array([[0, 1, 1, 3]], np.uint8)
    s = scores(confusion(y, p))
    assert s["iou"]["soil"] == 0.5 and s["iou"]["bedrock"] == 0.5
    assert s["miou"] == 0.5 and s["pixel_accuracy"] == 2 / 3  # sand and big rock absent from labels: not averaged
    assert s["iou"]["big rock"] is None


def test_bootstrap_intervals_and_paired_differences():
    rng = np.random.default_rng(0)
    good = np.stack([np.diag(rng.integers(50, 100, 4)) + rng.integers(0, 5, (4, 4)) for _ in range(30)])
    bad = np.stack([rng.integers(0, 50, (4, 4)) for _ in range(30)])
    b = bootstrap({"good": good, "bad": bad}, "good", 200)
    lo, hi = b["miou_ci95"]["good"]
    assert lo <= scores(good.sum(0))["miou"] <= hi
    assert b["miou_difference_vs"]["bad"]["ci95"][0] > 0
