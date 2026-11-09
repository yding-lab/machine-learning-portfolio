import json
import os
from pathlib import Path

import numpy as np


def test_quick_run_writes_results_and_figures(quick_run):
    out = Path(os.environ["REPORT_DIR"])
    r = json.loads((out / "results.json").read_text())
    assert r["selected"]["model"] in {"unet", "lraspp_mobilenet_v3"}
    assert set(r["test"]) == {"min1", "min2", "min3"}
    for name in ("examples", "per_class_iou", "confusion_matrix", "miou_by_agreement", "training_curves"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 5_000
    assert "scored once at full resolution" in (out / "results.md").read_text()


def test_stricter_agreement_scores_fewer_pixels(quick_run):
    px = {lvl: sum(quick_run["test"][lvl]["majority"]["pixels"].values()) for lvl in ("min1", "min2", "min3")}
    assert px["min1"] > px["min2"] > px["min3"]


def test_learned_models_beat_majority_on_easy_synthetic_data(quick_run):
    t = quick_run["test"]["min2"]
    assert max(t["unet"]["miou"], t["lraspp_mobilenet_v3"]["miou"]) > t["majority"]["miou"]


def test_selection_never_sees_test_labels(monkeypatch, quick_run):
    """Replacing every test label must leave the selection and the validation scores unchanged."""
    from terrain_seg import train

    monkeypatch.setattr(train, "read_label", lambda p: np.zeros((64, 64), np.uint8))
    other = train.run(train.QUICK)
    assert other["selected"] == quick_run["selected"]
    assert other["validation_miou"] == quick_run["validation_miou"]
