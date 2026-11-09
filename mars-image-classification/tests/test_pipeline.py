import json
import os
from pathlib import Path

import numpy as np


def test_quick_run_writes_results_and_figures(quick_run):
    out = Path(os.environ["REPORT_DIR"])
    r = json.loads((out / "results.json").read_text())
    assert r["selected"]["model"] in {"small_cnn", "linear_probe", "finetuned_resnet18"}
    assert r["data"]["classes"] == ["ground", "wheel", "horizon", "drill"]  # 'sun' never seen: not a model class
    for name in ("samples", "model_comparison", "per_class_f1", "confusion_matrix", "by_instrument"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 5_000
    assert "## Test (scored once)" in (out / "results.md").read_text()


def test_models_beat_majority_on_easy_synthetic_data(quick_run):
    t = quick_run["test"]
    assert t["linear_probe"]["macro_f1"] > t["majority"]["macro_f1"]


def test_selection_never_sees_test_labels(monkeypatch, quick_run):
    """Scrambling the test labels must not change the selection or any validation score."""
    from mars_classifier import train

    real = train.load_splits

    def scrambled(**kw):
        s = real(**kw)
        labels = np.random.default_rng(1).permutation([i.label for i in s["test"]])
        for it, lab in zip(s["test"], labels):
            it.label = int(lab)
        return s

    monkeypatch.setattr(train, "load_splits", scrambled)
    other = train.run(train.QUICK)
    assert other["selected"] == quick_run["selected"]
    assert other["validation"] == quick_run["validation"]
