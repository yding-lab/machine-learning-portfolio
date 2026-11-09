import json
from pathlib import Path

import numpy as np


def test_quick_run_writes_results_report_and_figures(quick_run):
    out = Path(__import__("os").environ["REPORT_DIR"])
    assert json.loads((out / "results.json").read_text())["selected"]["model"] == quick_run["selected"]["model"]
    assert "## Fairness audit" in (out / "results.md").read_text()
    for name in ("pr_curves", "calibration", "fairness_race", "importance", "error_by_dti"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 10_000


def test_selected_model_beats_the_prior(quick_run):
    t = quick_run["test"]
    assert t[quick_run["selected"]["model"]]["average_precision"] > 2 * t["prior"]["average_precision"]


def test_selection_and_threshold_never_see_test_labels(monkeypatch, quick_run):
    """Scrambling the 2024 labels must not change the selected model or its threshold."""
    from mortgage_approval import train
    from mortgage_approval.config import QUICK

    real = train.make_splits

    def scrambled(config):
        s = real(config)
        s.test = s.test.assign(denied=np.random.default_rng(1).permutation(s.test["denied"].to_numpy()))
        return s

    monkeypatch.setattr(train, "make_splits", scrambled)
    other = train.run(QUICK)
    assert other["selected"]["model"] == quick_run["selected"]["model"]
    assert other["selected"]["threshold"] == quick_run["selected"]["threshold"]
    assert other["search"] == quick_run["search"]
    sel = quick_run["selected"]["model"]
    assert other["test"][sel]["average_precision"] != quick_run["test"][sel]["average_precision"]
