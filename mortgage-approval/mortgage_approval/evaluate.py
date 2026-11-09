"""Metrics, thresholds, bootstrap intervals and error analysis."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def scores(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"average_precision": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y))}


def at_threshold(y, p, t) -> dict:
    y, flag = np.asarray(y), np.asarray(p) >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if tp else 0.0, "flag_rate": float(flag.mean()),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def best_f1_threshold(y, p) -> float:
    """Threshold with the highest F1 for 'denied' (chosen on validation data only)."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200)))
    return float(max(grid, key=lambda t: at_threshold(y, p, t)["f1"]))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def bootstrap(y, probs: dict, selected: str, threshold: float, n_boot: int, seed: int = 0) -> dict:
    """Percentile 95% intervals by resampling test applications (models fixed). Includes paired
    differences in average precision between the selected model and each other model."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    keys = ("average_precision", "roc_auc", "brier")
    draws = {k: [] for k in keys} | {"recall": [], "precision": []}
    diffs = {m: [] for m in probs if m != selected}
    p_sel = probs[selected]
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        s = scores(y[i], p_sel[i])
        for k in keys:
            draws[k].append(s[k])
        a = at_threshold(y[i], p_sel[i], threshold)
        draws["recall"].append(a["recall"])
        draws["precision"].append(a["precision"])
        for m in diffs:
            diffs[m].append(s["average_precision"] - average_precision_score(y[i], probs[m][i]))
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"selected": {k: ci(v) for k, v in draws.items()},
            "ap_difference_vs": {m: {"ci95": ci(v), "mean": float(np.mean(v))} for m, v in diffs.items()}}


def by_group(y, p, groups, threshold, min_n: int = 1) -> list[dict]:
    """Per group: size, actual rate, flag rate, recall (TPR), FPR with Wilson intervals, AP."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rows = []
    for g in sorted(set(groups), key=lambda g: -(groups == g).sum()):
        m = groups == g
        if m.sum() < min_n:
            continue
        a = at_threshold(y[m], p[m], threshold)
        pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
        rows.append({"group": str(g), "n": int(m.sum()), "denial_rate": float(y[m].mean()),
                     "mean_predicted": float(p[m].mean()), "flag_rate": a["flag_rate"],
                     "recall": a["recall"], "recall_ci95": wilson(a["tp"], pos),
                     "fpr": a["fpr"], "fpr_ci95": wilson(a["fp"], neg), "precision": a["precision"],
                     "average_precision": float(average_precision_score(y[m], p[m])) if 0 < pos < m.sum() else float("nan")})
    return rows


def calibration_bins(y, p, n_bins: int = 10) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [{"mean_predicted": float(p[which == b].mean()), "observed": float(y[which == b].mean()), "n": int((which == b).sum())}
            for b in range(len(edges) - 1) if (which == b).any()]
