"""Multi-class metrics restricted to the classes present in the evaluated split."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def metrics(y, pred, prob, present) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    rec = [float(np.mean(pred[y == c] == c)) for c in present]
    out = {"accuracy": float(np.mean(y == pred)),
           "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
           "balanced_accuracy": float(np.mean(rec)), "n": int(len(y))}
    if prob is not None:
        top3 = np.argsort(-prob, axis=1)[:, :3]
        out["top3_accuracy"] = float(np.mean([y[i] in top3[i] for i in range(len(y))]))
    return out


def per_class(y, pred, labels) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(labels))), zero_division=0)
    return [{"class": labels[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(len(labels)) if s[i] > 0]


def confusion(y, pred, K, present) -> dict:
    m = np.zeros((K, K), dtype=int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return {"matrix": m.tolist(), "rows_present": list(present)}


def by_group(y, pred, groups, present) -> list[dict]:
    out = []
    for g in sorted(set(groups)):
        mask = groups == g
        pres = sorted(set(np.asarray(y)[mask].tolist()))
        out.append({"group": g, **metrics(np.asarray(y)[mask], np.asarray(pred)[mask], None, pres)})
    return out


def bootstrap(y, preds: dict, selected: str, present, n_boot: int, seed: int = 0) -> dict:
    """95% percentile intervals by resampling test images (models fixed); paired macro-F1 differences."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    draws = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    diffs = {k: [] for k in preds if k != selected}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        pres = sorted(set(y[i].tolist()))
        m = metrics(y[i], preds[selected][i], None, pres)
        for k in draws:
            draws[k].append(m[k])
        for k in diffs:
            diffs[k].append(m["macro_f1"] - metrics(y[i], preds[k][i], None, pres)["macro_f1"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"selected": {k: ci(v) for k, v in draws.items()},
            "macro_f1_difference_vs": {k: {"mean": float(np.mean(v)), "ci95": ci(v)} for k, v in diffs.items()}}
