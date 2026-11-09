"""Segmentation metrics from per-image confusion matrices (labelled pixels only)."""

from __future__ import annotations

import numpy as np

from .config import CLASSES, IGNORE

K = len(CLASSES)


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    m = y != IGNORE
    return np.bincount(y[m].astype(np.int64) * K + pred[m], minlength=K * K).reshape(K, K)


def scores(cm: np.ndarray) -> dict:
    """cm[true, pred]. IoU is averaged over classes that occur in the labels."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    union = support + predicted - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    present = support > 0
    return {"miou": float(np.nanmean(iou[present])), "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "iou": {c: (None if np.isnan(v) else float(v)) for c, v in zip(CLASSES, iou)},
            "recall": {c: (float(tp[i] / support[i]) if support[i] else None) for i, c in enumerate(CLASSES)},
            "pixels": {c: int(support[i]) for i, c in enumerate(CLASSES)}}


def bootstrap(per_image: dict[str, np.ndarray], selected: str, n_boot: int, seed: int = 0) -> dict:
    """Resample test images; mIoU recomputed from summed confusions. Paired differences vs selected."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_image.values())))
    draws = {m: [] for m in per_image}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for m, cms in per_image.items():
            draws[m].append(scores(cms[idx].sum(0))["miou"])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    sel = np.array(draws[selected])
    return {"miou_ci95": {m: ci(v) for m, v in draws.items()},
            "miou_difference_vs": {m: {"mean": float(np.mean(sel - np.array(v))), "ci95": ci(sel - np.array(v))}
                                   for m, v in draws.items() if m != selected}}
