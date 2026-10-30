"""Model diagnostics beyond a single score: uncertainty, thresholds, calibration, figures.

Encoding reminder: the target is 0 = malignant, 1 = benign, and the model's probability is
P(benign). The clinically relevant error is a malignant tumour called benign, so the
threshold analysis is phrased in terms of malignant recall (sensitivity) and malignant
precision. A case is called malignant when P(benign) < threshold.

Isolation rules this module follows:
  - The threshold trade-off table and any chosen operating threshold come from
    out-of-fold (OOF) predictions on the TRAINING split only (cross_val_predict with the
    selected hyperparameters and the same folds as model selection).
  - The test split is touched once, by the already-selected model, to report metrics,
    intervals and figures. Nothing is tuned on it.
  - No probability calibrator is fitted. Calibration is assessed (reliability curve,
    Brier score), not adjusted.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict

THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_MALIGNANT_RECALL = 0.98  # operating-point rule, applied to OOF training predictions
N_BOOTSTRAP = 2000

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def oof_probabilities(estimator, X, y, cv) -> np.ndarray:
    """P(benign) for every training row, each predicted by a model that didn't see it."""
    return cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def confusion_at(y, p_benign, threshold: float) -> dict:
    """Counts and rates with malignant as the class of interest."""
    y = np.asarray(y)
    called_malignant = p_benign < threshold
    malignant = y == 0
    tp = int(np.sum(called_malignant & malignant))  # malignant caught
    fn = int(np.sum(~called_malignant & malignant))  # malignant called benign (the costly error)
    fp = int(np.sum(called_malignant & ~malignant))  # benign flagged as malignant
    tn = int(np.sum(~called_malignant & ~malignant))
    return {
        "threshold": threshold,
        "malignant_caught": tp, "malignant_missed": fn, "benign_flagged": fp, "benign_cleared": tn,
        "malignant_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "malignant_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "benign_recall": tn / (tn + fp) if tn + fp else float("nan"),
    }


def threshold_table(y, p_benign, grid=THRESHOLD_GRID) -> list[dict]:
    return [confusion_at(y, p_benign, t) for t in grid]


def choose_threshold(y, p_benign, min_recall: float = MIN_MALIGNANT_RECALL) -> dict:
    """Lowest P(benign) threshold whose OOF malignant recall reaches min_recall.

    Lower thresholds flag fewer cases, so the lowest qualifying one keeps the most benign
    cases cleared. Candidates are the distinct OOF probabilities (plus 0.5).
    """
    candidates = sorted(set(np.round(p_benign, 6)) | {0.5})
    for t in candidates:
        t_next = float(np.nextafter(t, 1))  # "< threshold" must include ties at t
        row = confusion_at(y, p_benign, t_next)
        if row["malignant_recall"] >= min_recall:
            row["threshold"] = round(t_next, 6)
            row["rule"] = (f"lowest threshold with out-of-fold malignant recall >= {min_recall:.2f} "
                           "on the training split")
            return row
    row = confusion_at(y, p_benign, 1.0)
    row["rule"] = "no threshold reached the recall target; flags everything"
    return row


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(y, p_benign, threshold: float, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Percentile 95% intervals from a stratified bootstrap of the test set.

    Resampling within each class keeps both classes present in every replicate. This
    covers sampling variability of the test set only: the model is held fixed, so
    variability from re-training on different data isn't included.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    stats = {"roc_auc": [], "average_precision_malignant": [], "brier_score": [], "accuracy": []}
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(idx0, idx0.size), rng.choice(idx1, idx1.size)])
        yb, pb = y[i], p_benign[i]
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["average_precision_malignant"].append(average_precision_score(1 - yb, 1 - pb))
        stats["brier_score"].append(brier_score_loss(yb, pb))
        stats["accuracy"].append(np.mean((pb >= threshold).astype(int) == yb))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def test_summary(y, p_benign, threshold: float, n_boot: int = N_BOOTSTRAP) -> dict:
    """Point estimates with intervals: Wilson for count proportions, bootstrap for the rest."""
    y = np.asarray(y)
    c = confusion_at(y, p_benign, threshold)
    n_mal = c["malignant_caught"] + c["malignant_missed"]
    n_flag = c["malignant_caught"] + c["benign_flagged"]
    n_ben = c["benign_cleared"] + c["benign_flagged"]
    boot = bootstrap_ci(y, p_benign, threshold, n_boot)
    return {
        "n_test": int(len(y)), "n_malignant": n_mal, "n_benign": n_ben, "threshold": threshold,
        "roc_auc": {"value": float(roc_auc_score(y, p_benign)), "ci95": boot["roc_auc"], "method": "stratified bootstrap"},
        "average_precision_malignant": {"value": float(average_precision_score(1 - y, 1 - p_benign)),
                                        "ci95": boot["average_precision_malignant"], "method": "stratified bootstrap"},
        "brier_score": {"value": float(brier_score_loss(y, p_benign)), "ci95": boot["brier_score"],
                        "method": "stratified bootstrap"},
        "accuracy": {"value": float(np.mean((p_benign >= threshold).astype(int) == y)), "ci95": boot["accuracy"],
                     "method": "stratified bootstrap"},
        "malignant_recall": {"value": c["malignant_recall"], "k": c["malignant_caught"], "n": n_mal,
                             "ci95": list(wilson(c["malignant_caught"], n_mal)), "method": "Wilson"},
        "malignant_precision": {"value": c["malignant_precision"], "k": c["malignant_caught"], "n": n_flag,
                                "ci95": list(wilson(c["malignant_caught"], n_flag)), "method": "Wilson"},
        "benign_recall": {"value": c["benign_recall"], "k": c["benign_cleared"], "n": n_ben,
                          "ci95": list(wilson(c["benign_cleared"], n_ben)), "method": "Wilson"},
    }


def calibration_bins(y, p_benign, n_bins: int = 5) -> list[dict]:
    """Quantile bins: mean predicted P(benign) vs observed benign rate, with bin sizes."""
    y = np.asarray(y)
    edges = np.unique(np.quantile(p_benign, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, p_benign, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            out.append({"mean_predicted": float(p_benign[m].mean()), "observed_benign_rate": float(y[m].mean()),
                        "n": int(m.sum())})
    return out


# ---------------------------------------------------------------- figures

def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save_figures(out_dir: Path, y_test, p_test, y_train, p_oof, threshold: float, chosen: dict,
                 caption: str) -> dict[str, str]:
    """Write the four diagnostic PNGs and return {name: path}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    y_test, y_train = np.asarray(y_test), np.asarray(y_train)
    paths = {}

    def finish(fig, name):
        import textwrap

        width = int(fig.get_size_inches()[0] * 17)  # characters that fit at 7 pt
        text = "\n".join(textwrap.wrap(caption, width))
        fig.text(0.01, 0.01, text, fontsize=7, color=INK2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.03 + 0.03 * text.count("\n") + 0.02, 1, 1))
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths[name] = str(p)

    # 1. confusion matrix on the test split
    c = confusion_at(y_test, p_test, threshold)
    cm = np.array([[c["malignant_caught"], c["malignant_missed"]], [c["benign_flagged"], c["benign_cleared"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), facecolor=SURFACE)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef4fc", "#256abf"]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["called malignant", "called benign"])
    ax.set_yticks([0, 1], ["malignant", "benign"])
    ax.set_xlabel(f"predicted (malignant if P(benign) < {threshold:g})", color=INK2, fontsize=9)
    ax.set_ylabel("true class", color=INK2, fontsize=9)
    ax.set_title(f"Test confusion matrix (n = {len(y_test)})", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)
    finish(fig, "confusion_matrix")

    # 2. ROC and PR (malignant as the class of interest)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8), facecolor=SURFACE)
    for yy, pp, color, label in ((y_train, p_oof, ORANGE, "training, out-of-fold"), (y_test, p_test, BLUE, "test")):
        fpr, tpr, _ = roc_curve(1 - yy, 1 - pp)
        a1.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {roc_auc_score(yy, pp):.3f})")
        prec, rec, _ = precision_recall_curve(1 - yy, 1 - pp)
        a2.plot(rec, prec, color=color, lw=2, drawstyle="steps-post",
                label=f"{label} (AP {average_precision_score(1 - yy, 1 - pp):.3f})")
    a1.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    _style(a1, "ROC curve", "false positive rate (benign flagged)", "malignant recall")
    _style(a2, "Precision-recall, malignant", "malignant recall", "malignant precision")
    a2.axhline(np.mean(y_test == 0), color=GRID, lw=1, ls="--")
    for a in (a1, a2):
        a.legend(fontsize=8, frameon=False, loc="lower right" if a is a1 else "lower left")
        a.set_xlim(-0.01, 1.01)
        a.set_ylim(-0.01, 1.02)
    finish(fig, "roc_pr_curves")

    # 3. calibration (reliability) on P(benign)
    fig, ax = plt.subplots(figsize=(4.4, 3.9), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for yy, pp, color, label, marker in ((y_train, p_oof, ORANGE, "training, out-of-fold", "s"),
                                         (y_test, p_test, BLUE, "test", "o")):
        bins = calibration_bins(yy, pp, 5)
        ax.plot([b["mean_predicted"] for b in bins], [b["observed_benign_rate"] for b in bins], color=color,
                lw=2, marker=marker, ms=6, label=f"{label} (Brier {brier_score_loss(yy, pp):.3f})")
    _style(ax, "Calibration of P(benign)", "mean predicted P(benign), 5 quantile bins", "observed benign rate")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    finish(fig, "calibration")

    # 4. threshold trade-off from OOF training predictions
    ts = np.linspace(0.05, 0.95, 91)
    rows = [confusion_at(y_train, p_oof, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor=SURFACE)
    ax.plot(ts, [r["malignant_recall"] for r in rows], color=BLUE, lw=2, label="malignant recall")
    ax.plot(ts, [r["malignant_precision"] for r in rows], color=ORANGE, lw=2, label="malignant precision")
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.text(0.51, 0.04, "default 0.5", fontsize=7, color=INK2, ha="left", transform=ax.get_xaxis_transform())
    ax.axvline(chosen["threshold"], color=INK2, lw=1, ls="--")
    ax.text(chosen["threshold"] - 0.01, 0.04, f"CV rule: recall ≥ {MIN_MALIGNANT_RECALL:.2f} ({chosen['threshold']:.3f})",
            fontsize=7, color=INK2, ha="right", transform=ax.get_xaxis_transform())
    _style(ax, "Threshold trade-off (training split, out-of-fold)", "threshold on P(benign): malignant if below",
           "rate")
    ax.set_ylim(0.6, 1.01)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    finish(fig, "threshold_tradeoff")
    return paths
