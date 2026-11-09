"""Train, select and evaluate denial models; audit them by demographic group.

    python -m mortgage_approval.train            # full run, a few minutes on CPU
    python -m mortgage_approval.train --quick    # small version for tests and CI

1. Preprocessing is fitted on the 2023 training split only.
2. Each model family's settings are chosen by validation average precision (2023 hold-out).
3. The model with the best validation average precision is selected, and its decision
   threshold (best F1 for "denied") is chosen on validation.
4. The 2024 test split is scored once: every model, bootstrap intervals, paired differences,
   the fairness audit and the error analysis.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

from . import __version__
from .config import CASE_ROOT, CATEGORICAL, DEFAULT, NUMERIC, PROTECTED, QUICK, REPORT_DIR, SOURCE_PATH, Config
from .data import make_splits
from .evaluate import at_threshold, best_f1_threshold, bootstrap, by_group, calibration_bins, scores
from .features import Preprocessor, raw_features
from .models import BoostingModel, LogisticModel, MLPModel, PriorModel

torch.set_num_threads(2)


def _arrays(pre: Preprocessor, df):
    return pre.numeric(df), pre.categorical(df), df["denied"].to_numpy()


def select(config: Config, card, tr, va) -> tuple[dict, list[dict]]:
    """Fit every candidate on train, score on validation; return the best of each family."""
    candidates = [("prior", {}, lambda: PriorModel())]
    candidates += [("logistic_regression", {"C": c}, lambda c=c: LogisticModel(card, C=c, seed=config.seed))
                   for c in config.lr_c_values]
    candidates += [("gradient_boosting", {"learning_rate": lr, "max_leaf_nodes": ml},
                    lambda lr=lr, ml=ml: BoostingModel(card, lr, ml, seed=config.seed))
                   for lr in config.hgb_learning_rates for ml in config.hgb_max_leaf_nodes]
    candidates += [("pytorch_mlp", {"hidden": list(h), "dropout": d},
                    lambda h=h, d=d: MLPModel(card, h, d, config.mlp_lr, config.mlp_batch_size, config.mlp_max_epochs,
                                              config.mlp_patience, seed=config.seed))
                   for h in config.mlp_hidden for d in config.mlp_dropout]
    log, best = [], {}
    for family, params, make in candidates:
        model = make().fit(*tr, val=va)
        s = scores(va[2], model.predict_proba(va[0], va[1]))
        row = {"model": family, "params": params, "val_average_precision": s["average_precision"],
               "val_roc_auc": s["roc_auc"], "val_brier": s["brier"]}
        if isinstance(model, MLPModel):
            row["best_epoch"] = model.best_epoch
        log.append(row)
        if family not in best or s["average_precision"] > best[family][1]["val_average_precision"]:
            best[family] = (model, row)
    return best, log


def permutation_importance(model, num, cat, y, seed=0, repeats=3) -> list[dict]:
    """Drop in validation average precision when one input column is shuffled."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(num, cat))
    out = []
    for kind, names, arr in (("numeric", NUMERIC, num), ("categorical", CATEGORICAL, cat)):
        for j, name in enumerate(names):
            drops = []
            for _ in range(repeats):
                a = arr.copy()
                a[:, j] = a[rng.permutation(len(a)), j]
                p = model.predict_proba(a, cat) if kind == "numeric" else model.predict_proba(num, a)
                drops.append(base - average_precision_score(y, p))
            out.append({"feature": name, "ap_drop": float(np.mean(drops))})
    return sorted(out, key=lambda r: -r["ap_drop"])


def error_analysis(df, y, p, t) -> dict:
    f = raw_features(df)
    bands = {
        "dti": (f["dti"], [0, 30, 36, 43, 50, 100], ["<30", "30-36", "36-43", "43-50", "50+"]),
        "cltv": (f["cltv"], [0, 60, 80, 90, 95, 1000], ["<60", "60-80", "80-90", "90-95", "95+"]),
    }
    out = {}
    import pandas as pd

    for name, (values, edges, labels) in bands.items():
        g = pd.cut(values, edges, labels=labels, right=False).astype(str)
        out[name] = by_group(y, p, g.to_numpy(), t)
    out["loan_type"] = by_group(y, p, df["loan_type"].map({"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA/FSA"})
                                .fillna("other").to_numpy(), t)
    return out


def run(config: Config = DEFAULT) -> dict:
    s = make_splits(config)
    pre = Preprocessor().fit(s.train)
    card = [len(pre.vocab[c]) for c in CATEGORICAL]
    tr, va, te = _arrays(pre, s.train), _arrays(pre, s.validation), _arrays(pre, s.test)

    best, search_log = select(config, card, tr, va)
    selected = max((f for f in best if f != "prior"), key=lambda f: best[f][1]["val_average_precision"])
    model = best[selected][0]
    threshold = best_f1_threshold(va[2], model.predict_proba(va[0], va[1]))
    val_at_t = at_threshold(va[2], model.predict_proba(va[0], va[1]), threshold)

    # ---- the test split, once ----
    probs = {f: m.predict_proba(te[0], te[1]) for f, (m, _) in best.items()}
    y = te[2]
    test = {f: {**scores(y, p), "at_threshold": at_threshold(y, p, best_f1_threshold(va[2], best[f][0].predict_proba(va[0], va[1])))
                if f != "prior" else None} for f, p in probs.items()}
    p_sel = probs[selected]
    boot = bootstrap(y, {f: p for f, p in probs.items() if f != "prior"}, selected, threshold, config.n_bootstrap, config.seed)

    fairness = {}
    for attr in PROTECTED:
        g = s.test[attr].to_numpy()
        fairness[attr] = by_group(y, p_sel, g, threshold, min_n=100)
    importance = permutation_importance(model, *va, seed=config.seed)
    errors = error_analysis(s.test, y, p_sel, threshold)

    source = json.loads(SOURCE_PATH.read_text()) if SOURCE_PATH.exists() else {}
    result = {
        "case": "mortgage-approval", "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "torch": torch.__version__},
        "config": {k: v for k, v in config.__dict__.items()},
        "data": {"snapshot_sha256": source.get("snapshot_sha256"), "retrieved_utc": source.get("retrieved_utc"),
                 "source_counts": source.get("counts"), "excluded_incomplete": s.excluded,
                 "n_train": len(s.train), "n_validation": len(s.validation), "n_test": len(s.test),
                 "denial_rate": {"train": float(tr[2].mean()), "validation": float(va[2].mean()), "test": float(y.mean())}},
        "features": {"numeric": list(NUMERIC), "categorical": list(CATEGORICAL)},
        "search": search_log,
        "selected": {"model": selected, "params": best[selected][1]["params"], "threshold": threshold,
                     "validation_at_threshold": val_at_t,
                     "mlp_history": getattr(best["pytorch_mlp"][0], "history", None)},
        "test": test, "test_intervals": boot,
        "fairness": fairness, "permutation_importance_validation": importance, "error_analysis": errors,
        "calibration": {"validation": calibration_bins(va[2], model.predict_proba(va[0], va[1])),
                        "test": calibration_bins(y, p_sel)},
    }
    from .figures import save_figures
    result["figures"] = save_figures(REPORT_DIR / "figures", result, y, probs, CASE_ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(result, indent=2, default=float))
    from .report import markdown
    (REPORT_DIR / "results.md").write_text(markdown(result))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    r = run(QUICK if ap.parse_args().quick else DEFAULT)
    t = r["test"][r["selected"]["model"]]
    print(f"selected {r['selected']['model']} {r['selected']['params']}; test AP {t['average_precision']:.3f}, "
          f"ROC-AUC {t['roc_auc']:.3f}; report: {REPORT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
