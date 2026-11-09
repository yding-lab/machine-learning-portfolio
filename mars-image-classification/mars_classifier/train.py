"""Train and compare models on the official sol-based split.

    python -m mars_classifier.train            # full run (CPU, roughly an hour)
    python -m mars_classifier.train --quick    # tiny run for tests

Candidates, each tuned on the validation split (sols 182-564) by macro-F1:
  - majority class (always the most common training class)
  - small CNN trained from scratch
  - linear probe: logistic regression on frozen ImageNet ResNet-18 features
  - fine-tuned ResNet-18 (last block + classifier)
The model with the best validation macro-F1 is selected; the test split (sols 565-1060) is
scored once.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from . import __version__
from .config import CASE_ROOT, DEFAULT, PRETRAINED, QUICK, RAW_DIR, REPORT_DIR, Config
from .data import MarsImages, class_names, load_splits
from .evaluate import bootstrap, by_group, confusion, metrics, per_class
from .models import SmallCNN, freeze_except_last_block, predict_logits, resnet18

torch.set_num_threads(max(1, os.cpu_count() or 1))


def loaders(splits, index, size, batch, train_aug=True, seed=0):
    # a seeded generator fixes the shuffle order and the augmentation seeds of the worker processes
    mk = lambda items, tr: DataLoader(MarsImages(items, index, size, tr), batch_size=batch, shuffle=tr,
                                      num_workers=2 if len(items) > 500 else 0,
                                      generator=torch.Generator().manual_seed(seed))
    return {"train": mk(splits["train"], train_aug), "train_eval": mk(splits["train"], False),
            "val": mk(splits["val"], False), "test": mk(splits["test"], False)}


def fit_torch(model, dl, y_val, val_labels, epochs, lr, patience, class_weights=None, seed=0):
    """Adam + cross-entropy; keep the epoch with the best validation macro-F1."""
    torch.manual_seed(seed + 1)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    best, best_state, waited, hist = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in dl["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total, n = total + float(loss.detach()) * len(y), n + len(y)
        pred = predict_logits(model, dl["val"]).argmax(1).numpy()
        f1 = metrics(y_val, pred, None, val_labels)["macro_f1"]
        hist.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": f1, "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {epoch}: loss {total / n:.3f} val macro-F1 {f1:.3f}", flush=True)
        if f1 > best + 1e-4:
            best, best_state, waited = f1, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist, best


def run(config: Config = DEFAULT) -> dict:
    names = class_names()
    splits = load_splits(limit=config.limit_per_split, seed=config.seed)
    train_labels = sorted({i.label for i in splits["train"]})
    index = {lab: k for k, lab in enumerate(train_labels)}
    dropped = {s: sum(i.label not in index for i in v) for s, v in splits.items()}
    splits = {s: [i for i in v if i.label in index] for s, v in splits.items()}
    labels = [names[l] for l in train_labels]
    y = {s: np.array([index[i.label] for i in v]) for s, v in splits.items()}
    present = {s: sorted(set(y[s].tolist())) for s in y}
    K = len(train_labels)
    counts = np.bincount(y["train"], minlength=K)
    cw = torch.tensor(len(y["train"]) / (K * np.maximum(counts, 1)), dtype=torch.float32)

    candidates, search = {}, []

    # 1. majority class
    maj = int(np.argmax(counts))
    candidates["majority"] = {"val_pred": np.full(len(y["val"]), maj), "test_pred": np.full(len(y["test"]), maj),
                              "test_prob": None, "params": {"class": labels[maj]}}

    # 2. small CNN from scratch (class-weighted loss)
    dl = loaders(splits, index, config.scratch_size, config.batch_size, seed=config.seed)
    print("small CNN", flush=True)
    torch.manual_seed(config.seed)  # weight initialisation
    m, hist, best = fit_torch(SmallCNN(K), dl, y["val"], present["val"], config.scratch_epochs, config.scratch_lr,
                              config.patience, cw, config.seed)
    search.append({"model": "small_cnn", "params": {"lr": config.scratch_lr, "size": config.scratch_size}, "val_macro_f1": best, "history": hist})
    lt = predict_logits(m, dl["test"])
    candidates["small_cnn"] = {"val_pred": predict_logits(m, dl["val"]).argmax(1).numpy(), "test_pred": lt.argmax(1).numpy(),
                               "test_prob": torch.softmax(lt, 1).numpy(), "params": search[-1]["params"]}

    # 3. linear probe on frozen ResNet-18 features
    dl = loaders(splits, index, config.image_size, config.batch_size, train_aug=False, seed=config.seed)
    print("ResNet-18 features", flush=True)
    torch.manual_seed(config.seed)
    fx = resnet18(None)
    feats = {s: predict_logits(fx, dl[k]).numpy() for s, k in (("train", "train_eval"), ("val", "val"), ("test", "test"))}
    sc = StandardScaler().fit(feats["train"])
    best_probe = None
    for c in config.probe_c:
        for w in config.probe_class_weight:
            clf = LogisticRegression(C=c, class_weight=w, max_iter=3000).fit(sc.transform(feats["train"]), y["train"])
            f1 = metrics(y["val"], clf.predict(sc.transform(feats["val"])), None, present["val"])["macro_f1"]
            search.append({"model": "linear_probe", "params": {"C": c, "class_weight": w}, "val_macro_f1": f1})
            if best_probe is None or f1 > best_probe[0]:
                best_probe = (f1, clf, {"C": c, "class_weight": w})
    clf = best_probe[1]
    candidates["linear_probe"] = {"val_pred": clf.predict(sc.transform(feats["val"])), "test_pred": clf.predict(sc.transform(feats["test"])),
                                  "test_prob": _full_proba(clf, sc.transform(feats["test"]), K), "params": best_probe[2]}

    # 4. fine-tune ResNet-18 last block (class-weighted loss)
    dl = loaders(splits, index, config.image_size, config.batch_size, seed=config.seed)
    best_ft = None
    for lr in config.finetune_lrs:
        print(f"fine-tune lr={lr}", flush=True)
        torch.manual_seed(config.seed)
        m = freeze_except_last_block(resnet18(K))
        m, hist, f1 = fit_torch(m, dl, y["val"], present["val"], config.finetune_epochs, lr, config.patience, cw, config.seed)
        search.append({"model": "finetuned_resnet18", "params": {"lr": lr}, "val_macro_f1": f1, "history": hist})
        if best_ft is None or f1 > best_ft[0]:
            best_ft = (f1, m, {"lr": lr})
    m = best_ft[1]
    lt = predict_logits(m, dl["test"])
    candidates["finetuned_resnet18"] = {"val_pred": predict_logits(m, dl["val"]).argmax(1).numpy(), "test_pred": lt.argmax(1).numpy(),
                                        "test_prob": torch.softmax(lt, 1).numpy(), "params": best_ft[2]}

    # selection on validation, then the test split once
    val = {k: metrics(y["val"], c["val_pred"], None, present["val"]) for k, c in candidates.items()}
    selected = max((k for k in candidates if k != "majority"), key=lambda k: val[k]["macro_f1"])
    test = {k: metrics(y["test"], c["test_pred"], c["test_prob"], present["test"]) for k, c in candidates.items()}
    sel = candidates[selected]
    instruments = np.array([i.instrument for i in splits["test"]])
    result = {
        "case": "mars-image-classification", "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "scikit_learn": sklearn.__version__, "pretrained_weights": "ImageNet1K_V1" if PRETRAINED else None},
        "config": config.__dict__,
        "data": {"n": {s: len(v) for s, v in splits.items()}, "dropped_unseen_class": dropped, "classes": labels,
                 "train_counts": dict(zip(labels, counts.tolist())),
                 "test_counts": dict(zip(labels, np.bincount(y["test"], minlength=K).tolist())),
                 "sols": {s: [min(i.sol for i in v), max(i.sol for i in v)] for s, v in splits.items()},
                 "instrument_mix": {s: {str(k): int(n) for k, n in zip(*np.unique([i.instrument for i in v], return_counts=True))} for s, v in splits.items()}},
        "search": search, "validation": val, "selected": {"model": selected, "params": sel["params"]},
        "test": test,
        "test_intervals": bootstrap(y["test"], {k: c["test_pred"] for k, c in candidates.items()}, selected, present["test"],
                                    config.n_bootstrap, config.seed),
        "per_class_test": per_class(y["test"], sel["test_pred"], labels),
        "confusion_test": confusion(y["test"], sel["test_pred"], K, present["test"]),
        "by_instrument_test": by_group(y["test"], sel["test_pred"], instruments, present["test"]),
    }
    from .figures import save_figures
    result["figures"] = save_figures(REPORT_DIR / "figures", result, splits, CASE_ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(result, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    from .report import markdown
    (REPORT_DIR / "results.md").write_text(markdown(result))
    return result


def _full_proba(clf, X, K):
    p = np.zeros((len(X), K))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    r = run(QUICK if ap.parse_args().quick else DEFAULT)
    t = r["test"][r["selected"]["model"]]
    print(f"selected {r['selected']['model']}: test macro-F1 {t['macro_f1']:.3f}, accuracy {t['accuracy']:.3f}")


if __name__ == "__main__":
    main()
