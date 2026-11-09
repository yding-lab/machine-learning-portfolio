"""Train, select and evaluate terrain segmentation models on Curiosity NAVCAM images.

    python -m terrain_seg.train            # full run (CPU, about 3 h)
    python -m terrain_seg.train --quick    # small version for tests

1. Training labels (crowd-sourced) are split by rover position: 10% of positions are held out
   for validation, so overlapping views from one stop never straddle the split.
2. Models are compared on validation mIoU; the best epoch of each is kept by validation mIoU.
3. The expert-labelled test set is scored once, at the labels' full resolution, at all three
   agreement levels, with bootstrap intervals over test images.
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
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from . import __version__
from .config import (AGREEMENT, CASE_ROOT, CLASSES, DEFAULT, IGNORE, IMAGENET_MEAN, IMAGENET_STD, PRETRAINED,
                     PRIMARY_AGREEMENT, QUICK, REPORT_DIR, Config)
from .data import cache, load_image, read_label, split_by_position, test_samples, training_samples
from .evaluate import bootstrap, confusion, scores
from .models import MODELS

torch.set_num_threads(max(1, os.cpu_count() or 1))
K = len(CLASSES)
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
NAMES = {"majority": "Majority class", "unet": "U-Net (scratch)", "lraspp_mobilenet_v3": "LR-ASPP MobileNetV3 (ImageNet)"}


def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
    """(N,H,W) uint8 greyscale -> (N,3,H,W) normalised floats."""
    x = x_uint8.float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - MEAN) / STD


class Arrays(Dataset):
    def __init__(self, x, y, flip: bool):
        self.x, self.y, self.flip = x, y, flip

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = torch.from_numpy(np.array(self.x[i])), torch.from_numpy(np.array(self.y[i]).astype(np.int64))
        if self.flip and torch.rand(()) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


@torch.no_grad()
def predict(model, x: np.ndarray, batch: int = 32) -> np.ndarray:
    model.eval()
    return np.concatenate([model(to_input(torch.from_numpy(np.array(x[i:i + batch])))).argmax(1).numpy().astype(np.uint8)
                           for i in range(0, len(x), batch)])


def miou(y: np.ndarray, pred: np.ndarray) -> float:
    return scores(confusion(y, pred))["miou"]


def fit(name: str, config: Config, tr, va, class_weights) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(config.seed)
    model = MODELS[name](K)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr[name], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE)
    g = torch.Generator().manual_seed(config.seed)
    hist, best, best_state, waited, t_start = [], -1.0, None, 0, time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        idx = torch.randint(0, len(tr[0]), (config.samples_per_epoch,), generator=g)
        dl = DataLoader(Arrays(tr[0], tr[1], flip=True), batch_size=config.batch_size,
                        sampler=idx.tolist(), num_workers=0)
        t0, total, n = time.time(), 0.0, 0
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(to_input(x)), y)
            loss.backward()
            opt.step()
            total, n = total + float(loss.detach()) * len(y), n + len(y)
        v = miou(np.asarray(va[1]), predict(model, va[0]))
        hist.append({"epoch": epoch, "train_loss": total / n, "val_miou": v, "seconds": round(time.time() - t0, 1)})
        print(f"  {name} epoch {epoch}: loss {total / n:.3f} val mIoU {v:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if v > best + 1e-4:
            best, best_state, waited = v, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
        if waited >= config.patience or (time.time() - t_start) / 60 > config.time_budget_min:
            break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_full(model, imgs: list[np.ndarray], size: int) -> list[np.ndarray]:
    """Predict at `size`, upsample the logits to each image's own resolution, then argmax."""
    from PIL import Image

    model.eval()
    out = []
    for im in imgs:
        small = np.array(Image.fromarray(im).resize((size, size), Image.BILINEAR))
        logits = model(to_input(torch.from_numpy(small)[None]))
        out.append(F.interpolate(logits, size=im.shape, mode="bilinear", align_corners=False).argmax(1)[0].numpy().astype(np.uint8))
    return out


def run(config: Config = DEFAULT) -> dict:
    samples = training_samples(limit=config.limit_train)
    test = test_samples()
    tr_s, va_s = split_by_position(samples, config.val_fraction, config.seed)
    print(f"train {len(tr_s)}, validation {len(va_s)}, test {len(test[PRIMARY_AGREEMENT])} images", flush=True)
    tr, va = cache(tr_s, config.image_size, config.workers), cache(va_s, config.image_size, config.workers)
    if config.val_limit and len(va_s) > config.val_limit:
        keep = np.sort(np.random.default_rng(config.seed).choice(len(va_s), config.val_limit, replace=False))
        va = (np.asarray(va[0][keep]), np.asarray(va[1][keep]))

    pix = np.zeros(K, dtype=np.int64)
    for i in range(0, len(tr[1]), 512):
        b = np.asarray(tr[1][i:i + 512])
        pix += np.bincount(b[b != IGNORE].ravel(), minlength=K)[:K]
    freq = pix / pix.sum()
    w = 1 / np.sqrt(np.maximum(freq, 1e-6))
    class_weights = torch.tensor(w / w.mean(), dtype=torch.float32)

    # ---- model search on validation ----
    maj = int(np.argmax(pix))
    val, histories, models = {"majority": miou(np.asarray(va[1]), np.full(va[1].shape, maj, np.uint8))}, {}, {}
    for name in MODELS:
        print(f"training {name}", flush=True)
        models[name], histories[name] = fit(name, config, tr, va, class_weights)
        val[name] = max(h["val_miou"] for h in histories[name])
    selected = max(models, key=lambda m: val[m])

    # ---- the expert-labelled test set, once ----
    stems = [s.stem for s in test[PRIMARY_AGREEMENT]]
    imgs = [load_image(s.image) for s in test[PRIMARY_AGREEMENT]]
    preds = {"majority": [np.full(im.shape, maj, np.uint8) for im in imgs]}
    preds.update({m: predict_full(models[m], imgs, config.image_size) for m in models})
    labels = {lvl: [read_label(s.label) for s in test[lvl]] for lvl in AGREEMENT}
    per_image = {lvl: {m: np.stack([confusion(y, p) for y, p in zip(labels[lvl], preds[m])]) for m in preds}
                 for lvl in AGREEMENT}
    test_scores = {lvl: {m: scores(c.sum(0)) for m, c in per_image[lvl].items()} for lvl in AGREEMENT}
    intervals = bootstrap(per_image[PRIMARY_AGREEMENT], selected, config.n_bootstrap, config.seed)

    train_positions = {s.position for s in samples}
    unseen = np.array([s.position not in train_positions for s in test[PRIMARY_AGREEMENT]])
    unseen_scores = ({m: scores(c[unseen].sum(0)) for m, c in per_image[PRIMARY_AGREEMENT].items()}
                     if unseen.any() else None)

    result = {
        "case": "ai4mars-terrain-segmentation", "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "pretrained_backbone": PRETRAINED},
        "config": config.__dict__,
        "data": {"n_train": len(tr_s), "n_validation": len(va_s), "n_validation_scored": len(va[0]),
                 "n_test": len(stems), "train_positions": len({s.position for s in tr_s}),
                 "validation_positions": len({s.position for s in va_s}),
                 "test_images_at_unseen_positions": int(unseen.sum()),
                 "train_pixel_share": {c: float(f) for c, f in zip(CLASSES, freq)},
                 "class_weights": {c: float(v) for c, v in zip(CLASSES, class_weights)}},
        "validation_miou": val, "histories": histories,
        "selected": {"model": selected, "majority_class": CLASSES[maj]},
        "test": test_scores, "test_intervals_primary": intervals,
        "test_unseen_positions_primary": unseen_scores,
        "confusion_primary": {m: c.sum(0).tolist() for m, c in per_image[PRIMARY_AGREEMENT].items()},
    }
    from .figures import save_figures
    result["figures"] = save_figures(REPORT_DIR / "figures", result, imgs, labels[PRIMARY_AGREEMENT], preds, CASE_ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(result, indent=2, default=float))
    from .report import markdown
    (REPORT_DIR / "results.md").write_text(markdown(result))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    r = run(QUICK if ap.parse_args().quick else DEFAULT)
    s = r["selected"]["model"]
    print(f"selected {s}; test mIoU ({PRIMARY_AGREEMENT}) {r['test'][PRIMARY_AGREEMENT][s]['miou']:.3f}; "
          f"report: {REPORT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
