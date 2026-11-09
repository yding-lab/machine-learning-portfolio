"""Tests use a small synthetic data set with AI4Mars' real folder layout and file names."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TMP = Path(tempfile.mkdtemp(prefix="ai4mars-tests-"))
os.environ.update(AI4MARS_RAW_DIR=str(TMP / "raw"), AI4MARS_CACHE_DIR=str(TMP / "cache"),
                  REPORT_DIR=str(TMP / "reports"), AI4MARS_PRETRAINED="0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BRIGHT = [90, 150, 210, 40]  # soil, bedrock, sand, big rock


def scene(rng, size=64):
    """Image and label with horizontal bands of class-specific brightness."""
    y = np.zeros((size, size), np.uint8)
    cuts = np.sort(rng.choice(np.arange(8, size - 8), 3, replace=False))
    for k, (a, b) in enumerate(zip([0, *cuts], [*cuts, size])):
        y[a:b] = rng.integers(0, 4) if k else 0
    x = np.array(BRIGHT, float)[y] + rng.normal(0, 12, y.shape)
    return np.clip(x, 0, 255).astype(np.uint8), y


def make_fake_dataset(raw: Path, n_train=40, n_test=8, seed=0) -> None:
    rng = np.random.default_rng(seed)
    d = raw / "ai4mars-dataset-merged-0.6" / "msl" / "ncam"
    for sub in ("images/edr", "images/mxy", "images/rng-30m", "labels/train",
                *(f"labels/test/masked-gold-{l}-100agree" for l in ("min1", "min2", "min3"))):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n_train + n_test):
        pos = f"{i % 10:03d}{i // 10:04d}"  # site + drive code
        stem = f"NLB_{400000000 + 1000 * i}EDR_F{pos}NCAM00348M1"
        x, y = scene(rng)
        Image.fromarray(x).save(d / "images/edr" / f"{stem}.JPG", quality=95)
        if i < n_train:
            mxy = np.zeros_like(y); mxy[-6:, :] = 1          # rover in the bottom rows
            rng30 = np.zeros_like(y); rng30[:4, :] = 1       # beyond 30 m in the top rows
            Image.fromarray(mxy).save(d / "images/mxy" / f"{stem.replace('EDR', 'MXY')}.png")
            Image.fromarray(rng30).save(d / "images/rng-30m" / f"{stem.replace('EDR', 'RNG')}.png")
            lab = y.copy(); lab[:, :3] = 255                 # some unlabelled pixels
            Image.fromarray(lab).save(d / "labels/train" / f"{stem}.png")
        else:
            for k, lvl in enumerate(("min1", "min2", "min3")):
                lab = y.copy(); lab[:, : 4 * (k + 1)] = 255  # stricter agreement -> fewer labelled pixels
                Image.fromarray(lab).save(d / f"labels/test/masked-gold-{lvl}-100agree" / f"{stem}_merged.png")


make_fake_dataset(TMP / "raw")


@pytest.fixture(scope="session")
def quick_run():
    from terrain_seg.config import QUICK
    from terrain_seg.train import run

    return run(QUICK)
