"""Tests run on a small synthetic data set with the real layout: no download, no pretrained weights."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TMP = Path(tempfile.mkdtemp(prefix="mars-tests-"))
RAW = TMP / "raw"
os.environ.update(MARS_RAW_DIR=str(RAW), REPORT_DIR=str(TMP / "reports"), MARS_PRETRAINED="0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["ground", "wheel", "horizon", "drill", "sun"]  # 'sun' has no images, as in the real data
SOLS = {"train": (3, 180), "val": (182, 564), "test": (565, 1060)}


def make_fake_dataset(root: Path, per_class=12, seed=0) -> None:
    rng = np.random.default_rng(seed)
    base = root / "msl-images"
    (base / "calibrated").mkdir(parents=True, exist_ok=True)
    (base / "msl_synset_words-indexed.txt").write_text("".join(f"{i} {c}\n" for i, c in enumerate(CLASSES)))
    colors = [(150, 100, 60), (60, 60, 60), (200, 170, 120), (90, 140, 200)]
    for split, (lo, hi) in SOLS.items():
        lines = []
        for label, color in enumerate(colors):
            for k in range(per_class):
                sol = int(rng.integers(lo, hi + 1))
                cam = ["MH", "ML", "MR"][k % 3]
                name = f"{sol:04d}{cam}{label:02d}{k:03d}{split[:2]}_DRCL.JPG"
                arr = np.clip(np.array(color) + rng.normal(0, 25, (40, 48, 3)), 0, 255).astype(np.uint8)
                im = Image.fromarray(arr)
                if k == 0:
                    im = im.convert("L")  # some real images are greyscale
                im.save(base / "calibrated" / name)
                lines.append(f"calibrated/{name} {label}\n")
        rng.shuffle(lines)
        (base / f"{split}-calibrated-shuffled.txt").write_text("".join(lines))


make_fake_dataset(RAW)


@pytest.fixture(scope="session")
def quick_run():
    from mars_classifier.config import QUICK
    from mars_classifier.train import run

    return run(QUICK)
