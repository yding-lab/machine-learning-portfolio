"""Find, mask, resize and cache the MSL NAVCAM part of AI4Mars.

Training labels are crowd-sourced. Following the data set's documentation, pixels on the rover
(mxy mask) and beyond 30 m (rng-30m mask) are set to IGNORE. Test labels come pre-masked by
the data set ('masked-gold-*'), from three expert labellers per image.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from .config import AGREEMENT, CACHE_DIR, IGNORE, RAW_DIR


class DataError(RuntimeError):
    pass


@dataclass
class Sample:
    stem: str          # e.g. NLB_397586934EDR_F0010008AUT_04096M1
    image: Path
    label: Path
    mxy: Path | None = None
    rng: Path | None = None

    @property
    def sclk(self) -> int:
        return int(self.stem.split("_")[1][:9])

    @property
    def position(self) -> str:
        """Site + drive code of the rover position (characters 2-8 of the third field)."""
        return self.stem.split("_")[2][1:8]


def ncam_dir(raw: Path = RAW_DIR) -> Path:
    d = raw / "ai4mars-dataset-merged-0.6" / "msl" / "ncam"
    if not d.exists():
        raise DataError(f"{d} not found: run scripts/fetch_data.py first")
    return d


def _aux(d: Path, kind: str, stem: str) -> Path | None:
    p = d / "images" / kind.lower() / (stem.replace("EDR", kind.split("-")[0].upper()) + ".png")
    return p if p.exists() else None


def training_samples(raw: Path = RAW_DIR, limit: int | None = None) -> list[Sample]:
    d = ncam_dir(raw)
    out = []
    for lab in sorted((d / "labels" / "train").glob("*.png")):
        img = d / "images" / "edr" / (lab.stem + ".JPG")
        if not img.exists():
            raise DataError(f"image missing for training label {lab.name}")
        out.append(Sample(lab.stem, img, lab, _aux(d, "MXY", lab.stem), _aux(d, "RNG-30m", lab.stem)))
    if not out:
        raise DataError("no training labels found")
    return out[:limit] if limit else out


def test_samples(raw: Path = RAW_DIR) -> dict[str, list[Sample]]:
    d = ncam_dir(raw)
    out = {}
    for lvl in AGREEMENT:
        items = []
        for lab in sorted((d / "labels" / "test" / f"masked-gold-{lvl}-100agree").glob("*.png")):
            stem = lab.name.replace("_merged.png", "")
            img = d / "images" / "edr" / (stem + ".JPG")
            if not img.exists():
                raise DataError(f"image missing for test label {lab.name}")
            items.append(Sample(stem, img, lab))
        out[lvl] = items
    if len({tuple(s.stem for s in v) for v in out.values()}) != 1:
        raise DataError("the three agreement levels should label the same images")
    return out


def split_by_position(samples: list[Sample], fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Hold out whole rover positions, so overlapping views from one stop stay on one side."""
    positions = sorted({s.position for s in samples})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(positions, max(1, round(fraction * len(positions))), replace=False).tolist())
    return [s for s in samples if s.position not in held], [s for s in samples if s.position in held]


def read_label(path: Path) -> np.ndarray:
    y = np.array(Image.open(path), dtype=np.uint8)
    return y[..., 0] if y.ndim == 3 else y


def load_label(s: Sample) -> np.ndarray:
    y = read_label(s.label)
    bad = ~np.isin(y, [0, 1, 2, 3, IGNORE])
    if bad.any():
        raise DataError(f"unexpected label values {np.unique(y[bad])[:5]} in {s.label.name}")
    for m in (s.mxy, s.rng):
        if m is not None:
            mask = np.array(Image.open(m))
            if mask.shape == y.shape:
                y[mask > 0] = IGNORE
    return y


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _resized(args):
    s, size = args
    x = Image.fromarray(load_image(s.image)).resize((size, size), Image.BILINEAR)
    y = Image.fromarray(load_label(s)).resize((size, size), Image.NEAREST)
    return np.asarray(x), np.asarray(y)


def cache(samples: list[Sample], size: int, workers: int = 0, root: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Resized images and masked labels as uint8 memory maps (built once per sample list and size)."""
    key = hashlib.sha1(json.dumps([size] + [s.stem for s in samples]).encode()).hexdigest()[:12]
    xp, yp = root / f"x_{size}_{key}.npy", root / f"y_{size}_{key}.npy"
    if not (xp.exists() and yp.exists()):
        root.mkdir(parents=True, exist_ok=True)
        x = np.lib.format.open_memmap(xp.with_suffix(".tmp"), "w+", np.uint8, (len(samples), size, size))
        y = np.lib.format.open_memmap(yp.with_suffix(".tmp"), "w+", np.uint8, (len(samples), size, size))
        jobs = [(s, size) for s in samples]
        it = Pool(workers).imap(_resized, jobs, chunksize=32) if workers else map(_resized, jobs)
        for i, (a, b) in enumerate(it):
            x[i], y[i] = a, b
            if i and i % 2000 == 0:
                print(f"  cached {i}/{len(samples)}", flush=True)
        x.flush(), y.flush()
        del x, y
        xp.with_suffix(".tmp").rename(xp)
        yp.with_suffix(".tmp").rename(yp)
    return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")
