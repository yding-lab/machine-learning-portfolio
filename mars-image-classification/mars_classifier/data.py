"""Read the MSL labeled data set: official sol-based train/val/test split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, INSTRUMENTS, RAW_DIR

SPLITS = ("train", "val", "test")


class DataError(ValueError):
    """The downloaded data set is missing or malformed."""


@dataclass
class Item:
    path: Path
    label: int
    sol: int
    instrument: str


def _find(raw: Path, name: str) -> Path:
    hits = list(raw.rglob(name))
    if not hits:
        raise DataError(f"{name} not found under {raw}; run scripts/fetch_data.py first")
    return hits[0]


def class_names(raw: Path = RAW_DIR) -> dict[int, str]:
    out = {}
    for line in _find(raw, "msl_synset_words-indexed.txt").read_text().splitlines():
        if line.strip():
            idx, *name = line.split()
            out[int(idx)] = " ".join(name)
    return out


def read_split(split: str, raw: Path = RAW_DIR) -> list[Item]:
    root = _find(raw, "train-calibrated-shuffled.txt").parent
    items = []
    for line in (root / f"{split}-calibrated-shuffled.txt").read_text().splitlines():
        if not line.strip():
            continue
        rel, label = line.split()
        name = Path(rel).name
        p = root / rel
        if not p.exists():
            raise DataError(f"listed image missing: {p}")
        items.append(Item(p, int(label), int(name[:4]), INSTRUMENTS.get(name[4:6], "other")))
    return items


def load_splits(raw: Path = RAW_DIR, limit: int | None = None, seed: int = 0) -> dict[str, list[Item]]:
    splits = {s: read_split(s, raw) for s in SPLITS}
    # Official split is by sol; check that it really is chronological.
    if not max(i.sol for i in splits["train"]) < min(i.sol for i in splits["val"]) <= \
            max(i.sol for i in splits["val"]) < min(i.sol for i in splits["test"]):
        raise DataError("splits are not in sol order")
    if limit:
        rng = np.random.default_rng(seed)
        splits = {s: [v[i] for i in sorted(rng.choice(len(v), min(limit, len(v)), replace=False))]
                  for s, v in splits.items()}
    return splits


def to_rgb(im):
    return im.convert("RGB")  # 172 images are greyscale


def transform(size: int, train: bool):
    ops = [to_rgb]
    if train:
        ops += [transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2)]
    else:
        ops += [transforms.Resize((size, size))]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class MarsImages(Dataset):
    def __init__(self, items: list[Item], label_index: dict[int, int], size: int, train: bool):
        self.items, self.index, self.tf = items, label_index, transform(size, train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        with Image.open(it.path) as im:
            x = self.tf(im)
        return x, torch.tensor(self.index[it.label])
