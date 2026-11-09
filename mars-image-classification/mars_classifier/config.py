"""Paths and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.getenv("MARS_RAW_DIR", CASE_ROOT / "data" / "raw"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", CASE_ROOT / "reports"))
PRETRAINED = os.getenv("MARS_PRETRAINED", "1") == "1"  # tests set 0: no weight download

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INSTRUMENTS = {"MH": "MAHLI (close-up)", "ML": "Mastcam left", "MR": "Mastcam right"}


@dataclass(frozen=True)
class Config:
    seed: int = 0
    image_size: int = 224
    batch_size: int = 64
    n_bootstrap: int = 1000
    # linear probe on frozen ResNet-18 features
    probe_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    probe_class_weight: tuple[str | None, ...] = (None, "balanced")
    # fine-tuning ResNet-18 (layer4 + classifier)
    finetune_lrs: tuple[float, ...] = (3e-4, 1e-3)
    finetune_epochs: int = 6
    # small CNN from scratch
    scratch_size: int = 128
    scratch_epochs: int = 15
    scratch_lr: float = 1e-3
    patience: int = 3
    limit_per_split: int | None = None  # quick runs only


DEFAULT = Config()
QUICK = Config(image_size=64, batch_size=32, n_bootstrap=30, probe_c=(1.0,), probe_class_weight=(None,),
               finetune_lrs=(1e-3,), finetune_epochs=1, scratch_size=48, scratch_epochs=2, patience=1,
               limit_per_split=120)
