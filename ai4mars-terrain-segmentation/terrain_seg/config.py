from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.getenv("AI4MARS_RAW_DIR", CASE_ROOT / "data" / "raw"))
CACHE_DIR = Path(os.getenv("AI4MARS_CACHE_DIR", CASE_ROOT / "artifacts" / "cache"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", CASE_ROOT / "reports"))
PRETRAINED = os.getenv("AI4MARS_PRETRAINED", "1") != "0"

CLASSES = ("soil", "bedrock", "sand", "big rock")
IGNORE = 255
AGREEMENT = ("min1", "min2", "min3")  # pixels kept where >= 1, 2 or 3 of 3 expert labellers agree
PRIMARY_AGREEMENT = "min2"            # majority of the experts
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    image_size: int = 256          # training/inference resolution; test is scored at the labels' full 1024x1024
    batch_size: int = 16
    samples_per_epoch: int = 4000  # random training images drawn per epoch (CPU budget)
    max_epochs: int = 8
    patience: int = 2
    lr: dict = field(default_factory=lambda: {"unet": 1e-3, "lraspp_mobilenet_v3": 1e-3})
    time_budget_min: float = 80.0  # per model
    val_fraction: float = 0.10     # of rover positions (site+drive), not of images
    val_limit: int | None = 800
    n_bootstrap: int = 1000
    limit_train: int | None = None
    workers: int = 4


DEFAULT = Config()
QUICK = replace(DEFAULT, image_size=32, batch_size=8, samples_per_epoch=48, max_epochs=2, patience=1,
                time_budget_min=5, val_fraction=0.25, val_limit=None, n_bootstrap=30, workers=0)
