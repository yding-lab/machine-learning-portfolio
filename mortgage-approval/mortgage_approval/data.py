"""Load and verify the committed HMDA snapshot, apply exclusions, and split it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, REQUIRED_FIELDS, SOURCE_PATH, Config

MISSING = {"NA", "Exempt", ""}


class DatasetError(ValueError):
    """The snapshot is missing, altered or structurally wrong."""


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    excluded: dict = field(default_factory=dict)


def load_snapshot(path: Path = DATA_PATH, source: Path = SOURCE_PATH, verify: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(f"{path} not found; rebuild with scripts/fetch_hmda.py")
    if verify and source.exists():
        expected = json.loads(source.read_text())["snapshot_sha256"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise DatasetError(f"checksum {actual[:12]} doesn't match the SOURCE record {expected[:12]}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    needed = {"split", "activity_year", "action_taken", *REQUIRED_FIELDS}
    missing = needed - set(df.columns)
    if missing:
        raise DatasetError(f"missing columns: {sorted(missing)}")
    if set(df["split"]) != {"train", "test"}:
        raise DatasetError(f"split must be train/test, got {sorted(set(df['split']))}")
    if not set(df["action_taken"]) <= {"1", "2", "3"}:
        raise DatasetError(f"unexpected action_taken codes {sorted(set(df['action_taken']) - {'1', '2', '3'})}")
    years = df.groupby("split")["activity_year"].agg(lambda s: set(s))
    if not max(map(int, years["train"])) < min(map(int, years["test"])):
        raise DatasetError(f"time split violated: train years {years['train']}, test years {years['test']}")


def is_complete(df: pd.DataFrame) -> pd.Series:
    return ~df[list(REQUIRED_FIELDS)].isin(MISSING).any(axis=1)


def make_splits(config: Config, df: pd.DataFrame | None = None) -> Splits:
    df = load_snapshot() if df is None else df
    df = df.assign(denied=(df["action_taken"] == "3").astype(int))
    complete = is_complete(df)
    excluded = {}
    for s in ("train", "test"):
        part = df["split"] == s
        excluded[s] = {"total": int(part.sum()), "excluded": int((part & ~complete).sum()),
                       "denial_rate_excluded": float(df.loc[part & ~complete, "denied"].mean()),
                       "denial_rate_kept": float(df.loc[part & complete, "denied"].mean())}
    df = df[complete]
    train_all, test = df[df["split"] == "train"], df[df["split"] == "test"]
    if config.train_sample and len(train_all) > config.train_sample:
        train_all = train_all.sample(config.train_sample, random_state=config.seed)
    if config.test_sample and len(test) > config.test_sample:
        test = test.sample(config.test_sample, random_state=config.seed)
    # Stratified random split of 2023 into train / validation.
    rng = np.random.default_rng(config.seed)
    val_idx = []
    for _, grp in train_all.groupby("denied"):
        n = int(round(len(grp) * config.validation_share))
        val_idx.extend(rng.choice(grp.index.to_numpy(), n, replace=False))
    val_mask = train_all.index.isin(val_idx)
    return Splits(train=train_all[~val_mask].reset_index(drop=True), validation=train_all[val_mask].reset_index(drop=True),
                  test=test.reset_index(drop=True), excluded=excluded)
