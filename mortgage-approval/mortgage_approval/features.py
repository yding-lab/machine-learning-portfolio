"""Turn raw HMDA fields into model inputs. Protected attributes and proxies never enter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CATEGORICAL, NUMERIC, PROTECTED, PROXIES

# HMDA reports DTI in buckets below 36% and above 49%, exact percentages in between.
DTI_BUCKETS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}


def dti_to_number(value: str) -> float:
    if value in DTI_BUCKETS:
        return DTI_BUCKETS[value]
    try:
        return float(value)
    except ValueError:
        return np.nan


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered numeric and categorical columns (unscaled, unencoded)."""
    income = _num(df["income"])  # thousands of dollars; may be missing (not associated with denial)
    loan = _num(df["loan_amount"])
    out = pd.DataFrame({
        "log_loan_amount": np.log1p(loan),
        "log_income": np.log1p(income.clip(lower=0)),
        "cltv": _num(df["combined_loan_to_value_ratio"]).clip(0, 200),
        "log_property_value": np.log1p(_num(df["property_value"])),
        "loan_term": _num(df["loan_term"]).clip(0, 480),
        "dti": df["debt_to_income_ratio"].map(dti_to_number),
        "loan_to_income": (loan / (income.clip(lower=1) * 1000)).clip(0, 20),
        "tract_population": _num(df["tract_population"]),
        "tract_to_msa_income_percentage": _num(df["tract_to_msa_income_percentage"]).clip(0, 500),
        "log_msa_median_income": np.log1p(_num(df["ffiec_msa_md_median_family_income"])),
        "tract_owner_occupied_units": _num(df["tract_owner_occupied_units"]),
        "tract_one_to_four_family_homes": _num(df["tract_one_to_four_family_homes"]),
        "tract_median_age_of_housing_units": _num(df["tract_median_age_of_housing_units"]),
        "income_missing": income.isna().astype(float),
    })
    for c in CATEGORICAL:
        if c == "has_intro_rate":
            out[c] = np.where(df["intro_rate_period"].isin({"NA", "Exempt", ""}), "no", "yes")
        else:
            out[c] = df[c].astype(str)
    assert not set(out.columns) & (set(PROTECTED) | set(PROXIES))
    return out


@dataclass
class Preprocessor:
    """Fitted on the training split only: medians, means/stds and category vocabularies."""

    medians: dict = field(default_factory=dict)
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    vocab: dict = field(default_factory=dict)
    min_count: int = 20

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        f = raw_features(df)
        for c in NUMERIC:
            med = float(f[c].median())
            filled = f[c].fillna(med)
            self.medians[c], self.means[c], self.stds[c] = med, float(filled.mean()), float(filled.std() or 1.0)
        for c in CATEGORICAL:
            counts = f[c].value_counts()
            levels = sorted(counts[counts >= self.min_count].index)
            self.vocab[c] = {"<other>": 0, **{v: i + 1 for i, v in enumerate(levels)}}  # rare/unseen -> 0
        return self

    def numeric(self, df: pd.DataFrame) -> np.ndarray:
        f = raw_features(df)
        cols = [((f[c].fillna(self.medians[c]) - self.means[c]) / self.stds[c]).to_numpy() for c in NUMERIC]
        return np.column_stack(cols).astype(np.float32)

    def categorical(self, df: pd.DataFrame) -> np.ndarray:
        f = raw_features(df)
        cols = [f[c].map(lambda v, vv=self.vocab[c]: vv.get(v, 0)).to_numpy() for c in CATEGORICAL]
        return np.column_stack(cols).astype(np.int64)

    def one_hot(self, df: pd.DataFrame) -> np.ndarray:
        """Numeric block + one-hot categoricals, for the scikit-learn baselines."""
        cat = self.categorical(df)
        blocks = [self.numeric(df)]
        for j, c in enumerate(CATEGORICAL):
            k = len(self.vocab[c])
            oh = np.zeros((len(cat), k), dtype=np.float32)
            oh[np.arange(len(cat)), cat[:, j]] = 1
            blocks.append(oh)
        return np.hstack(blocks)

    def feature_names(self) -> list[str]:
        names = list(NUMERIC)
        for c in CATEGORICAL:
            names += [f"{c}={v}" for v in self.vocab[c]]
        return names
