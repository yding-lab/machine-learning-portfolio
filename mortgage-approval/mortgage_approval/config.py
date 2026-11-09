"""Paths and settings for the mortgage-approval case."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", CASE_ROOT / "data" / "hmda_ny_home_purchase.csv.gz"))
SOURCE_PATH = DATA_PATH.with_name(DATA_PATH.name.replace(".csv.gz", ".SOURCE.json"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", CASE_ROOT / "reports"))

# Applications missing any of these are excluded: in HMDA they are mostly missing because the
# application was denied before the information was collected (e.g. no appraisal), so a model
# could learn "missing -> denied", which would not hold at decision time.
REQUIRED_FIELDS = ("combined_loan_to_value_ratio", "property_value", "loan_term", "debt_to_income_ratio")

# Never model inputs. Used only by the fairness audit.
PROTECTED = ("derived_race", "derived_ethnicity", "derived_sex", "applicant_age")
# Kept out of the model because they can act as proxies for protected attributes.
PROXIES = ("county_code", "tract_minority_population_percent")

NUMERIC = ("log_loan_amount", "log_income", "cltv", "log_property_value", "loan_term", "dti",
           "loan_to_income", "tract_population", "tract_to_msa_income_percentage", "log_msa_median_income",
           "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
           "income_missing")
CATEGORICAL = ("loan_type", "applicant_credit_score_type", "co_applicant_credit_score_type", "construction_method",
               "total_units", "conforming_loan_limit", "derived_dwelling_category", "interest_only_payment",
               "balloon_payment", "negative_amortization", "other_nonamortizing_features", "has_intro_rate")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    validation_share: float = 0.2  # of the 2023 applications
    n_bootstrap: int = 1000
    # PyTorch MLP search (validation set only)
    mlp_hidden: tuple[tuple[int, ...], ...] = ((64, 32), (128, 64))
    mlp_dropout: tuple[float, ...] = (0.1, 0.3)
    mlp_max_epochs: int = 40
    mlp_patience: int = 5
    mlp_batch_size: int = 512
    mlp_lr: float = 1e-3
    lr_c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    hgb_learning_rates: tuple[float, ...] = (0.05, 0.1)
    hgb_max_leaf_nodes: tuple[int, ...] = (15, 31)
    train_sample: int | None = None  # quick runs only
    test_sample: int | None = None


DEFAULT = Config()
QUICK = Config(n_bootstrap=50, mlp_hidden=((32,),), mlp_dropout=(0.1,), mlp_max_epochs=4, lr_c_values=(1.0,),
               hgb_learning_rates=(0.1,), hgb_max_leaf_nodes=(15,), train_sample=6000, test_sample=3000)
