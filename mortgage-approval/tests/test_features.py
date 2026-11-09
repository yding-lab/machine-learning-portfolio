import numpy as np

from mortgage_approval.config import DEFAULT, PROTECTED, PROXIES
from mortgage_approval.data import make_splits
from mortgage_approval.features import Preprocessor, dti_to_number, raw_features


def test_dti_buckets_and_exact_values():
    assert dti_to_number("<20%") == 15 and dti_to_number("42") == 42 and dti_to_number(">60%") == 65
    assert np.isnan(dti_to_number("NA"))


def test_protected_attributes_and_proxies_never_reach_the_model():
    s = make_splits(DEFAULT)
    df = s.validation.head(2000)
    assert not set(raw_features(df).columns) & (set(PROTECTED) | set(PROXIES))
    pre = Preprocessor().fit(s.train)
    scrambled = df.copy()
    for c in (*PROTECTED, *PROXIES):
        scrambled[c] = np.random.default_rng(0).permutation(scrambled[c].to_numpy())
    assert np.array_equal(pre.numeric(df), pre.numeric(scrambled))
    assert np.array_equal(pre.categorical(df), pre.categorical(scrambled))


def test_preprocessor_uses_training_statistics_and_maps_unseen_levels_to_other():
    s = make_splits(DEFAULT)
    pre = Preprocessor().fit(s.train)
    df = s.test.head(5).copy()
    df["loan_type"] = "never-seen"
    assert (pre.categorical(df)[:, 0] == 0).all()
    assert np.isfinite(pre.numeric(s.test)).all()
