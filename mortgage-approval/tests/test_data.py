import gzip

import pytest

from mortgage_approval.config import DATA_PATH, DEFAULT, SOURCE_PATH
from mortgage_approval.data import DatasetError, is_complete, load_snapshot, make_splits, validate


def test_snapshot_matches_checksum_and_time_split():
    df = load_snapshot()
    assert len(df) == 120000
    assert set(df.loc[df.split == "train", "activity_year"]) == {"2023"}
    assert set(df.loc[df.split == "test", "activity_year"]) == {"2024"}


def test_altered_snapshot_is_rejected(tmp_path):
    raw = gzip.decompress(DATA_PATH.read_bytes()).replace(b",3,", b",1,", 1)
    p = tmp_path / DATA_PATH.name
    p.write_bytes(gzip.compress(raw))
    with pytest.raises(DatasetError, match="checksum"):
        load_snapshot(p, SOURCE_PATH)


def test_incomplete_applications_are_excluded_and_counted():
    s = make_splits(DEFAULT)
    for part in (s.train, s.validation, s.test):
        assert is_complete(part).all()
    ex = s.excluded["train"]
    assert ex["excluded"] + len(s.train) + len(s.validation) == ex["total"]
    assert ex["denial_rate_excluded"] > 2 * ex["denial_rate_kept"]  # why they are excluded


def test_validation_is_a_stratified_slice_of_2023_only():
    s = make_splits(DEFAULT)
    assert set(s.validation["activity_year"]) == {"2023"}
    assert abs(len(s.validation) / (len(s.train) + len(s.validation)) - 0.2) < 0.001
    assert abs(s.validation.denied.mean() - s.train.denied.mean()) < 0.002


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.drop(columns=["action_taken"]), "missing columns"),
    (lambda d: d.assign(action_taken=d.action_taken.where(d.index != 0, "4")), "action_taken"),
    (lambda d: d.assign(activity_year=d.activity_year.where(d.split == "train", "2022")), "time split"),
])
def test_validation_rejects_bad_snapshots(mutate, msg):
    with pytest.raises(DatasetError, match=msg):
        validate(mutate(load_snapshot()))
