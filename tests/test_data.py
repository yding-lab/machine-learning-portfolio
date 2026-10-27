import pandas as pd
import pytest

from ml_portfolio.data import DatasetValidationError, load_dataset, make_train_test_split, validate_dataset


def test_dataset_loads_and_validates():
    X, y = load_dataset()
    validate_dataset(X, y)
    assert X.shape == (569, 30) and set(y.unique()) == {0, 1}


def test_split_is_stratified_and_disjoint():
    b = make_train_test_split()
    assert len(b.X_train) == 455 and len(b.X_test) == 114
    assert set(b.X_train.index).isdisjoint(b.X_test.index)
    assert abs(b.y_train.mean() - b.y_test.mean()) < 0.02


@pytest.mark.parametrize("mutate, message", [
    (lambda X, y: (X.iloc[:0], y.iloc[:0]), "empty"),
    (lambda X, y: (X, y.iloc[:-1]), "lengths differ"),
    (lambda X, y: (X.assign(**{"mean radius": None}), y), "Missing feature values"),
    (lambda X, y: (X, y.replace(1, 2)), "binary"),
    (lambda X, y: (X, pd.Series([0] * len(y), name="target")), "Both classes"),
])
def test_validation_rejects_bad_data(mutate, message):
    X, y = load_dataset()
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(*mutate(X, y))
