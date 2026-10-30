import pytest
from fastapi.testclient import TestClient

from ml_portfolio.data import load_dataset


@pytest.fixture()
def client(trained):
    from ml_portfolio import api

    with TestClient(api.app) as c:
        yield c


@pytest.fixture()
def record():
    X, _ = load_dataset()
    return {k: float(v) for k, v in X.iloc[0].items()}


def test_health_and_ready_report_a_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] is True and r.json()["n_features"] == 30


def test_predict_returns_labelled_probability(client, record):
    r = client.post("/predict", json={"features": record})
    assert r.status_code == 200
    j = r.json()
    assert j["predicted_label"] in {"benign", "malignant"} and 0 <= j["probability_positive"] <= 1
    assert "Not for medical use" in j["disclaimer"]


def test_missing_and_unknown_features_get_a_clear_422(client, record):
    record.pop("mean radius")
    record["shoe size"] = 9.0
    r = client.post("/predict", json={"features": record})
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["missing"] == ["mean radius"] and d["unexpected"] == ["shoe size"]


def test_non_numeric_and_bad_threshold_are_rejected(client, record):
    assert client.post("/predict", json={"features": {**record, "mean radius": "abc"}}).status_code == 422
    assert client.post("/predict", json={"features": record, "threshold": 1.5}).status_code == 422


def test_not_ready_without_a_model(trained, tmp_path):
    from ml_portfolio import api

    with TestClient(api.app) as c:
        api.load_state(tmp_path / "missing.joblib")
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
        r = c.get("/ready")
        assert r.status_code == 503 and "not found" in r.json()["reason"]
        assert c.post("/predict", json={"features": {"a": 1.0}}).status_code == 503
