"""FastAPI service for real-time model inference.

    uvicorn ml_portfolio.api:app --port 8000      # docs at http://127.0.0.1:8000/docs

Educational benchmark only: the model is not clinically validated and must not be used
for medical decisions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import MODEL_PATH, POSITIVE_CLASS
from .inference import (
    FeatureValidationError,
    ModelArtifactError,
    expected_features,
    load_metadata,
    load_model,
    predict_one,
)

STATE: dict[str, Any] = {"model": None, "features": [], "metadata": {}, "error": None}


class PredictionRequest(BaseModel):
    """Request body keyed by the original feature names (all 30 are required)."""

    features: dict[str, float] = Field(
        ...,
        min_length=1,
        description="Mapping of model feature names to numeric values. GET /model lists the names.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: int = Field(description="1 = benign, 0 = malignant (scikit-learn encoding)")
    predicted_label: str
    probability_positive: float = Field(description=f"Probability of the positive class ({POSITIVE_CLASS})")
    threshold: float
    disclaimer: str = "Educational benchmark model. Not for medical use."


def load_state(path=None) -> None:
    """Load the model once; on failure keep serving so /ready can say why."""
    try:
        model = load_model(path)
        STATE.update(model=model, features=expected_features(model), metadata=load_metadata(), error=None)
    except ModelArtifactError as exc:
        STATE.update(model=None, features=[], metadata={}, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    load_state()
    yield
    STATE.update(model=None, features=[], metadata={})


app = FastAPI(
    title="Breast-cancer benchmark classifier (educational)",
    version="1.1.0",
    description=(
        "A lightweight inference service around a versioned sklearn Pipeline trained on the "
        "scikit-learn Breast Cancer Wisconsin dataset. Educational benchmark only: not clinically "
        "validated, not for medical decisions."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness: the process is up. Says whether a model is loaded, but never fails on that."""
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/ready")
def ready():
    """Readiness: 200 only when a model is loaded and can serve predictions; 503 otherwise."""
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={
            "ready": False, "model_loaded": False, "model_path": str(MODEL_PATH), "reason": STATE["error"]})
    meta = STATE["metadata"]
    return {"ready": True, "model_loaded": True, "selected_model": meta.get("selected_model"),
            "trained_at_utc": meta.get("trained_at_utc"), "n_features": len(STATE["features"])}


@app.get("/model")
def model_info() -> dict[str, Any]:
    """What the loaded model is: selection, metrics on the held-out split, expected features."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {STATE['error']}")
    return {**STATE["metadata"], "feature_names": STATE["features"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one observation with the trained model."""
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Run `python -m ml_portfolio.train` first.",
        )

    try:
        p = predict_one(STATE["model"], request.features, request.threshold)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), **exc.to_dict()}) from exc

    return PredictionResponse(
        predicted_class=p.predicted_class,
        predicted_label=p.predicted_label,
        probability_positive=p.probability_positive,
        threshold=p.threshold,
    )
