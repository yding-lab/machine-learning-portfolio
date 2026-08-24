"""FastAPI service for real-time model inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import ModelArtifactError, load_model


MODEL = None


class PredictionRequest(BaseModel):
    """Flexible request body keyed by original feature names."""

    features: dict[str, float] = Field(
        ...,
        min_length=1,
        description="Mapping of model feature names to numeric values.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: int
    probability_positive: float
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    global MODEL
    try:
        MODEL = load_model()
    except ModelArtifactError:
        # Service can still start so /health explains the artifact state.
        MODEL = None
    yield
    MODEL = None


app = FastAPI(
    title="Production ML Portfolio API",
    version="1.0.0",
    description=(
        "A lightweight inference service around a versioned sklearn Pipeline."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness endpoint."""
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one observation with the trained model."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Train the model first.",
        )

    try:
        frame = pd.DataFrame([request.features])
        probability = float(MODEL.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature payload: {exc}",
        ) from exc

    return PredictionResponse(
        predicted_class=int(probability >= request.threshold),
        probability_positive=probability,
        threshold=request.threshold,
    )
