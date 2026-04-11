from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from .features import CropType, FEATURE_NAMES, build_feature_row

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
ONNX_PATH = MODEL_DIR / "model.onnx"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "agro-moisture-notebook")

app = FastAPI(
    title="Soil moisture (ONNX)",
    description="Serves the GradientBoostingRegressor exported from agro_moisture.ipynb.",
    version=MODEL_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_session: ort.InferenceSession | None = None
_scaler = None
_load_error: str | None = None


def _load_artifacts() -> None:
    global _session, _scaler, _load_error
    _load_error = None
    try:
        if not ONNX_PATH.is_file():
            raise FileNotFoundError(f"Missing {ONNX_PATH} — run all cells in agro_moisture.ipynb")
        if not SCALER_PATH.is_file():
            raise FileNotFoundError(f"Missing {SCALER_PATH} — run all cells in agro_moisture.ipynb")
        _session = ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"]
        )
        _scaler = joblib.load(SCALER_PATH)
    except Exception as e:  # noqa: BLE001
        _session = None
        _scaler = None
        _load_error = str(e)


@app.on_event("startup")
def startup() -> None:
    _load_artifacts()


class PredictRequest(BaseModel):
    soil_ph: float = Field(..., ge=0, le=14)
    temperature_c: float = Field(..., ge=-50, le=60)
    humidity_pct: float = Field(..., ge=0, le=100)
    fertilizer_kg_ha: float = Field(..., ge=0)
    irrigation_mm: float = Field(..., ge=0)
    month: int = Field(..., ge=1, le=12)
    crop: CropType
    moisture_lag1: float = Field(..., ge=0, le=100)
    moisture_lag2: float = Field(..., ge=0, le=100)
    moisture_rolling_mean3: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    soil_moisture_percent: float
    model_version: str
    latency_ms: float
    n_features: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    ready: bool
    feature_order: list[str]
    detail: str | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok = _session is not None and _scaler is not None
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_version=MODEL_VERSION,
        ready=ok,
        feature_order=list(FEATURE_NAMES),
        detail=_load_error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if _session is None or _scaler is None:
        raise HTTPException(status_code=503, detail=_load_error or "Artifacts not loaded")
    t0 = time.perf_counter()
    raw = build_feature_row(
        soil_ph=body.soil_ph,
        temperature_c=body.temperature_c,
        humidity_pct=body.humidity_pct,
        fertilizer_kg_ha=body.fertilizer_kg_ha,
        irrigation_mm=body.irrigation_mm,
        month=body.month,
        crop=body.crop,
        moisture_lag1=body.moisture_lag1,
        moisture_lag2=body.moisture_lag2,
        moisture_rolling_mean3=body.moisture_rolling_mean3,
    )
    scaled = _scaler.transform(raw)
    name = _session.get_inputs()[0].name
    out = _session.run(None, {name: scaled.astype(np.float32)})[0]
    pred = float(np.asarray(out).reshape(-1)[0])
    ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        soil_moisture_percent=pred,
        model_version=MODEL_VERSION,
        latency_ms=round(ms, 3),
        n_features=len(FEATURE_NAMES),
    )


@app.post("/reload")
def reload() -> dict:
    _load_artifacts()
    return {"ok": _session is not None, "detail": _load_error}
