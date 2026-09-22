"""
Diabetes Risk Screening API
============================
Serves the pickled model bundle (model + scaler + label encoders) produced by
`diabetes_prediction_pipeline.ipynb` behind a small, well-typed FastAPI service,
plus a static single-page frontend for manual testing / demos.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload

Then open http://127.0.0.1:8000 in a browser (the form UI),
or http://127.0.0.1:8000/docs for interactive Swagger API docs.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, create_model

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = BASE_DIR / "diabetes_model_bundle.pkl"
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diabetes-api")

# ----------------------------------------------------------------------------
# Load the model bundle once at startup
# ----------------------------------------------------------------------------

if not MODEL_BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Model bundle not found at '{MODEL_BUNDLE_PATH}'. "
        "Run the training notebook first and place "
        "'diabetes_model_bundle.pkl' next to this file."
    )

with open(MODEL_BUNDLE_PATH, "rb") as f:
    bundle: dict[str, Any] = pickle.load(f)

model = bundle["model"]
model_name: str = bundle["model_name"]
scaler = bundle["scaler"]
label_encoders: dict[str, Any] = bundle["label_encoders"]
numerical_cols: list[str] = bundle["numerical_cols"]
categorical_cols: list[str] = bundle["categorical_cols"]
feature_order: list[str] = bundle["feature_order"]

# Known category choices per categorical column, taken straight from the
# encoders that were fit during training -- this is what the frontend uses
# to render dropdowns, and what the backend uses to validate incoming values.
categorical_choices: dict[str, list[str]] = {
    col: list(label_encoders[col].classes_) for col in categorical_cols
}

logger.info("Loaded model bundle: %s", model_name)
logger.info("Numerical features (%d): %s", len(numerical_cols), numerical_cols)
logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)

# ----------------------------------------------------------------------------
# Dynamically build the request schema from the bundle, so the API always
# matches exactly what the model was trained on -- no hand-maintained,
# easy-to-drift column list.
# ----------------------------------------------------------------------------

_pydantic_fields: dict[str, tuple[type, Any]] = {}
for col in numerical_cols:
    _pydantic_fields[col] = (float, Field(..., description=f"Numerical feature: {col}"))
for col in categorical_cols:
    _pydantic_fields[col] = (
        str,
        Field(..., description=f"Categorical feature: {col}. One of {categorical_choices[col]}"),
    )

PatientFeatures: type[BaseModel] = create_model("PatientFeatures", **_pydantic_fields)  # type: ignore[call-overload]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Diabetes, 1 = Diabetes")
    prediction_label: str
    probability_diabetes: float = Field(..., ge=0.0, le=1.0)
    risk_band: str
    model_used: str


class SchemaResponse(BaseModel):
    numerical_cols: list[str]
    categorical_cols: dict[str, list[str]]
    feature_order: list[str]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    n_features: int


# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------

app = FastAPI(
    title="Diabetes Risk Screening API",
    description=(
        "Predicts diabetes risk from clinical and lifestyle features using a "
        f"trained {model_name} model. This is a screening aid, not a diagnostic "
        "tool -- results should always be interpreted by a qualified clinician."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _risk_band(probability: float) -> str:
    """Bucket a raw probability into a human-readable risk band."""
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "High"
    return "Very High"


def _encode_and_scale(payload: dict[str, Any]) -> pd.DataFrame:
    """Turn a raw feature dict into a single-row, model-ready DataFrame."""
    row = pd.DataFrame([payload])

    # Encode categoricals using the SAME encoders fit during training.
    for col in categorical_cols:
        encoder = label_encoders[col]
        value = str(row.at[0, col])
        if value not in encoder.classes_:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid value '{value}' for field '{col}'. "
                    f"Expected one of: {list(encoder.classes_)}"
                ),
            )
        row[col] = encoder.transform([value])

    # Scale numericals using the SAME scaler fit during training.
    row[numerical_cols] = scaler.transform(row[numerical_cols])

    # Ensure column order exactly matches what the model was trained on.
    return row[feature_order]


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Simple liveness/readiness check."""
    return HealthResponse(status="ok", model_name=model_name, n_features=len(feature_order))


@app.get("/schema", response_model=SchemaResponse, tags=["Meta"])
def schema() -> SchemaResponse:
    """
    Describes exactly which fields /predict expects, and the valid choices
    for every categorical field. The frontend uses this to render its form.
    """
    return SchemaResponse(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_choices,
        feature_order=feature_order,
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: PatientFeatures) -> PredictionResponse:  # type: ignore[valid-type]
    """
    Run a single-patient diabetes risk prediction.

    Send one value per feature listed in GET /schema. Categorical fields must
    use one of the exact choices returned by /schema for that field.
    """
    payload = features.model_dump()

    try:
        X_row = _encode_and_scale(payload)
        proba = float(model.predict_proba(X_row)[0, 1])
        pred = int(model.predict(X_row)[0])
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        prediction=pred,
        prediction_label="Diabetes" if pred == 1 else "No Diabetes",
        probability_diabetes=round(proba, 4),
        risk_band=_risk_band(proba),
        model_used=model_name,
    )


# ----------------------------------------------------------------------------
# Static frontend
# ----------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["Meta"])
def index() -> FileResponse:
    """Serves the single-page demo UI."""
    return FileResponse(STATIC_DIR / "index.html")
