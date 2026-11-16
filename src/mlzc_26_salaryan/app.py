"""Minimal FastAPI prediction service."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_config, project_path

CONFIG = get_config()
APP = FastAPI(title="ML Zoomcamp AI Salary Predictor", version="1.0")


class SalaryRequest(BaseModel):
    job_title: str = Field(min_length=1)
    experience_level: str = Field(min_length=1)
    employment_type: str = Field(min_length=1)
    employee_residence: str = Field(min_length=1)
    remote_ratio: int = Field(ge=0, le=100)
    company_location: str = Field(min_length=1)
    company_size: str = Field(min_length=1)


@lru_cache(maxsize=1)
def load_model():
    model_path = Path(os.getenv("MODEL_PATH", str(project_path("model"))))
    return joblib.load(model_path)


@APP.get("/health")
def health():
    model_path = Path(os.getenv("MODEL_PATH", str(project_path("model"))))
    if not model_path.is_file():
        raise HTTPException(status_code=503, detail="Model artifact not found")
    return {"status": "ok"}


@APP.post("/predict")
def predict(payload: SalaryRequest):
    try:
        model = load_model()
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=503, detail="Model not trained")
    data = pd.DataFrame([payload.model_dump()])
    # Training stores every input category as text, including remote_ratio.
    data = data.astype(str)
    predicted = float(np.maximum(0, model.predict(data)[0]))
    return {"predicted_salary_usd_annual": round(predicted, 2), "unit": "USD/year", "disclaimer": "Educational estimate, not a real job offer"}

# Backwards-compatible app name for uvicorn.
app = APP
