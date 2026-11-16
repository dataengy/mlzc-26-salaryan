"""Shared contract for offline training and live inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

TARGET = "salary_in_usd"
FEATURES = [
    "experience_level", "employment_type", "job_title",
    "employee_residence", "remote_ratio", "company_location", "company_size",
]
REQUIRED = FEATURES + [TARGET, "work_year"]


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    df = frame[REQUIRED].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["work_year"] = pd.to_numeric(df["work_year"], errors="coerce")
    df = df.dropna(subset=[TARGET, "work_year"])
    df = df[(df[TARGET] > 5000) & (df[TARGET] < 1_000_000)]
    df["work_year"] = df["work_year"].astype(int)
    for col in FEATURES:
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()
        df[col] = df[col].replace("", "UNKNOWN")
    return df.drop_duplicates().reset_index(drop=True)


def make_model(kind: str, param: float | int = 1):
    if kind == "dummy":
        return DummyRegressor(strategy="median")
    encoding = ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore", min_frequency=5), FEATURES)],
        remainder="drop",
    )
    if kind == "ridge":
        estimator = Ridge(alpha=float(param), solver="lsqr")
    elif kind == "forest":
        estimator = RandomForestRegressor(
            n_estimators=40, max_depth=int(param), min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {kind}")
    pipeline = Pipeline([("encode", encoding), ("model", estimator)])
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1,
        check_inverse=False,
    )
