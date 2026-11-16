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

from .config import get_config

PROJECT_CONFIG = get_config()
TARGET = PROJECT_CONFIG["data"]["target"]
YEAR = PROJECT_CONFIG["data"]["year"]
FEATURES = PROJECT_CONFIG["model"]["features"]
REQUIRED = FEATURES + [TARGET, YEAR]


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    df = frame[REQUIRED].copy()
    data_config = PROJECT_CONFIG["data"]
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df[YEAR] = pd.to_numeric(df[YEAR], errors="coerce")
    df = df.dropna(subset=[TARGET, YEAR])
    df = df[(df[TARGET] > data_config["minimum_salary_usd"]) & (df[TARGET] < data_config["maximum_salary_usd"])]
    df[YEAR] = df[YEAR].astype(int)
    for col in FEATURES:
        unknown = data_config["unknown_category"]
        df[col] = df[col].fillna(unknown).astype(str).str.strip()
        df[col] = df[col].replace("", unknown)
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
