"""Reproducible ML training and holdout evaluation; no cloud required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from model_utils import FEATURES, TARGET, clean_data, make_model


def split_data(df: pd.DataFrame):
    # Prefer chronological train/validation/test; never evaluate on training data.
    years = set(df["work_year"].unique())
    if {2023, 2024, 2025}.issubset(years):
        train = df[df["work_year"] <= 2023]
        validation = df[df["work_year"] == 2024]
        test = df[df["work_year"] >= 2025]
        if min(map(len, (train, validation, test))) >= 20:
            return train, validation, test, "temporal: <=2023 / 2024 / >=2025"
    # If source year distribution differs, prefer a documented deterministic fallback.
    train_val, test = train_test_split(df, test_size=.20, random_state=42)
    train, validation = train_test_split(train_val, test_size=.25, random_state=42)
    return train, validation, test, "random 60/20/20 (fallback; document limitation)"


def metrics(model, df: pd.DataFrame) -> dict[str, float]:
    actual = df[TARGET].to_numpy()
    prediction = np.maximum(0, model.predict(df[FEATURES]))
    return {
        "mae_usd": round(float(mean_absolute_error(actual, prediction)), 2),
        "rmse_usd": round(float(np.sqrt(mean_squared_error(actual, prediction))), 2),
    }


def run(csv_path: Path, out_dir: Path, max_rows: int = 50_000) -> dict:
    raw = pd.read_csv(csv_path)
    df = clean_data(raw)
    if len(df) < 100:
        raise ValueError(f"Dataset too small after cleaning: {len(df)}")
    original_count = len(df)
    if len(df) > max_rows:
        # Deterministic sample speeds up validation for deadline-driven builds.
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    train, validation, test, split_info = split_data(df)
    candidates = [
        ("dummy_median", "dummy", 1),
        ("ridge_alpha_1", "ridge", 1),
        ("ridge_alpha_20", "ridge", 20),
        ("forest_depth_8", "forest", 8),
        ("forest_depth_12", "forest", 12),
    ]
    comparisons = []
    for name, kind, param in candidates:
        model = make_model(kind, param)
        model.fit(train[FEATURES], train[TARGET])
        comparisons.append({"candidate": name, "kind": kind, "param": param, **metrics(model, validation)})
    winner = min(comparisons, key=lambda r: r["mae_usd"])
    final = make_model(winner["kind"], winner["param"])
    combined = pd.concat([train, validation], ignore_index=True)
    final.fit(combined[FEATURES], combined[TARGET])
    test_metrics = metrics(final, test)   # test only after model choice
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_dir / "model.joblib")
    report = {
        "dataset_rows_clean": original_count, "used_rows": len(df),
        "split": split_info, "train_rows": len(train), "validation_rows": len(validation),
        "test_rows": len(test), "target": TARGET, "features": FEATURES,
        "validation_comparison": comparisons, "selected_model": winner,
        "holdout_test": test_metrics,
        "note": "Predicts annual USD compensation, not a guaranteed offer. Dataset may be nonrepresentative.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/salaries.csv"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--max-rows", type=int, default=50_000)
    args = parser.parse_args()
    run(args.data, args.output, args.max_rows)
