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

from .config import get_config, project_path
from .model_utils import FEATURES, TARGET, YEAR, clean_data, make_model

CONFIG = get_config()


def split_data(df: pd.DataFrame):
    # Prefer chronological train/validation/test; never evaluate on training data.
    split = CONFIG["model"]["temporal_split"]
    years = set(df[YEAR].unique())
    if {split["train_through_year"], split["validation_year"], split["test_from_year"]}.issubset(years):
        train = df[df[YEAR] <= split["train_through_year"]]
        validation = df[df[YEAR] == split["validation_year"]]
        test = df[df[YEAR] >= split["test_from_year"]]
        if min(map(len, (train, validation, test))) >= split["minimum_rows_per_partition"]:
            return train, validation, test, (
                f"temporal: <={split['train_through_year']} / "
                f"{split['validation_year']} / >={split['test_from_year']}"
            )
    # If source year distribution differs, prefer a documented deterministic fallback.
    fallback = CONFIG["model"]["fallback_split"]
    train_val, test = train_test_split(df, test_size=fallback["test_fraction"], random_state=fallback["seed"])
    validation_fraction_of_train_val = fallback["validation_fraction"] / (fallback["train_fraction"] + fallback["validation_fraction"])
    train, validation = train_test_split(train_val, test_size=validation_fraction_of_train_val, random_state=fallback["seed"])
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
    if len(df) < CONFIG["model"]["minimum_clean_rows"]:
        raise ValueError(f"Dataset too small after cleaning: {len(df)}")
    original_count = len(df)
    if len(df) > max_rows:
        # Deterministic sample speeds up validation for deadline-driven builds.
        df = df.sample(n=max_rows, random_state=CONFIG["model"]["sample_seed"]).reset_index(drop=True)
    train, validation, test, split_info = split_data(df)
    candidates = [
        (candidate["name"], candidate["kind"], candidate["param"])
        for candidate in CONFIG["model"]["candidates"]
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
    parser.add_argument("--data", type=Path, default=project_path("dataset"))
    parser.add_argument("--output", type=Path, default=project_path("model").parent)
    parser.add_argument("--max-rows", type=int, default=CONFIG["model"]["max_rows"])
    args = parser.parse_args()
    run(args.data, args.output, args.max_rows)
