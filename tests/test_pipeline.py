"""Offline tests on deliberately synthetic fixture (never a submission dataset)."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi.testclient import TestClient

from mlzoomcamp_salary_minimal.train import run
from mlzoomcamp_salary_minimal.app import app, load_model


def test_training_and_api(tmp_path, monkeypatch):
    rows = []
    rng = np.random.default_rng(42)
    for year in (2022, 2023, 2024, 2025):
        for j in range(35):
            level = ("EN", "MI", "SE", "EX")[j % 4]
            rows.append({"work_year": year,
                "experience_level": level,
                "employment_type": "FT",
                "job_title": ("Data Engineer", "Data Scientist")[j % 2],
                "salary": 50_000, "salary_currency": "USD",
                "salary_in_usd": 50_000 + (j % 4) * 22_000 + (year - 2022) * 3000 + float(rng.normal(0, 2000)),
                "employee_residence": "GE",
                "remote_ratio": 100,
                "company_location": "DE",
                "company_size": "M"})
    dataset = tmp_path / "salaries.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    artifacts = tmp_path / "models"
    report = run(dataset, artifacts, max_rows=1000)
    assert report["split"].startswith("temporal")
    assert len(report["validation_comparison"]) == 5
    assert report["holdout_test"]["mae_usd"] >= 0
    assert (artifacts / "model.joblib").is_file()
    assert joblib.load(artifacts / "model.joblib")

    monkeypatch.setenv("MODEL_PATH", str(artifacts / "model.joblib"))
    load_model.cache_clear()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    payload = {"job_title":"Data Engineer", "experience_level":"SE",
        "employment_type":"FT", "employee_residence":"GE", "remote_ratio":100,
        "company_location":"DE", "company_size":"M"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_salary_usd_annual"] >= 0
    assert client.post("/predict", json={"remote_ratio": 999}).status_code == 422
    load_model.cache_clear()
