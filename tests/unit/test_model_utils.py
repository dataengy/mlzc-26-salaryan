"""Unit tests for cleaning, deterministic splitting, and model factory behavior."""
import pandas as pd
import pytest

from mlzc_26_salaryan.config import get_config
from mlzc_26_salaryan.model_utils import FEATURES, REQUIRED, TARGET, clean_data, make_model
from mlzc_26_salaryan.train import split_data


def _row():
    row = {feature: "  value  " for feature in FEATURES}
    row.update({TARGET: "72000", "work_year": "2024"})
    return row


def test_clean_data_drops_invalid_and_duplicate_rows_and_normalizes_categories():
    valid = _row()
    duplicate = dict(valid)
    invalid_salary = dict(valid, salary_in_usd="not-a-number")
    too_high = dict(valid, salary_in_usd=2_000_000)
    blank_category = dict(valid, job_title="  ")

    result = clean_data(pd.DataFrame([valid, duplicate, invalid_salary, too_high, blank_category]))

    assert len(result) == 2
    assert result[TARGET].tolist() == [72000, 72000]
    assert result.iloc[0][FEATURES[0]] == "value"
    assert result.iloc[1]["job_title"] == get_config()["data"]["unknown_category"]
    assert result["work_year"].dtype.kind in "iu"


def test_clean_data_reports_missing_required_columns():
    with pytest.raises(ValueError, match="Missing dataset columns"):
        clean_data(pd.DataFrame(columns=REQUIRED[:-1]))


def test_split_data_uses_deterministic_fallback_when_years_are_incomplete():
    frame = pd.DataFrame({"work_year": [2022] * 100, "value": range(100)})
    first = split_data(frame)
    second = split_data(frame)
    assert first[3] == "random 60/20/20 (fallback; document limitation)"
    assert [len(part) for part in first[:3]] == [60, 20, 20]
    assert [part["value"].tolist() for part in first[:3]] == [part["value"].tolist() for part in second[:3]]


def test_model_factory_rejects_unknown_candidate():
    with pytest.raises(ValueError, match="Unknown model type"):
        make_model("not-a-model")
