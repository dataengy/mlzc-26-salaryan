"""Fast checks for imports, shared settings, and the checked-in service artifact."""
import json

from fastapi.testclient import TestClient

from mlzc_26_salaryan import app, config, model_utils, train
from mlzc_26_salaryan.app import app as fastapi_app, load_model


def test_core_modules_import():
    assert callable(train.run)
    assert callable(model_utils.clean_data)
    assert callable(app.health)


def test_yaml_settings_and_config_cover_repository_layout():
    project = config.get_config()
    assert project["project"]["distribution"] == "mlzc-26-salaryan"
    assert project["project"]["package"] == "mlzc_26_salaryan"
    assert config.project_path("model").is_file()
    assert config.project_path("request_example").is_file()


def test_checked_in_model_health_and_prediction():
    project = config.get_config()
    payload = json.loads(config.project_path("request_example").read_text(encoding="utf-8"))
    client = TestClient(fastapi_app)
    load_model.cache_clear()
    try:
        health = client.get(project["api"]["health_path"])
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        response = client.post(project["api"]["predict_path"], json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["predicted_salary_usd_annual"] >= 0
        assert result["unit"] == "USD/year"
        assert result["disclaimer"]
    finally:
        load_model.cache_clear()
