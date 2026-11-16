"""Fast checks for imports, shared settings, and the checked-in service artifact."""
from fastapi.testclient import TestClient

from mlzc_26_salaryan import app, config, model_utils, train
from mlzc_26_salaryan.app import app as fastapi_app


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


def test_checked_in_model_health():
    project = config.get_config()
    client = TestClient(fastapi_app)
    health = client.get(project["api"]["health_path"])
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
