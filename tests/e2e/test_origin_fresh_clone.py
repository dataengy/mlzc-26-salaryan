"""Clone the configured origin into .run and validate a full clean-checkout workflow."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml

from tests.settings import CONFIG, E2E, PROJECT_ROOT, SETTINGS
from tests.utils import assert_paths_exist, bash, free_port


def _bash(
    cmd: str | list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = E2E["command_timeout_seconds"],
    live_output: bool | None = None,
) -> str:
    if live_output is None:
        live_output = os.environ.get("E2E_LIVE_OUTPUT", "").lower() in {"1", "true", "yes"}
    return bash(cmd, cwd=cwd, env=env, timeout=timeout, live_output=live_output)


def _check_live_api(root: Path, env: dict[str, str]) -> None:
    host = CONFIG["api"]["host"]
    port = free_port()
    base = f"http://{host}:{port}"
    package = CONFIG["project"]["package"]
    log_path = root / ".run" / "e2e-api.log"
    command = [
        str(root / ".run" / "venv" / "bin" / "python"), "-m", "uvicorn",
        f"{package}.app:app", "--host", host, "--port", str(port),
    ]
    process = subprocess.Popen(command, cwd=root, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 45
        health = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urlopen(base + CONFIG["api"]["health_path"], timeout=2) as response:
                    health = json.loads(response.read())
                break
            except (OSError, URLError):
                time.sleep(0.5)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        assert health == {"status": "ok"}, f"API health failed; process={process.poll()}\n{log_text}"

        request_body = (root / CONFIG["paths"]["request_example"]).read_bytes()
        request = Request(
            base + CONFIG["api"]["predict_path"],
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            prediction = json.loads(response.read())
        assert prediction["predicted_salary_usd_annual"] >= 0
        assert prediction["unit"] == "USD/year"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.e2e
def test_origin_clone_runs_all_project_checkpoints_from_scratch():
    remote_name = E2E["origin_remote"]
    remote = subprocess.run(
        ["git", "remote", "get-url", remote_name],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if remote.returncode:
        pytest.skip(f"Git remote '{remote_name}' is not configured; set it before running the origin-clone e2e")
    if not shutil.which("uv"):
        pytest.fail("uv is required for fresh-clone dependency installation")
    if E2E.get("include_docker", True) and not shutil.which("docker"):
        pytest.fail("Docker is required because settings.yml enables the Docker e2e checkpoint")

    ref_env = E2E["git_ref_env"]
    git_ref = os.environ.get(ref_env)
    if not git_ref:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
            text=True, stdout=subprocess.PIPE, check=True,
        ).stdout.strip()
        git_ref = branch or "main"

    workspace = PROJECT_ROOT / SETTINGS["runtime"]["e2e_workspace"]
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="origin-clone-", dir=workspace) as temporary:
        checkout = Path(temporary) / CONFIG["project"]["distribution"]
        clone_command = ["git", "clone", "--depth", "1", "--branch", git_ref, remote.stdout.strip(), str(checkout)]
        _bash(
            clone_command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            timeout=E2E["clone_timeout_seconds"],
        )

        config = yaml.safe_load((checkout / "settings" / "settings.yml").read_text(encoding="utf-8"))
        e2e = config["quality_gates"]["e2e"]
        runtime = config["runtime"]
        run_env = os.environ.copy()
        run_env.update({
            "UV_CACHE_DIR": str(checkout / runtime["uv_cache"]),
            "UV_PROJECT_ENVIRONMENT": str(checkout / runtime["venv"]),
            "MPLCONFIGDIR": str(checkout / runtime["matplotlib_cache"]),
            "PYTHONPYCACHEPREFIX": str(checkout / runtime["python_cache"]),
        })

        commands = e2e["commands"]
        uv_sync = commands[0]
        _bash(uv_sync, cwd=checkout, env=run_env, timeout=e2e["command_timeout_seconds"])
        gate_commands = commands[1:5]
        for command in gate_commands:
            _bash(command, cwd=checkout, env=run_env, timeout=e2e["command_timeout_seconds"])
            checkpoint_map = e2e["checkpoints"]
            if command == ["make", "data"]:
                expected = checkpoint_map.get("data", ["data/salaries.csv", "data/source.json"])
                assert_paths_exist(checkout, expected, "download")
            elif command == ["make", "eda"]:
                expected = checkpoint_map.get("eda", ["reports/EDA.md", "reports/salary_histogram.png", "reports/feature_importance.csv", "reports/feature_importance.png"])
                assert_paths_exist(checkout, expected, "EDA")
            elif command == ["make", "train"]:
                expected = checkpoint_map.get("train", ["models/model.joblib", "models/metrics.json"])
                assert_paths_exist(checkout, expected, "training")

        _check_live_api(checkout, run_env)
        if e2e.get("include_docker", True):
            docker_command = commands[5]
            docker_output = _bash(
                docker_command,
                cwd=checkout,
                env=run_env,
                timeout=e2e["command_timeout_seconds"],
            )
            # `make docker-check` prints curl responses; tolerate whitespace and
            # line breaks between JSON tokens in its captured output.
            assert '"status"' in docker_output and '"ok"' in docker_output
            assert '"predicted_salary_usd_annual"' in docker_output
