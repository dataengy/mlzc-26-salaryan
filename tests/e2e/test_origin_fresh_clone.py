"""Clone the configured origin into .run and validate a full clean-checkout workflow."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((PROJECT_ROOT / "settings" / "config.yml").read_text(encoding="utf-8"))
SETTINGS = yaml.safe_load((PROJECT_ROOT / "settings" / "settings.yml").read_text(encoding="utf-8"))
E2E = SETTINGS["quality_gates"]["e2e"]


def _run(command: list[str], cwd: Path, env: dict[str, str], timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout
    if completed.returncode:
        pytest.fail(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"Working directory: {cwd}\nOutput:\n{output}",
            pytrace=False,
        )
    return output


def _assert_paths_exist(root: Path, paths: list[str], checkpoint: str) -> None:
    missing = [path for path in paths if not (root / path).is_file()]
    assert not missing, f"Missing {checkpoint} checkpoint files: {missing}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((CONFIG["api"]["host"], 0))
        return int(sock.getsockname()[1])


def _check_live_api(root: Path, env: dict[str, str]) -> None:
    host = CONFIG["api"]["host"]
    port = _free_port()
    base = f"http://{host}:{port}"
    package = CONFIG["project"]["package"]
    log_path = root / ".run" / "e2e-api.log"
    command = [
        str(root / ".venv" / "bin" / "python"), "-m", "uvicorn",
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
        _run(clone_command, PROJECT_ROOT, os.environ.copy(), E2E["clone_timeout_seconds"])

        config = yaml.safe_load((checkout / "settings" / "settings.yml").read_text(encoding="utf-8"))
        e2e = config["quality_gates"]["e2e"]
        runtime = config["runtime"]
        run_env = os.environ.copy()
        run_env.update({
            "UV_CACHE_DIR": str(checkout / runtime["uv_cache"]),
            "MPLCONFIGDIR": str(checkout / runtime["matplotlib_cache"]),
            "PYTHONPYCACHEPREFIX": str(checkout / runtime["python_cache"]),
        })

        commands = e2e["commands"]
        uv_sync = commands[0]
        _run(uv_sync, checkout, run_env, e2e["command_timeout_seconds"])
        gate_commands = commands[1:5]
        for command in gate_commands:
            _run(command, checkout, run_env, e2e["command_timeout_seconds"])
            if command == ["make", "data"]:
                _assert_paths_exist(checkout, e2e["checkpoints"]["data"], "download")
            elif command == ["make", "eda"]:
                _assert_paths_exist(checkout, e2e["checkpoints"]["eda"], "EDA")
            elif command == ["make", "train"]:
                _assert_paths_exist(checkout, e2e["checkpoints"]["train"], "training")

        _check_live_api(checkout, run_env)
        if e2e.get("include_docker", True):
            docker_command = commands[5]
            docker_output = _run(docker_command, checkout, run_env, e2e["command_timeout_seconds"])
            assert '"status":"ok"' in docker_output or '"status": "ok"' in docker_output
            assert '"predicted_salary_usd_annual"' in docker_output
