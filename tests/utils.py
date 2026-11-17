"""Common helpers for pytest and shell-backed project checks."""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tests.settings import CONFIG


def bash(
    cmd: str | list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    live_output: bool = False,
) -> str:
    """Run a shell command, returning combined output or failing the pytest test."""
    command = ["bash", "-lc", cmd] if isinstance(cmd, str) else cmd
    started = time.monotonic()
    label = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f"\n[e2e] START {label} (timeout: {timeout}s)", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env or os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout or ""
    if live_output and output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode:
        pytest.fail(
            f"Command failed ({completed.returncode}): {label}\n"
            f"Working directory: {cwd}\nOutput:\n{output}",
            pytrace=False,
        )
    print(f"[e2e] DONE  {label} ({time.monotonic() - started:.1f}s)", flush=True)
    return output


def assert_paths_exist(root: Path, paths: list[str], checkpoint: str) -> None:
    missing = [path for path in paths if not (root / path).is_file()]
    assert not missing, f"Missing {checkpoint} checkpoint files: {missing}"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((CONFIG["api"]["host"], 0))
        return int(sock.getsockname()[1])
