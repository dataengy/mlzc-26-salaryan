"""Load the repository's shared YAML configuration."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "settings" / "config.yml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")
    return config


def project_path(key: str) -> Path:
    """Resolve a configured project-relative path from the repository root."""
    return PROJECT_ROOT / get_config()["paths"][key]
