"""Shared project and quality-gate settings for tests."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((PROJECT_ROOT / "settings" / "config.yml").read_text(encoding="utf-8"))
SETTINGS = yaml.safe_load((PROJECT_ROOT / "settings" / "settings.yml").read_text(encoding="utf-8"))
E2E = SETTINGS["quality_gates"]["e2e"]
