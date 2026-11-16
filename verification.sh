#!/usr/bin/env bash
set -euo pipefail
python scripts/download.py
python eda.py
python train.py
pytest -q
curl -fsS http://localhost:8080/health || echo 'API not running: start uvicorn, then retry health/predict requests'
