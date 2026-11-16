#!/usr/bin/env bash
set -euo pipefail
make data
make eda
make train
make test
make e2e
curl -fsS http://localhost:8080/health || echo 'API not running: start `make serve`, then retry health/predict requests'
